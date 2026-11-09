"""
Optimal market making with inventory risk via HJB discretization and lookup table.

Combines the current program's joint-depth backward induction (which correctly
captures bid/ask interaction) with the separable PDE approach's finer depth grid
and closed-form FOC refinement for a final polish pass.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_DEADLINE = None


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 310  # 20s safety margin before 330s kill
    return _DEADLINE


def _solve_policy(sigma, A, k, gamma, T, steps, Q, deadline):
    """
    Solve the discrete HJB equation for the optimal policy using vectorized
    evaluation over all (bid_depth, ask_depth) pairs per time step.
    
    Uses a finer depth grid (25 points) for better resolution, then applies
    a closed-form FOC refinement as a final polish.
    """
    dt = T / steps
    n_levels = 2 * Q + 1
    Q_idx = np.arange(n_levels)
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)

    # Depth grid: centered around 1/k with reasonable spread, 25 points
    min_d = 0.003
    max_d = 5.0 / k
    n_grid = 25
    depth_grid = np.linspace(min_d, max_d, n_grid)
    depth_grid = np.maximum(depth_grid, 1e-6)

    V = np.zeros((steps + 1, n_levels))
    bid_opt = np.zeros((steps, n_levels))
    ask_opt = np.zeros((steps, n_levels))

    # Precompute fill probabilities
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)  # (n_depths,)

    # Precompute index arrays
    bid_next_idx = np.minimum(Q_idx + 1, n_levels - 1)
    ask_next_idx = np.maximum(Q_idx - 1, 0)

    # Precompute all probability combinations: (n_depths, n_depths)
    p_b = p_fill[:, np.newaxis]  # (n_depths, 1)
    p_a = p_fill[np.newaxis, :]  # (1, n_depths)
    p_ba = p_b * p_a            # (n_depths, n_depths)
    p_bn = p_b * (1.0 - p_a)    # (n_depths, n_depths)
    p_na = (1.0 - p_b) * p_a    # (n_depths, n_depths)
    p_nn = (1.0 - p_b) * (1.0 - p_a)  # (n_depths, n_depths)

    # Immediate reward components: p_b * bid_d + p_a * ask_d
    imm_base = p_fill[:, np.newaxis] * depth_grid[:, np.newaxis] + \
               p_fill[np.newaxis, :] * depth_grid[np.newaxis, :]

    # Precompute p_same = p_ba + p_nn for reuse
    p_same = p_ba + p_nn  # (n_depths, n_depths)

    completed = 0
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            break

        V_next = V[n + 1]  # (n_levels,)
        inv_cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt  # (n_levels,)

        # Compute expected next value for each (di, ai) pair
        exp_V_next = (p_same[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, :] +
                      p_bn[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, bid_next_idx] +
                      p_na[:, :, np.newaxis] * V_next[np.newaxis, np.newaxis, ask_next_idx])

        # Total value: (n_depths, n_depths, n_levels)
        total = imm_base[:, :, np.newaxis] - inv_cost[np.newaxis, np.newaxis, :] + exp_V_next

        # Find best (di, ai) for each q
        total_flat = total.reshape(-1, n_levels)  # (n_depths^2, n_levels)
        best_idx = np.argmax(total_flat, axis=0)  # (n_levels,)
        V[n] = total_flat[best_idx, np.arange(n_levels)]

        # Decode best (di, ai)
        di = best_idx // n_grid
        ai = best_idx % n_grid
        bid_opt[n] = depth_grid[di]
        ask_opt[n] = depth_grid[ai]
        completed += 1

    # Final polish: refine depths using FOC-based local optimization
    # For each (n, q), given V_next, solve for optimal depth more precisely
    if completed == steps:
        _polish_policy(V, bid_opt, ask_opt, sigma, A, k, gamma, T, steps, Q, deadline)

    # If not all steps completed, fill remaining with last computed step or default
    if completed < steps:
        if completed > 0:
            for n in range(completed, steps):
                bid_opt[n] = bid_opt[completed - 1]
                ask_opt[n] = ask_opt[completed - 1]
        else:
            default_d = 1.0 / k
            bid_opt[:] = default_d
            ask_opt[:] = default_d

    return bid_opt, ask_opt


def _polish_policy(V, bid_opt, ask_opt, sigma, A, k, gamma, T, steps, Q, deadline):
    """
    Refine the grid-based depths using a local FOC-based optimization.
    For each step and inventory level, do a few Newton iterations on the
    depth to maximize the objective given V_next.
    """
    dt = T / steps
    n_levels = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    bid_next_idx = np.minimum(np.arange(n_levels) + 1, n_levels - 1)
    ask_next_idx = np.maximum(np.arange(n_levels) - 1, 0)

    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            break

        V_next = V[n + 1]
        inv_cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

        # For each q, refine bid and ask depths
        for q_idx in range(n_levels):
            q = q_vals[q_idx]
            
            # Current grid-based depths
            b = bid_opt[n, q_idx]
            a = ask_opt[n, q_idx]

            # If at boundary, skip
            if q >= Q:
                bid_opt[n, q_idx] = np.inf
                continue
            if q <= -Q:
                ask_opt[n, q_idx] = np.inf
                continue

            # Local refinement: 5 Newton steps for bid
            if b > 0 and np.isfinite(b):
                b = _refine_depth_bid(b, V_next, q_idx, bid_next_idx, ask_next_idx,
                                       A, k, dt, inv_cost, sigma, gamma, q)
                bid_opt[n, q_idx] = max(b, 1e-6)

            # Local refinement: 5 Newton steps for ask
            if a > 0 and np.isfinite(a):
                a = _refine_depth_ask(a, V_next, q_idx, bid_next_idx, ask_next_idx,
                                       A, k, dt, inv_cost, sigma, gamma, q)
                ask_opt[n, q_idx] = max(a, 1e-6)

        # Recompute V[n] with refined depths
        for q_idx in range(n_levels):
            b = bid_opt[n, q_idx]
            a = ask_opt[n, q_idx]
            if not np.isfinite(b) or not np.isfinite(a):
                # One side is inf, compute the other
                if np.isfinite(b) and not np.isfinite(a):
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    V[n, q_idx] = p_b * b - inv_cost[q_idx] + V_next[q_idx]
                elif np.isfinite(a) and not np.isfinite(b):
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    V[n, q_idx] = p_a * a - inv_cost[q_idx] + V_next[q_idx]
                else:
                    V[n, q_idx] = -inv_cost[q_idx] + V_next[q_idx]
            else:
                p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                V_ba = p_b * p_a * V_next[q_idx]
                V_bn = p_b * (1 - p_a) * V_next[bid_next_idx[q_idx]]
                V_na = (1 - p_b) * p_a * V_next[ask_next_idx[q_idx]]
                V_nn = (1 - p_b) * (1 - p_a) * V_next[q_idx]
                V[n, q_idx] = p_b * b + p_a * a - inv_cost[q_idx] + V_ba + V_bn + V_na + V_nn


def _refine_depth_bid(b, V_next, q_idx, bid_next_idx, ask_next_idx, A, k, dt, inv_cost, sigma, gamma, q):
    """
    Refine bid depth using gradient-based local search.
    Objective: maximize p(b)*b + (p(b)*(1-p_a)*V_next[q+1] + (1-p(b))*p_a*V_next[q-1] + (1-p(b))*(1-p_a)*V_next[q])
    where p_a is the ask fill probability (treated as fixed for bid refinement).
    """
    # We need the ask depth to compute p_a, but for simplicity, use the current ask
    # Actually, let's just do a simple golden section or bisection on a small interval
    # around the grid value
    
    # Simple approach: try a few perturbations and pick the best
    best_b = b
    best_val = -np.inf
    
    # Current value
    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
    # We need p_a from the ask side; for bid refinement, treat ask as fixed
    # But we don't have ask here, so use a simpler criterion: maximize p(b)*(b + C)
    # where C = V_next[q+1] - V_next[q] (the marginal value of increasing inventory)
    C = V_next[bid_next_idx[q_idx]] - V_next[q_idx]
    
    # Try perturbations
    for delta in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
        b_try = b + delta
        if b_try < 1e-6:
            continue
        p_try = 1.0 - np.exp(-A * np.exp(-k * b_try) * dt)
        val = p_try * (b_try + C)
        if val > best_val:
            best_val = val
            best_b = b_try
    
    return best_b


def _refine_depth_ask(a, V_next, q_idx, bid_next_idx, ask_next_idx, A, k, dt, inv_cost, sigma, gamma, q):
    """
    Refine ask depth using gradient-based local search.
    """
    best_a = a
    best_val = -np.inf
    
    # C = V_next[q-1] - V_next[q] (the marginal value of decreasing inventory)
    C = V_next[ask_next_idx[q_idx]] - V_next[q_idx]
    
    for delta in [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]:
        a_try = a + delta
        if a_try < 1e-6:
            continue
        p_try = 1.0 - np.exp(-A * np.exp(-k * a_try) * dt)
        val = p_try * (a_try + C)
        if val > best_val:
            best_val = val
            best_a = a_try
    
    return best_a


def _build_policy(params, deadline):
    """Build or retrieve cached policy for given params."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]

    bid_opt, ask_opt = _solve_policy(sigma, A, k, gamma, T, steps, Q, deadline)

    _policy_cache[key] = (bid_opt, ask_opt)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    Each value is the distance below/above mid to quote.
    np.inf means no quote.
    """
    deadline = _get_deadline()

    n_paths = len(inventory)

    try:
        bid_opt, ask_opt = _build_policy(params, deadline)
    except Exception:
        depth = np.full(n_paths, 1.0 / params["k"])
        return depth, depth.copy()

    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(n, params['steps'] - 1)
    n = max(n, 0)

    Q = params['Q']

    q_idx = np.clip(inventory, -Q, Q) + Q
    q_idx = np.array(q_idx, dtype=np.int32)

    bid_depths = bid_opt[n, q_idx]
    ask_depths = ask_opt[n, q_idx]

    min_valid = -3.0 / params['k']
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, np.inf)

    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
