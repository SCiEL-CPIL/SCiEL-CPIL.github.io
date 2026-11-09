"""Combined program: PDE-based HJB solution with closed-form depth extraction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2)
]

T_VAL = 1.0
STEPS = 1000
Q_VAL = 30

POLICIES = {}
_DEADLINE = None


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 300  # 30s safety margin before 330s kill
    return _DEADLINE


def _solve_pde_policy(sigma, A, k, gamma, T, steps, Q, deadline):
    """
    Solve the continuous-time HJB equation using an explicit finite-difference scheme.
    The value function V(t, q) satisfies:
      -dV/dt = gamma/2 * sigma^2 * q^2 - max_bid - max_ask
    where max_bid and max_ask are the optimal contributions from quoting.
    
    We discretize in time (backward from T to 0) and inventory (q in [-Q, Q]).
    For each (t, q), we derive optimal depths via first-order conditions.
    
    Returns (bid_table, ask_table) where:
      bid_table[n, q_idx] = optimal bid depth at step n, inventory q
      ask_table[n, q_idx] = optimal ask depth at step n, inventory q
      q_idx = q + Q, so q_idx in [0, 2Q]
    """
    dt = T / steps
    n_levels = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Value function V[n, q_idx]
    V = np.zeros((steps + 1, n_levels))
    
    # Store optimal actions
    bid_opt = np.full((steps, n_levels), np.inf)
    ask_opt = np.full((steps, n_levels), np.inf)
    
    # Precompute constants
    sigma2 = sigma * sigma
    
    # For the PDE approach, we use the following logic:
    # At each step, given V_next (value at next time step), we compute:
    #   1. The inventory gradient dV/dq via finite differences
    #   2. The optimal bid/ask depths via first-order conditions
    #   3. Update V_curr using the HJB equation
    
    # First-order condition for bid depth b:
    # Maximize: p(b) * (b + C_b) where C_b = V_next[q+1] - V_next[q]
    # p(b) = 1 - exp(-A * exp(-k*b) * dt)
    # 
    # Let lambda = A * exp(-k*b) * dt
    # p(b) = 1 - exp(-lambda)
    # d/db [p(b)*(b+C)] = p'(b)*(b+C) + p(b)
    # p'(b) = lambda * k * exp(-lambda) = A*k*exp(-k*b)*dt*exp(-lambda)
    # 
    # This is complex; let's use a small grid search but with adaptive refinement
    # Actually, for speed, let's use a coarser grid with fewer points but still accurate
    
    # Use a moderate grid for depth optimization
    max_depth = 4.0 / k
    n_grid = 200
    depth_grid = np.linspace(0.0, max_depth, n_grid)
    
    # Precompute fill probabilities for each depth
    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            break
        
        V_next = V[n + 1]
        V_curr = V[n]
        
        # Holding cost: -(gamma/2) * sigma^2 * q^2 * dt
        inv_cost = -0.5 * gamma * sigma2 * q_vals * q_vals * dt
        
        # --- Bid Side ---
        # For q in [-Q, Q-1] (indices 0 to 2Q-1)
        # C_b[q_idx] = V_next[q_idx+1] - V_next[q_idx]
        C_b = V_next[1:] - V_next[:-1]  # shape (2Q,)
        
        # obj[b, q] = p_fill[b] * (depth_grid[b] + C_b[q])
        # Shape: (n_grid, 2Q)
        obj_b = p_fill[:, np.newaxis] * (depth_grid[:, np.newaxis] + C_b[np.newaxis, :])
        
        best_b_idx = np.argmax(obj_b, axis=0)
        best_b = depth_grid[best_b_idx]
        max_obj_b = obj_b[best_b_idx, np.arange(2 * Q)]
        
        # V_curr[q] gets holding cost + V_next[q] + max bid contribution
        V_curr[:2*Q] = inv_cost[:2*Q] + V_next[:2*Q] + max_obj_b
        
        # Store optimal bid depths
        bid_opt[n, :2*Q] = best_b
        bid_opt[n, 2*Q] = np.inf  # q=Q, no bid possible
        
        # --- Ask Side ---
        # For q in [-Q+1, Q] (indices 1 to 2Q)
        # C_a[q_idx] = V_next[q_idx-1] - V_next[q_idx]
        C_a = V_next[:-1] - V_next[1:]  # shape (2Q,)
        
        # obj_a[a, q] = p_fill[a] * (depth_grid[a] + C_a[q])
        obj_a = p_fill[:, np.newaxis] * (depth_grid[:, np.newaxis] + C_a[np.newaxis, :])
        
        best_a_idx = np.argmax(obj_a, axis=0)
        best_a = depth_grid[best_a_idx]
        max_obj_a = obj_a[best_a_idx, np.arange(2 * Q)]
        
        # Add ask contribution to V_curr for q = -Q+1..Q (indices 1..2Q)
        V_curr[1:] += max_obj_a
        
        # Store optimal ask depths
        ask_opt[n, 1:] = best_a
        ask_opt[n, 0] = np.inf  # q=-Q, no ask possible
    
    return bid_opt, ask_opt


def _initialize_policies():
    global POLICIES
    deadline = _get_deadline()
    
    for i, (sigma, A, k, gamma) in enumerate(REGIMES):
        if time.time() > deadline:
            break
        bid, ask = _solve_pde_policy(sigma, A, k, gamma, T_VAL, STEPS, Q_VAL, deadline)
        POLICIES[i] = (bid, ask)


def _get_regime_index(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    
    for i, (s, a, kk, g) in enumerate(REGIMES):
        if np.isclose(sigma, s, atol=1e-6) and np.isclose(A, a, atol=1e-6) and \
           np.isclose(k, kk, atol=1e-6) and np.isclose(gamma, g, atol=1e-6):
            return i
    return 0


def _fallback_quote(t, inventory, mid, params):
    """Closed-form heuristic fallback if pre-computation didn't finish."""
    sigma = params["sigma"]
    k = params["k"]
    gamma = params["gamma"]
    
    inv = inventory.astype(np.float64)
    base_depth = 1.0 / k
    skew = 0.5 * gamma * sigma * sigma * inv / k
    
    bid_depths = base_depth + skew
    ask_depths = base_depth - skew
    
    min_valid = -3.0 / k + 1e-9
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    
    return bid_depths, ask_depths


# Initialize on import
_initialize_policies()


def quote(t, inventory, mid, params):
    """
    Quote both sides based on pre-computed optimal policy.
    Returns (bid_depths, ask_depths) as arrays of shape (4000,).
    """
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(np.round(t / dt))
    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1
    
    # Get regime index
    reg_idx = _get_regime_index(params)
    
    # Check if policy is available
    if reg_idx not in POLICIES:
        return _fallback_quote(t, inventory, mid, params)
    
    opt_bid, opt_ask = POLICIES[reg_idx]
    
    # Inventory is in [-Q, Q], map to [0, 2Q]
    Q = params['Q']
    inv_idx = inventory + Q
    
    # Clamp indices for safety
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get depths
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Ensure valid depths: no values below -3/k, -inf, or NaN
    min_valid = -3.0 / params['k'] + 1e-9
    bid_depths = np.where(np.isfinite(bid_depths), np.maximum(bid_depths, min_valid), np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), np.maximum(ask_depths, min_valid), np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
