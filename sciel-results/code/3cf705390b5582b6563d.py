"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 310.0  # Safety margin before 330s hard kill
_CACHE = {}


def _simulate_objective(coeffs, sigma, A, k, gamma, T, steps, Q, n_paths=2000, seed=42):
    """
    Simulate the objective for a given parametric policy on n_paths random paths.
    coeffs: (c0, c1, c2, c3, c4, c5)
      bid_depth = max(0, c0 + c1*q + c2*q^2 + c3*tau)
      ask_depth = max(0, c4 + c5*q + c2*q^2 + c3*tau)  # share c2, c3 for symmetry
    Actually, let's use separate but structured:
      bid_depth = max(0, a0 + a1*q + a2*tau)
      ask_depth = max(0, b0 + b1*q + a2*tau)
    To keep params small: 5 params (a0, a1, a2, b0, b1)
    But bid should increase with q (more long -> quote bid tighter to offload), ask decrease with q.
    So a1 > 0, b1 < 0.
    """
    dt = T / steps
    rng = np.random.default_rng(seed)
    
    # Initial mid
    mids = np.full(n_paths, 100.0)
    inventories = np.zeros(n_paths, dtype=np.int64)
    
    total_obj = 0.0
    
    # Precompute tau for each step: tau = (steps - 1 - n) * dt, i.e., time remaining
    # At step n, time to maturity is (steps - 1 - n) * dt
    taus = np.array([(steps - 1 - n) * dt for n in range(steps)])
    
    for n in range(steps):
        # Check deadline
        if time.time() > _DEADLINE - 5.0:
            return -np.inf  # Signal timeout
        
        tau = taus[n]
        q = inventories.astype(np.float64)
        
        # Compute depths from params
        # bid: a0 + a1*q + a2*tau
        bid_d = coeffs[0] + coeffs[1] * q + coeffs[2] * tau
        # ask: b0 + b1*q + a2*tau
        ask_d = coeffs[3] + coeffs[4] * q + coeffs[2] * tau
        
        # Clip to non-negative
        bid_d = np.maximum(bid_d, 0.0)
        ask_d = np.maximum(ask_d, 0.0)
        
        # Boundary: no bid fill at q=Q, no ask fill at q=-Q
        # Set depth to inf (no quote) at boundaries
        at_q_max = (inventories == Q)
        at_q_min = (inventories == -Q)
        
        # Probabilities
        # P = 1 - exp(-A * exp(-k * depth) * dt)
        # For depth=inf, P=0. We handle via mask.
        
        bid_prob = np.zeros(n_paths)
        ask_prob = np.zeros(n_paths)
        
        valid_bid = ~at_q_max
        valid_ask = ~at_q_min
        
        if np.any(valid_bid):
            bd = bid_d[valid_bid]
            bid_prob[valid_bid] = 1.0 - np.exp(-A * np.exp(-k * bd) * dt)
        
        if np.any(valid_ask):
            ad = ask_d[valid_ask]
            ask_prob[valid_ask] = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
        
        # Inventory penalty: (gamma/2) * sigma^2 * q^2 * dt
        penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
        total_obj -= np.sum(penalty)
        
        # Random fills
        r_bid = rng.random(n_paths)
        r_ask = rng.random(n_paths)
        
        bid_fill = (r_bid < bid_prob) & valid_bid
        ask_fill = (r_ask < ask_prob) & valid_ask
        
        # Earned depths
        total_obj += np.sum(bid_d[bid_fill])
        total_obj += np.sum(ask_d[ask_fill])
        
        # Update inventory
        inventories = inventories + bid_fill.astype(np.int64) - ask_fill.astype(np.int64)
        
        # Mid moves (doesn't affect objective directly in this model, but included for completeness)
        # mids += sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
    
    return total_obj / n_paths


def _optimize_params(sigma, A, k, gamma, T, steps, Q):
    """
    Optimize 5 parameters: (a0, a1, a2, b0, b1)
    bid_depth = max(0, a0 + a1*q + a2*tau)
    ask_depth = max(0, b0 + b1*q + a2*tau)
    """
    # Initial guess
    # Typical depth scale is ~1/k
    d0 = 1.0 / k
    x0 = np.array([d0, 0.05, 0.0, d0, -0.05])
    
    # Grid search over a coarse grid, then refine
    # a0, b0 in [0.1/k, 5/k]
    # a1, b1 in [-0.5, 0.5] (per unit inventory)
    # a2 in [-0.5, 0.5] (per unit time)
    
    best_obj = -np.inf
    best_x = x0.copy()
    
    # Coarse grid
    a0_grid = np.linspace(0.2/k, 4.0/k, 8)
    a1_grid = np.linspace(0.0, 0.3, 5)
    a2_grid = np.linspace(-0.2, 0.2, 5)
    b1_grid = np.linspace(-0.3, 0.0, 5)
    
    # To save time, use fewer paths for grid search
    n_paths_coarse = 500
    
    for a0 in a0_grid:
        if time.time() > _DEADLINE - 30.0:
            break
        for a1 in a1_grid:
            if time.time() > _DEADLINE - 30.0:
                break
            for a2 in a2_grid:
                if time.time() > _DEADLINE - 30.0:
                    break
                for b1 in b1_grid:
                    if time.time() > _DEADLINE - 30.0:
                        break
                    # b0 = a0 by symmetry (same base depth)
                    coeffs = np.array([a0, a1, a2, a0, b1])
                    obj = _simulate_objective(coeffs, sigma, A, k, gamma, T, steps, Q, n_paths=n_paths_coarse, seed=42)
                    if obj > best_obj:
                        best_obj = obj
                        best_x = coeffs.copy()
    
    # Local refinement with more paths
    if time.time() < _DEADLINE - 10.0:
        n_paths_fine = 2000
        # Simple coordinate ascent
        for _ in range(3):
            if time.time() > _DEADLINE - 10.0:
                break
            improved = False
            for i in range(5):
                for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
                    x_try = best_x.copy()
                    x_try[i] += delta * (1.0/k if i in [0, 3] else 1.0)
                    obj = _simulate_objective(x_try, sigma, A, k, gamma, T, steps, Q, n_paths=n_paths_fine, seed=42)
                    if obj > best_obj:
                        best_obj = obj
                        best_x = x_try
                        improved = True
                        break
            if not improved:
                break
    
    return best_x


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve for optimal parametric policy coefficients.
    Returns coefficients array.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    coeffs = _optimize_params(sigma, A, k, gamma, T, steps, Q)
    _CACHE[key] = coeffs
    return coeffs


def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,).
    Uses parametric policy.
    """
    if time.time() > _DEADLINE:
        # Fallback to seed
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    coeffs = _solve_regime(
        params["sigma"], params["A"], params["k"], 
        params["gamma"], params["T"], params["steps"], params["Q"]
    )
    
    # t is n * dt, so step index n = int(round(t / dt))
    dt = params["T"] / params["steps"]
    n = int(round(t / dt))
    n = max(0, min(n, params["steps"] - 1))
    
    # Time to maturity
    tau = (params["steps"] - 1 - n) * dt
    
    q = inventory.astype(np.float64)
    
    # Compute depths
    bid_d = coeffs[0] + coeffs[1] * q + coeffs[2] * tau
    ask_d = coeffs[3] + coeffs[4] * q + coeffs[2] * tau
    
    # Clip to non-negative
    bid_d = np.maximum(bid_d, 0.0)
    ask_d = np.maximum(ask_d, 0.0)
    
    # Boundary: no quote at boundaries
    Q = params["Q"]
    bid_d[inventory == Q] = np.inf
    ask_d[inventory == -Q] = np.inf
    
    # Ensure no NaN
    bid_d = np.nan_to_num(bid_d, nan=0.0, posinf=np.inf, neginf=0.0)
    ask_d = np.nan_to_num(ask_d, nan=0.0, posinf=np.inf, neginf=0.0)
    
    return bid_d, ask_d
# EVOLVE-BLOCK-END
