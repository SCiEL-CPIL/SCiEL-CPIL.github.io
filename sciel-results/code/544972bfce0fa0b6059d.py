# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for precomputed optimal policies
_POLICY_CACHE = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0  # Safety margin before 330s kill

def _solve_mdp(params):
    """
    Solves the finite-horizon MDP for the given parameters.
    Returns arrays D_b and D_a of shape (steps, 2*Q + 1) containing optimal depths.
    Uses a high-resolution grid search for optimal depth.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    c = A * dt  # Constant in the exponent for fill probability
    
    # Value function V[n, q_index]
    # q_index = q + Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depths D_b[n, q_index], D_a[n, q_index]
    D_b = np.full((steps, 2 * Q + 1), np.inf)
    D_a = np.full((steps, 2 * Q + 1), np.inf)
    
    # High-resolution grid for depth search
    # The optimal depth is typically small. 5/k is a safe upper bound.
    # Use 5000 points for high precision.
    d_max = 5.0 / k
    d_grid = np.linspace(0, d_max, 5000)
    
    # Precompute probabilities for the grid
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-c * exp_term)
    
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # If time is running out, use a simple heuristic for remaining steps
            # This is a fallback; normally the loop finishes well within time.
            for q in range(-Q, Q + 1):
                q_idx = q + Q
                cost = 0.5 * gamma * sigma * sigma * q * q * dt
                # Simple heuristic: quote at mid (depth 0) or skip
                # For now, just set to 0 to be safe, though this is suboptimal
                if q < Q:
                    D_b[n, q_idx] = 0.0
                    V[n, q_idx] = -cost
                if q > -Q:
                    D_a[n, q_idx] = 0.0
                    V[n, q_idx] = -cost
            break
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            
            # Cost
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Bid side
            if q < Q:
                M_b = V[n + 1, q + 1 + Q]  # V(n+1, q+1)
                # Maximize P_grid * (d_grid + M_b)
                obj_b = P_grid * (d_grid + M_b)
                idx_b = np.argmax(obj_b)
                d_b_opt = d_grid[idx_b]
                val_b = obj_b[idx_b]
            else:
                d_b_opt = np.inf
                val_b = 0.0
                
            # Ask side
            if q > -Q:
                M_a = V[n + 1, q - 1 + Q]  # V(n+1, q-1)
                # Maximize P_grid * (d_grid + M_a)
                obj_a = P_grid * (d_grid + M_a)
                idx_a = np.argmax(obj_a)
                d_a_opt = d_grid[idx_a]
                val_a = obj_a[idx_a]
            else:
                d_a_opt = np.inf
                val_a = 0.0
                
            D_b[n, q_idx] = d_b_opt
            D_a[n, q_idx] = d_a_opt
            
            V[n, q_idx] = val_b + val_a - cost
            
    return D_b, D_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create a key for the cache
    key = (sigma, A, k, gamma, T, steps, Q)
    
    # Check if we have already solved this MDP
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _solve_mdp(params)
    
    D_b, D_a = _POLICY_CACHE[key]
    
    # Determine current step n
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to valid range [0, steps-1]
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    # Process inventory
    inv = np.asarray(inventory, dtype=np.int64)
    
    # Map inventory to index: q + Q
    # Ensure inventory is within [-Q, Q] for indexing, though problem says it is
    # If out of bounds, we should handle gracefully, but problem guarantees [-Q, Q]
    q_indices = inv + Q
    
    # Retrieve depths
    bid_depths = D_b[n, q_indices].astype(np.float64)
    ask_depths = D_a[n, q_indices].astype(np.float64)
    
    # Apply boundary constraints explicitly to be safe (though D_b/D_a should already be inf)
    # No bid fill at q = Q
    bid_depths[inv >= Q] = np.inf
    # No ask fill at q = -Q
    ask_depths[inv <= -Q] = np.inf
    
    # Ensure depths are non-negative and valid
    # np.inf is allowed. NaN is not.
    # Replace any negative values with 0 (shouldn't happen with grid starting at 0)
    bid_depths = np.where(bid_depths < 0, 0.0, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.0, ask_depths)
    # Replace any NaN with inf (no quote)
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
