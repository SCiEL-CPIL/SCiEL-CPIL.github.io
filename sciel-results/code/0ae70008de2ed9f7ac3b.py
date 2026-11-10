# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for precomputed optimal policies
_POLICY_CACHE = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0  # Safety margin before 330s kill

def _refine_depth(c, k, M, d_lo, d_hi, max_iter=50):
    """
    Refine the optimal depth d in [d_lo, d_hi] using Golden Section Search.
    Objective: maximize f(d) = (1 - exp(-c * exp(-k*d))) * (d + M)
    """
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    
    def f(d):
        p = 1.0 - np.exp(-c * np.exp(-k * d))
        return p * (d + M)

    a = d_lo
    b = d_hi
    c_val = b - gr * (b - a)
    d_val = a + gr * (b - a)
    
    fc = f(c_val)
    fd = f(d_val)
    
    for _ in range(max_iter):
        if fc < fd:
            a = c_val
            c_val = d_val
            d_val = a + gr * (b - a)
            fc = fd
            fd = f(d_val)
        else:
            b = d_val
            d_val = c_val
            c_val = b - gr * (b - a)
            fd = fc
            fc = f(c_val)
            
        if abs(b - a) < 1e-9:
            break
            
    return (a + b) / 2.0

def _solve_mdp(params):
    """
    Solves the finite-horizon MDP for the given parameters.
    Returns arrays D_b and D_a of shape (steps, 2*Q + 1) containing optimal depths.
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
    
    # Coarse grid to find initial bracket for Golden Section Search
    # The optimal depth is usually small. 10/k is a safe upper bound.
    # Use a slightly finer grid for better bracketing
    d_grid = np.linspace(0, 10.0 / k, 200)
    
    # Precompute probabilities for the coarse grid
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-c * exp_term)
    
    for n in range(steps - 1, -1, -1):
        # Check time
        if time.time() - _START_TIME > TIME_LIMIT - 10:
             # Fallback: return current best or simple heuristic
             for n2 in range(n, -1, -1):
                 for q2 in range(-Q, Q + 1):
                     q_idx2 = q2 + Q
                     D_b[n2, q_idx2] = 0.0
                     D_a[n2, q_idx2] = 0.0
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
                
                # Determine bracket from coarse grid
                if idx_b == 0:
                    d_lo = 0.0
                    d_hi = d_grid[1]
                elif idx_b == len(d_grid) - 1:
                    d_lo = d_grid[-2]
                    d_hi = d_grid[-1]
                else:
                    d_lo = d_grid[idx_b - 1]
                    d_hi = d_grid[idx_b + 1]
                    
                d_b_opt = _refine_depth(c, k, M_b, d_lo, d_hi)
                # Calculate value at optimal depth
                p_opt = 1.0 - np.exp(-c * np.exp(-k * d_b_opt))
                val_b = p_opt * (d_b_opt + M_b)
            else:
                d_b_opt = np.inf
                val_b = 0.0
                
            # Ask side
            if q > -Q:
                M_a = V[n + 1, q - 1 + Q]  # V(n+1, q-1)
                # Maximize P_grid * (d_grid + M_a)
                obj_a = P_grid * (d_grid + M_a)
                idx_a = np.argmax(obj_a)
                
                # Determine bracket from coarse grid
                if idx_a == 0:
                    d_lo = 0.0
                    d_hi = d_grid[1]
                elif idx_a == len(d_grid) - 1:
                    d_lo = d_grid[-2]
                    d_hi = d_grid[-1]
                else:
                    d_lo = d_grid[idx_a - 1]
                    d_hi = d_grid[idx_a + 1]
                    
                d_a_opt = _refine_depth(c, k, M_a, d_lo, d_hi)
                # Calculate value at optimal depth
                p_opt = 1.0 - np.exp(-c * np.exp(-k * d_a_opt))
                val_a = p_opt * (d_a_opt + M_a)
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
    
    # Ensure depths are positive (grid starts at 0, so this is safe, but check for inf/NaN)
    # np.inf is allowed. NaN is not.
    # The grid search returns finite values.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
