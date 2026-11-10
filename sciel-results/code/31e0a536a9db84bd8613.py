"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

# Global cache for precomputed optimal policies
_POLICY_CACHE = {}

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
    
    # Helper to find optimal depth for a given marginal value M
    # Maximize f(d) = (1 - exp(-c * exp(-k*d))) * (d + M)
    # We can search over d in a reasonable range.
    # d_min: where fill prob is negligible? Or just 0?
    # d_max: where d+M is large but prob is small.
    # Let's use a grid search or scalar minimization.
    # Since we call this many times, let's precompute a lookup table for M.
    
    # Estimate range of M (V values)
    # Max reward per step is bounded by max d * prob.
    # prob <= 1. d can be large. But optimal d is usually around 1/k.
    # Let's assume V is in [-10000, 10000] to be safe, but we can refine.
    # Actually, we can just solve for each (n, q) on the fly if it's fast enough, 
    # but 1000 * 61 = 61000 optimizations.
    # minimize_scalar is fast, but let's try to be efficient.
    
    # Precompute optimal depth for a range of M values
    # M = V_{n+1}(q')
    # V is likely small. Let's sample M from -100 to 100 initially?
    # Let's do a dynamic range or just solve directly.
    # 61k scalar optimizations might take a few seconds.
    # Let's try to optimize the inner loop.
    
    # We can approximate the optimal d analytically or use a fixed grid of d.
    # Let's use a grid of d values and pick the best one.
    # d range: [0, 10/k] is likely sufficient.
    d_grid = np.linspace(0, 10.0 / k, 1000)
    
    # Precompute probabilities for the grid
    # P(d) = 1 - exp(-c * exp(-k*d))
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-c * exp_term)
    
    # Now for a given M, we want to maximize P_grid * (d_grid + M)
    # This is equivalent to finding the argmax of a vector.
    # We can do this for all M at once if we vectorize, but M changes per (n,q).
    
    # Let's just do the vectorized argmax for each (n, q).
    # It's 61k operations of size 1000. That's 61M operations. NumPy handles this easily.
    
    for n in range(steps - 1, -1, -1):
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
        # Time guard: if we are running out of time, fall back to a heuristic
        # But the first call will take some time. 
        # 1000 steps * 61 states * 1000 grid size = 61M ops. 
        # This should be < 1 second in NumPy.
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
