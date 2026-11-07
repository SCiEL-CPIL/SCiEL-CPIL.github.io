import numpy as np
from functools import lru_cache

# Global cache for optimal policies
_policy_cache = {}

def _get_optimal_policy(params):
    """
    Computes the optimal bid and ask depths for each step and inventory level.
    Returns (bid_depths, ask_depths) arrays of shape (steps, 2*Q+1).
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _policy_cache:
        return _policy_cache[key]
        
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    num_states = 2 * Q + 1
    state_indices = np.arange(-Q, Q + 1)
    
    # Grid for depths
    # Max depth where probability is negligible
    # p = 1 - exp(-A exp(-k d) dt)
    # If k*d is large, exp(-k*d) is small, A*exp(-k*d)*dt is small, p is small.
    # Let's pick a grid up to 10/k or similar.
    max_depth = 10.0 / k if k > 0 else 10.0
    grid_size = 200
    depths_grid = np.linspace(0, max_depth, grid_size)
    
    # Precompute probabilities for all grid points
    # p = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * depths_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    # V_n(q) for n = N down to 0
    # V_N(q) = 0
    V_next = np.zeros(num_states)
    
    # Store optimal actions
    bid_depths = np.zeros((N, num_states))
    ask_depths = np.zeros((N, num_states))
    
    # Iterate backwards
    for n in range(N - 1, -1, -1):
        # V_n(q) = -gamma/2 * sigma^2 * q^2 * dt + max(b, a) [ ... ]
        # The penalty term is constant for a given q, so it doesn't affect the argmax
        # But we need to add it to V_n for the next iteration's expectation
        
        V_curr = np.zeros(num_states)
        
        # Vectorize over states? 
        # The term inside max depends on V_next(q-1), V_next(q), V_next(q+1).
        # We can loop over states (61 is small) and vectorize over grid (200x200 = 40000)
        
        for i, q in enumerate(state_indices):
            # Get neighbor values
            # q is integer
            q_minus = q - 1
            q_plus = q + 1
            
            # Handle boundaries
            # If q == Q, no bid fill. Effectively p_b = 0.
            # If q == -Q, no ask fill. Effectively p_a = 0.
            
            v_minus = V_next[q_minus + Q] if q_minus >= -Q else 0.0 # Should not happen if we handle p=0
            v_plus = V_next[q_plus + Q] if q_plus <= Q else 0.0
            v_curr = V_next[i]
            
            # Create 2D grids for b and a
            # b_grid shape (grid_size, 1), a_grid shape (1, grid_size)
            b_grid = depths_grid[:, np.newaxis]
            a_grid = depths_grid[np.newaxis, :]
            
            p_b_grid = p_grid[:, np.newaxis] # shape (grid_size, 1)
            p_a_grid = p_grid[np.newaxis, :] # shape (1, grid_size)
            
            # If at boundary, force prob to 0
            if q == Q:
                p_b_grid = np.zeros_like(p_b_grid)
            if q == -Q:
                p_a_grid = np.zeros_like(p_a_grid)
            
            # Calculate Expected Next Value
            # E[V] = p_b(1-p_a)v_plus + p_a(1-p_b)v_minus + (1 - p_b - p_a + 2 p_b p_a) v_curr
            # Note: (1-p_b)(1-p_a) + p_b p_a = 1 - p_b - p_a + 2 p_b p_a
            # Let's use the simpler form: v_curr + p_b(1-p_a)(v_plus - v_curr) + p_a(1-p_b)(v_minus - v_curr)
            
            delta_plus = v_plus - v_curr
            delta_minus = v_minus - v_curr
            
            # Vectorized calculation
            # p_b(1-p_a)
            term_b = p_b_grid * (1.0 - p_a_grid)
            term_a = p_a_grid * (1.0 - p_b_grid)
            
            E_V = v_curr + term_b * delta_plus + term_a * delta_minus
            
            # Immediate Reward: b * p_b + a * p_a
            reward = b_grid * p_b_grid + a_grid * p_a_grid
            
            # Total Value for this action
            total_value = reward + E_V
            
            # Find max
            max_val, max_idx = np.unravel_index(np.argmax(total_value), total_value.shape)
            
            opt_b = depths_grid[max_idx[0]]
            opt_a = depths_grid[max_idx[1]]
            
            bid_depths[n, i] = opt_b
            ask_depths[n, i] = opt_a
            
            # Update V_curr
            penalty = 0.5 * gamma * sigma**2 * q**2 * dt
            V_curr[i] = max_val - penalty
        
        V_next = V_curr
        
    _policy_cache[key] = (bid_depths, ask_depths)
    return bid_depths, ask_depths

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths for the given inventory and parameters.
    """
    bid_depths, ask_depths = _get_optimal_policy(params)
    
    # inventory is an array of ints in [-Q, Q]
    # Map to index: q + Q
    Q = params['Q']
    indices = inventory + Q
    
    # Get step index
    # t = n * dt, so n = t / dt
    # But it's safer to use the step count if available or derive from t
    # params has 'steps' and 'T'. dt = T/steps.
    # t is passed as n * dt.
    # We can just use the index of the call? No, we don't have a counter.
    # We can compute n = int(round(t / (params['T'] / params['steps'])))
    
    dt = params['T'] / params['steps']
    n = int(np.round(t / dt))
    
    # Clamp n to valid range
    n = np.clip(n, 0, params['steps'] - 1)
    
    b = bid_depths[n, indices]
    a = ask_depths[n, indices]
    
    return b, a

