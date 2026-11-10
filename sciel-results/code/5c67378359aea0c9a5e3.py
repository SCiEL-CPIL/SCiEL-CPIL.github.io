"""Inventory-aware market making with dynamic programming."""
import numpy as np
import time

# Cache for pre-computed optimal indices per regime
_cache = {}

def _precompute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Pre-compute optimal bid and ask depth indices for each (step, inventory) pair.
    Uses vectorized DP with a grid of depths.
    Index 0 corresponds to "no quote" (depth inf, prob 0).
    Indices 1..M correspond to actual quotes.
    """
    dt = T / steps
    n_states = 2 * Q + 1  # inventory from -Q to Q
    
    # Grid of candidate depths
    # Index 0 is "no quote"
    # Indices 1..M are actual quotes
    max_depth = 5.0 / k
    n_grid = 50
    # Actual depths
    D_actual = np.linspace(0.0, max_depth, n_grid)
    
    # Pre-compute probabilities for the grid
    P_actual = 1.0 - np.exp(-A * np.exp(-k * D_actual) * dt)
    
    # Construct full grid including "no quote"
    D_grid = np.concatenate(([0.0], D_actual)) # D_grid[0] is dummy, D_grid[1:] are actual
    P_grid = np.concatenate(([0.0], P_actual))
    M = len(D_grid)
    
    # V[n, q] for n in [0, steps], q in [-Q, Q]
    # V[steps, q] = 0
    V = np.zeros(n_states)
    
    # Store optimal indices
    bid_opt_idx = np.zeros((steps, n_states), dtype=np.int32)
    ask_opt_idx = np.zeros((steps, n_states), dtype=np.int32)
    
    # Pre-compute cost coefficients
    cost_coeff = (gamma / 2.0) * sigma**2 * dt
    
    # Work backwards
    for n in range(steps - 1, -1, -1):
        V_next = V
        q_idx = np.arange(n_states)
        
        V_curr = V_next[q_idx]
        
        idx_bid = np.clip(q_idx + 1, 0, n_states - 1)
        V_next_bid = V_next[idx_bid]
        
        idx_ask = np.clip(q_idx - 1, 0, n_states - 1)
        V_next_ask = V_next[idx_ask]
        
        # Broadcasting
        D_i = D_grid[:, np.newaxis, np.newaxis]
        D_j = D_grid[np.newaxis, :, np.newaxis]
        
        P_i = P_grid[:, np.newaxis, np.newaxis]
        P_j = P_grid[np.newaxis, :, np.newaxis]
        
        V_curr_b = V_curr[np.newaxis, np.newaxis, :]
        V_bid_b = V_next_bid[np.newaxis, np.newaxis, :]
        V_ask_b = V_next_ask[np.newaxis, np.newaxis, :]
        
        term1 = (1.0 - P_i) * (1.0 - P_j) * V_curr_b
        term2 = P_i * (1.0 - P_j) * (D_i + V_bid_b)
        term3 = (1.0 - P_i) * P_j * (D_j + V_ask_b)
        term4 = P_i * P_j * (D_i + D_j + V_curr_b)
        
        Val = term1 + term2 + term3 + term4
        
        # Apply boundary constraints
        if n_states > 0:
            mask_q_max = np.zeros(n_states, dtype=bool)
            mask_q_max[n_states - 1] = True
            mask_i_gt0 = np.ones(M, dtype=bool)
            mask_i_gt0[0] = False
            Val[mask_q_max, mask_i_gt0, :] = -np.inf
            
            mask_q_min = np.zeros(n_states, dtype=bool)
            mask_q_min[0] = True
            mask_j_gt0 = np.ones(M, dtype=bool)
            mask_j_gt0[0] = False
            Val[mask_q_min, :, mask_j_gt0] = -np.inf
        
        Val_flat = Val.reshape(n_states, -1)
        best_flat_idx = np.argmax(Val_flat, axis=1)
        
        best_i = best_flat_idx // M
        best_j = best_flat_idx % M
        
        bid_opt_idx[n, :] = best_i
        ask_opt_idx[n, :] = best_j
        
        best_val = Val[np.arange(n_states), best_i, best_j]
        q_vals = q_idx - Q
        costs = cost_coeff * q_vals**2
        V = best_val - costs
        
    # Store grid in cache to map indices to depths later
    return bid_opt_idx, ask_opt_idx, D_grid


def quote(t, inventory, mid, params):
    """
    Inventory-aware market making policy using pre-computed optimal depths.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    if not hasattr(quote, '_start_time'):
        quote._start_time = time.time()
    elapsed = time.time() - quote._start_time
    if elapsed > 280:
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _cache:
        bid_opt_idx, ask_opt_idx, D_grid = _precompute_policy(sigma, A, k, gamma, T, steps, Q)
        _cache[cache_key] = (bid_opt_idx, ask_opt_idx, D_grid)
    else:
        bid_opt_idx, ask_opt_idx, D_grid = _cache[cache_key]
    
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    inv = np.asarray(inventory, dtype=np.int32)
    inv_idx = inv + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_indices = bid_opt_idx[n, inv_idx]
    ask_indices = ask_opt_idx[n, inv_idx]
    
    # Map indices to depths
    # Index 0 is "no quote" -> np.inf
    # Index > 0 is D_grid[index]
    
    bid_depths = np.where(bid_indices == 0, np.inf, D_grid[bid_indices])
    ask_depths = np.where(ask_indices == 0, np.inf, D_grid[ask_indices])
    
    return bid_depths, ask_depths
