import numpy as np
from functools import lru_cache
import time

# Precompute DP tables for each regime
_dp_cache = {}

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """Solve the MDP via backward induction. Returns V table and optimal actions."""
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _dp_cache:
        return _dp_cache[cache_key]
    
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # Depth grid: we need to choose bid_depth and ask_depth
    # Valid depths must be > -3/k. Practically, use positive depths.
    # Let's use a grid from a small value to a large value
    # For k=1.5, -3/k = -2. For k=0.75, -3/k = -4.
    # We'll use depths in [0.01, 10] with some resolution
    
    # Number of depth levels - need to balance accuracy vs speed
    # 1000 steps * 61 inv states * depth_grid^2 operations
    # If depth_grid = 50, that's 1000 * 61 * 2500 = 152.5M operations - too slow
    # If depth_grid = 20, that's 1000 * 61 * 400 = 24.4M - might be ok
    # Let's try to be smarter
    
    # Actually, we can vectorize over depth choices
    # For each (n, q), we need to maximize over (bd, ad)
    
    # Let's use a smaller grid first
    depth_grid_size = 30
    depth_min = 0.01
    depth_max = 5.0 / k  # reasonable upper bound
    depths = np.linspace(depth_min, depth_max, depth_grid_size)
    
    # V[n, q_idx] = optimal value from step n with inventory q
    # q_idx = q + Q, so q = q_idx - Q
    V = np.zeros((n_steps + 1, n_inv))
    
    # Action tables
    opt_bid = np.zeros((n_steps, n_inv))
    opt_ask = np.zeros((n_steps, n_inv))
    
    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Inventory cost for this step
            inv_cost = -(gamma / 2.0) * sigma**2 * q**2 * dt
            
            best_val = -np.inf
            best_bd = depths[0]
            best_ad = depths[0]
            
            # If at boundary, one side must be inf (no quote)
            if q == Q:
                # No bid fill possible, so bid_depth = inf
                # Only ask side matters
                for ad in depths:
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
                    # No bid fill
                    # Outcomes: ask fill (q->q-1) or no fill (q stays)
                    val = inv_cost + p_ask * ad + p_ask * V[n+1, q_idx-1] + (1-p_ask) * V[n+1, q_idx]
                    if val > best_val:
                        best_val = val
                        best_bd = np.inf
                        best_ad = ad
            elif q == -Q:
                # No ask fill possible, so ask_depth = inf
                for bd in depths:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * bd) * dt)
                    # No ask fill
                    # Outcomes: bid fill (q->q+1) or no fill (q stays)
                    val = inv_cost + p_bid * bd + p_bid * V[n+1, q_idx+1] + (1-p_bid) * V[n+1, q_idx]
                    if val > best_val:
                        best_val = val
                        best_bd = bd
                        best_ad = np.inf
            else:
                # Both sides active
                # Vectorize over depth pairs
                p_bid_arr = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)  # shape (D,)
                p_ask_arr = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)  # shape (D,)
                
                # For each (bd_idx, ad_idx), compute value
                # This is a 2D grid of size D x D
                # V_next_q = V[n+1, q_idx]
                # V_next_q_plus = V[n+1, q_idx+1]  # bid fill: q->q+1
                # V_next_q_minus = V[n+1, q_idx-1]  # ask fill: q->q-1
                
                V_q = V[n+1, q_idx]
                V_q_plus = V[n+1, q_idx+1]
                V_q_minus = V[n+1, q_idx-1]
                
                # Create 2D arrays
                p_bid_2d = p_bid_arr[:, np.newaxis]  # (D, 1)
                p_ask_2d = p_ask_arr[np.newaxis, :]  # (1, D)
                
                bd_2d = depths[:, np.newaxis]  # (D, 1)
                ad_2d = depths[np.newaxis, :]  # (1, D)
                
                # Fill probabilities
                p_both = p_bid_2d * p_ask_2d
                p_bid_only = p_bid_2d * (1 - p_ask_2d)
                p_ask_only = (1 - p_bid_2d) * p_ask_2d
                p_neither = (1 - p_bid_2d) * (1 - p_ask_2d)
                
                # Earnings
                earnings = p_bid_2d * bd_2d + p_ask_2d * ad_2d
                
                # Future value
                future = p_both * V_q + p_bid_only * V_q_plus + p_ask_only * V_q_minus + p_neither * V_q
                
                total = inv_cost + earnings + future
                
                # Find max
                max_idx = np.unravel_index(np.argmax(total), total.shape)
                best_val = total[max_idx]
                best_bd = depths[max_idx[0]]
                best_ad = depths[max_idx[1]]
            
            V[n, q_idx] = best_val
            opt_bid[n, q_idx] = best_bd
            opt_ask[n, q_idx] = best_ad
    
    result = (V, opt_bid, opt_ask)
    _dp_cache[cache_key] = result
    return result


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    
    # Determine current step
    dt = T / steps
    step = int(round(t / dt))
    if step >= steps:
        step = steps - 1
    
    # Get or compute DP solution
    V, opt_bid, opt_ask = _solve_dp(sigma, A, k, gamma, T, steps, Q)
    
    # Map inventory to index
    inv_idx = inventory + Q
    
    bid_depths = opt_bid[step, inv_idx].copy()
    ask_depths = opt_ask[step, inv_idx].copy()
    
    # Ensure valid depths: must be > -3/k
    min_depth = -3.0 / k + 1e-9
    bid_depths[bid_depths <= min_depth] = np.inf
    ask_depths[ask_depths <= min_depth] = np.inf
    
    # Ensure shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n_paths)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n_paths)
    
    return bid_depths, ask_depths
