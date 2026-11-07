"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 325.0  # 5s safety margin before 330s kill
_cache = {}

def _get_dp_tables(params):
    """Compute optimal bid/ask depth tables via dynamic programming with vectorized grid search."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Value table: V[n, q] for n=0..steps, q=-Q..Q
    # V[steps][q] = 0 (no terminal charge)
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal depth tables
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute risk penalty
    q_indices = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma**2 * q_indices**2 * dt
    
    # Precompute depth grid for vectorized search
    d_max = 15.0 / k  # reasonable upper bound
    n_grid = 500
    depth_grid = np.linspace(0.0, d_max, n_grid)
    
    # Precompute fill probabilities for the grid
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Precompute depth_grid as column vector for broadcasting
    depth_col = depth_grid[np.newaxis, :]  # shape (1, n_grid)
    p_row = p_grid[np.newaxis, :]  # shape (1, n_grid)
    
    # Backward induction - vectorized over inventory levels
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > TIME_LIMIT - 30:
            # Not enough time; fill remaining with current best estimates
            break
        
        V_next = V[n + 1]  # shape (2Q+1,)
        
        # Bid side: for each q in [-Q, Q-1], delta_b = V[n+1, q+1] - V[n+1, q]
        # For q=Q, no bid
        # delta_b[i] = V_next[i+1] - V_next[i] for i=0..2Q-1, and -inf for i=2Q
        delta_b = np.empty(2 * Q + 1)
        delta_b[:2 * Q] = V_next[1:] - V_next[:2 * Q]
        delta_b[2 * Q] = 0.0  # won't be used
        
        # Ask side: for each q in [-Q+1, Q], delta_a = V[n+1, q-1] - V[n+1, q]
        # For q=-Q, no ask
        delta_a = np.empty(2 * Q + 1)
        delta_a[1:] = V_next[:-1] - V_next[1:]
        delta_a[0] = 0.0  # won't be used
        
        # Compute rewards for all inventory levels simultaneously
        # rewards_b[i, j] = p_grid[j] * (depth_grid[j] + delta_b[i])
        # But we need to handle the boundary: for q=Q, bid is not allowed
        # For q=-Q, ask is not allowed
        
        # Bid rewards: shape (2Q+1, n_grid)
        rewards_b = p_row * (depth_col + delta_b[:, np.newaxis])
        # Set invalid entries (q=Q) to -inf so they don't get selected
        rewards_b[2 * Q, :] = -np.inf
        
        # Ask rewards: shape (2Q+1, n_grid)
        rewards_a = p_row * (depth_col + delta_a[:, np.newaxis])
        # Set invalid entries (q=-Q) to -inf
        rewards_a[0, :] = -np.inf
        
        # Find optimal depth for each inventory level
        idx_b = np.argmax(rewards_b, axis=1)
        idx_a = np.argmax(rewards_a, axis=1)
        
        bid_depths = depth_grid[idx_b]
        ask_depths = depth_grid[idx_a]
        bid_rewards = rewards_b[np.arange(2 * Q + 1), idx_b]
        ask_rewards = rewards_a[np.arange(2 * Q + 1), idx_a]
        
        # Handle boundaries explicitly
        bid_depths[2 * Q] = np.inf
        ask_depths[0] = np.inf
        bid_rewards[2 * Q] = 0.0
        ask_rewards[0] = 0.0
        
        # Local refinement: for each inventory level, do a finer search around the best grid point
        # Only refine if we have time
        elapsed = time.time() - _START_TIME
        if elapsed < TIME_LIMIT - 40:
            n_refine = 50
            for i in range(2 * Q + 1):
                # Bid refinement
                if i < 2 * Q:
                    ib = idx_b[i]
                    if 0 < ib < n_grid - 1:
                        d_lo = depth_grid[ib - 1]
                        d_hi = depth_grid[ib + 1]
                        d_fine = np.linspace(d_lo, d_hi, n_refine)
                        p_fine = 1.0 - np.exp(-A * np.exp(-k * d_fine) * dt)
                        r_fine = p_fine * (d_fine + delta_b[i])
                        j_max = np.argmax(r_fine)
                        if r_fine[j_max] > bid_rewards[i]:
                            bid_depths[i] = d_fine[j_max]
                            bid_rewards[i] = r_fine[j_max]
                
                # Ask refinement
                if i > 0:
                    ia = idx_a[i]
                    if 0 < ia < n_grid - 1:
                        d_lo = depth_grid[ia - 1]
                        d_hi = depth_grid[ia + 1]
                        d_fine = np.linspace(d_lo, d_hi, n_refine)
                        p_fine = 1.0 - np.exp(-A * np.exp(-k * d_fine) * dt)
                        r_fine = p_fine * (d_fine + delta_a[i])
                        j_max = np.argmax(r_fine)
                        if r_fine[j_max] > ask_rewards[i]:
                            ask_depths[i] = d_fine[j_max]
                            ask_rewards[i] = r_fine[j_max]
        
        # Update value function
        V[n] = V_next - risk + bid_rewards + ask_rewards
        opt_bid[n] = bid_depths
        opt_ask[n] = ask_depths
    
    # If we broke out early, fill remaining steps with the last computed step
    for n in range(steps - 1, -1, -1):
        if np.all(np.isinf(opt_bid[n])):
            if n + 1 < steps:
                opt_bid[n] = opt_bid[n + 1]
                opt_ask[n] = opt_ask[n + 1]
            else:
                # Fallback: use 1/k
                base = 1.0 / k
                opt_bid[n] = np.where(q_indices < Q, base, np.inf)
                opt_ask[n] = np.where(q_indices > -Q, base, np.inf)
    
    _cache[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Quote optimal depths based on dynamic programming solution."""
    # Deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - 5:
        k = params['k']
        base = 1.0 / k
        N = len(inventory)
        bid_depths = np.full(N, base)
        ask_depths = np.full(N, base)
        # Handle boundaries
        Q = params['Q']
        bid_depths[inventory == Q] = np.inf
        ask_depths[inventory == -Q] = np.inf
        return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
    
    opt_bid, opt_ask = _get_dp_tables(params)
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    Q = params['Q']
    
    # Current step index
    n = int(t / dt)
    n = min(n, steps - 1)  # clamp to valid range
    
    # Map inventory to indices
    inv_idx = inventory + Q
    
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Safety: ensure valid values
    min_depth = -3.0 / params['k'] + 0.01
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_depth), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_depth), ask_depths, np.inf)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)

# EVOLVE-BLOCK-END
