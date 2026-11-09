"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for solved regimes
_regime_cache = {}

def _solve_regime(sigma, A, k, gamma, T, steps, Q, deadline):
    """
    Solve the optimal market making policy via backward induction (DP).
    Returns (bid_depths_table, ask_depths_table) where each is array of shape (steps+1, 2*Q+1)
    containing the optimal depth for each (time_step, inventory) pair.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _regime_cache:
        return _regime_cache[key]
    
    dt = T / steps
    n_inventory = 2 * Q + 1
    inv_indices = np.arange(-Q, Q + 1)
    
    # Precompute fill probabilities for a range of depths
    # We'll use a depth grid for optimization
    # Depth range: 0 to some max (e.g., 10/k or so)
    max_depth = 20.0 / k
    n_depths = 200
    depth_grid = np.linspace(0.001, max_depth, n_depths)
    
    # Precompute fill probabilities for each depth
    # p_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # DP tables
    # V[t, q] = optimal expected value from time t with inventory q
    # We store the value and the optimal action (depth index)
    V = np.zeros((steps + 1, n_inventory))
    # For bid: we store the optimal depth index for bid and ask separately
    bid_opt = np.zeros((steps + 1, n_inventory), dtype=np.int32)
    ask_opt = np.zeros((steps + 1, n_inventory), dtype=np.int32)
    
    # Terminal condition: V[steps, q] = 0 (no terminal charge)
    # Backward induction from t = steps-1 down to 0
    
    start_time = time.time()
    
    for t in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() - start_time > deadline:
            break
        
        # Inventory cost for this step: (gamma/2) * sigma^2 * q^2 * dt
        inv_cost = (gamma / 2.0) * sigma**2 * inv_indices**2 * dt
        
        for i, q in enumerate(inv_indices):
            # Value if we don't quote (depth = inf): no fill, just move to next step
            # But we must quote something. Let's consider all depth choices.
            
            # For bid:
            # If q == Q, no bid fill possible
            if q == Q:
                bid_val = V[t+1, i]  # no change in inventory
                bid_opt[t, i] = 0  # doesn't matter
            else:
                # If bid fills: gain depth, inventory increases by 1
                # Expected value = p * (depth + V[t+1, q+1]) + (1-p) * V[t+1, q]
                # = V[t+1, q] + p * (depth + V[t+1, q+1] - V[t+1, q])
                # We want to maximize this over depth
                # Note: depth is the actual depth value, not index
                # Let's compute for all depths in the grid
                p = p_fill
                # Value = V[t+1, i] + p * (depth_grid + V[t+1, i+1] - V[t+1, i])
                val_bid = V[t+1, i] + p * (depth_grid + V[t+1, i+1] - V[t+1, i])
                best_idx = np.argmax(val_bid)
                bid_opt[t, i] = best_idx
                bid_val = val_bid[best_idx]
            
            # For ask:
            # If q == -Q, no ask fill possible
            if q == -Q:
                ask_val = V[t+1, i]
                ask_opt[t, i] = 0
            else:
                # If ask fills: gain depth, inventory decreases by 1
                p = p_fill
                val_ask = V[t+1, i] + p * (depth_grid + V[t+1, i-1] - V[t+1, i])
                best_idx = np.argmax(val_ask)
                ask_opt[t, i] = best_idx
                ask_val = val_ask[best_idx]
            
            # The total value is the sum of bid and ask values minus inventory cost
            # Actually, the objective is: depths earned minus sum of (gamma/2) sigma^2 q^2 dt
            # The "depths earned" comes from fills. The inventory cost is separate.
            # So V[t, q] = max over bid_depth, ask_depth of [
            #   p_bid * (bid_depth + V[t+1, q+1]) + (1-p_bid) * V[t+1, q]
            #   + p_ask * (ask_depth + V[t+1, q-1]) + (1-p_ask) * V[t+1, q]
            #   - (gamma/2) sigma^2 q^2 dt
            # ]
            # But bid and ask are independent, so we can optimize them separately
            # and add the results.
            # Wait, the fills are independent, so the expected value is the sum.
            # V[t, q] = (bid_val + ask_val) - inv_cost[i]
            
            V[t, i] = bid_val + ask_val - inv_cost[i]
    
    # Now convert optimal indices to actual depths
    bid_depths_table = depth_grid[bid_opt]
    ask_depths_table = depth_grid[ask_opt]
    
    _regime_cache[key] = (bid_depths_table, ask_depths_table)
    return bid_depths_table, ask_depths_table


def quote(t, inventory, mid, params):
    """
    Quote using the precomputed optimal policy.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Determine current time step
    dt = T / steps
    t_step = int(round(t / dt))
    t_step = min(t_step, steps - 1)
    
    # Deadline: leave some margin
    deadline = 280.0  # seconds
    
    try:
        bid_table, ask_table = _solve_regime(sigma, A, k, gamma, T, steps, Q, deadline)
    except Exception:
        # Fallback to seed policy
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    # Map inventory to index
    # inventory is in [-Q, Q], index = inventory + Q
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get optimal depths
    bid_depths = bid_table[t_step, inv_idx]
    ask_depths = ask_table[t_step, inv_idx]
    
    # Ensure valid depths
    # Values below -3/k, -inf, NaN make result invalid
    min_valid = -3.0 / k
    bid_depths = np.where(bid_depths < min_valid, min_valid, bid_depths)
    ask_depths = np.where(ask_depths < min_valid, min_valid, ask_depths)
    
    # Handle boundary conditions: no bid at Q, no ask at -Q
    bid_depths = np.where(inventory == Q, np.inf, bid_depths)
    ask_depths = np.where(inventory == -Q, np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
