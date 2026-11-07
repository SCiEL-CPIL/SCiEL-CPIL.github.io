"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0

_OPT_TABLES = {}

def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation with a reduced depth grid.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # Improved depth grid: concentrate resolution around expected optimal region
    # Optimal depth is typically in [0.5/k, 3/k] range
    n_depths = 400
    # Use a piecewise grid: fine near 1/k, coarser at extremes
    depth_min = 0.005
    depth_max = 15.0 / k
    # Create a log-spaced grid with extra points near 1/k
    depths = np.geomspace(depth_min, depth_max, n_depths)
    
    # Pre-compute fill probabilities for each depth
    # p_fill = 1 - exp(-A * exp(-k * depth) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)  # shape (n_depths,)
    
    # Clip fill probs to valid range [0, 1]
    fill_probs = np.clip(fill_probs, 0.0, 1.0)
    
    # State space: inventory in [-Q, Q], step in [0, steps-1]
    n_inv = 2 * Q + 1
    inv_indices = np.arange(-Q, Q + 1)  # shape (n_inv,)
    
    # Value function: V[step, inv_idx] = optimal expected value from step to end
    # We'll store value and optimal depths
    # V shape: (steps, n_inv)
    # opt_bid_depth_idx shape: (steps, n_inv)
    # opt_ask_depth_idx shape: (steps, n_inv)
    
    # Initialize value at terminal step to 0
    V = np.zeros((steps + 1, n_inv))
    opt_bid_idx = np.zeros((steps, n_inv), dtype=np.int32)
    opt_ask_idx = np.zeros((steps, n_inv), dtype=np.int32)
    
    # Pre-compute holding costs for all inventory levels
    holding_cost = 0.5 * gamma * sigma**2 * inv_indices**2 * dt  # shape (n_inv,)
    
    # Backward induction - vectorized over inventory
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 15.0:
            # Fallback: use uniform depths
            fallback_depth = 1.0 / k
            fb_idx = np.argmin(np.abs(depths - fallback_depth))
            opt_bid_idx[n, :] = fb_idx
            opt_ask_idx[n, :] = fb_idx
            continue
        
        V_next = V[n+1, :]  # shape (n_inv,)
        
        # Bid side: for each inventory i, compute optimal bid depth
        # bid_vals[i, j] = fill_probs[j] * depths[j] + V_next[i] + fill_probs[j] * (V_next[i+1] - V_next[i])
        # For i = n_inv-1 (q=Q), no bid fill possible, so just maximize fill_probs*depths
        
        # V_next_shifted for bid: V_next[i+1] for i in 0..n_inv-2, 0 for i=n_inv-1
        V_next_bid = np.zeros(n_inv)
        V_next_bid[:n_inv-1] = V_next[1:]  # V_next[i+1] for i < n_inv-1
        
        # bid_vals shape: (n_inv, n_depths)
        # = fill_probs[None,:] * depths[None,:] + V_next[:,None] + fill_probs[None,:] * (V_next_bid[:,None] - V_next[:,None])
        bid_vals = fill_probs[None, :] * depths[None, :] + V_next[:, None] + fill_probs[None, :] * (V_next_bid[:, None] - V_next[:, None])
        opt_bid_idx[n, :] = np.argmax(bid_vals, axis=1)
        
        # Ask side: for each inventory i, compute optimal ask depth
        # V_next_ask[i] = V_next[i-1] for i > 0, 0 for i=0
        V_next_ask = np.zeros(n_inv)
        V_next_ask[1:] = V_next[:-1]  # V_next[i-1] for i > 0
        
        ask_vals = fill_probs[None, :] * depths[None, :] + V_next[:, None] + fill_probs[None, :] * (V_next_ask[:, None] - V_next[:, None])
        opt_ask_idx[n, :] = np.argmax(ask_vals, axis=1)
        
        # Compute V[n, i] using the optimal depths
        best_bid = opt_bid_idx[n, :]
        best_ask = opt_ask_idx[n, :]
        
        p_bid = fill_probs[best_bid]  # shape (n_inv,)
        p_ask = fill_probs[best_ask]  # shape (n_inv,)
        d_bid = depths[best_bid]      # shape (n_inv,)
        d_ask = depths[best_ask]      # shape (n_inv,)
        
        # EV_bid: only for i < n_inv-1
        EV_bid = np.zeros(n_inv)
        EV_bid[:n_inv-1] = p_bid[:n_inv-1] * (d_bid[:n_inv-1] + V_next[1:])
        
        # EV_ask: only for i > 0
        EV_ask = np.zeros(n_inv)
        EV_ask[1:] = p_ask[1:] * (d_ask[1:] + V_next[:-1])
        
        EV_none = (1.0 - p_bid) * (1.0 - p_ask) * V_next
        
        V[n, :] = -holding_cost + EV_bid + EV_ask + EV_none
    
    # Build lookup tables for actual depths
    bid_depths_table = depths[opt_bid_idx]  # shape (steps, n_inv)
    ask_depths_table = depths[opt_ask_idx]  # shape (steps, n_inv)
    
    # Refinement pass: for each (step, inventory), do a local search around the coarse optimum
    # Use a finer grid centered on the coarse-optimal depth
    if time.time() < _DEADLINE - 30.0:
        n_refine = 20
        for n in range(steps):
            for i in range(n_inv):
                # Bid refinement
                coarse_bid = bid_depths_table[n, i]
                # Create local grid around coarse bid
                local_min = max(0.001, coarse_bid * 0.5)
                local_max = min(depth_max, coarse_bid * 2.0)
                local_depths_bid = np.geomspace(local_min, local_max, n_refine)
                local_fill_bid = 1.0 - np.exp(-A * np.exp(-k * local_depths_bid) * dt)
                local_fill_bid = np.clip(local_fill_bid, 0.0, 1.0)
                
                if i < n_inv - 1:
                    V_nb = V[n+1, i+1]
                    V_ni = V[n+1, i]
                    local_bid_vals = local_fill_bid * local_depths_bid + V_ni + local_fill_bid * (V_nb - V_ni)
                else:
                    local_bid_vals = local_fill_bid * local_depths_bid
                
                best_local_bid = local_depths_bid[np.argmax(local_bid_vals)]
                bid_depths_table[n, i] = best_local_bid
                
                # Ask refinement
                coarse_ask = ask_depths_table[n, i]
                local_min = max(0.001, coarse_ask * 0.5)
                local_max = min(depth_max, coarse_ask * 2.0)
                local_depths_ask = np.geomspace(local_min, local_max, n_refine)
                local_fill_ask = 1.0 - np.exp(-A * np.exp(-k * local_depths_ask) * dt)
                local_fill_ask = np.clip(local_fill_ask, 0.0, 1.0)
                
                if i > 0:
                    V_na = V[n+1, i-1]
                    V_ni = V[n+1, i]
                    local_ask_vals = local_fill_ask * local_depths_ask + V_ni + local_fill_ask * (V_na - V_ni)
                else:
                    local_ask_vals = local_fill_ask * local_depths_ask
                
                best_local_ask = local_depths_ask[np.argmax(local_ask_vals)]
                ask_depths_table[n, i] = best_local_ask
    
    result = {
        'bid': bid_depths_table,
        'ask': ask_depths_table,
        'depths': depths,
        'n_inv': n_inv,
    }
    _OPT_TABLES[key] = result
    return result


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths using pre-computed HJB optimal policy.
    Returns (bid_depths, ask_depths) each of shape (len(inventory),).
    """
    table = _get_optimal_table(params)
    
    Q = params["Q"]
    steps = params["steps"]
    
    # Determine current step index
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to index: inv_idx = q + Q
    inv_idx = inventory + Q
    # Clamp to valid range
    inv_idx = np.clip(inv_idx, 0, table['n_inv'] - 1)
    
    # Look up depths
    bid_depths = table['bid'][n, inv_idx]
    ask_depths = table['ask'][n, inv_idx]
    
    # Ensure depths are valid: must be > -3/k, not -inf, not NaN
    k = params["k"]
    min_depth = -3.0 / k + 1e-10  # just above the invalid threshold
    
    # Replace any invalid values with a safe default
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_depth), bid_depths, 1.0/k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_depth), ask_depths, 1.0/k)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
