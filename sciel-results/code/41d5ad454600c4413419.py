"""Optimal market making with inventory risk via pre-computed HJB lookup table."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache

# Global cache for lookup tables per regime
_LUT_CACHE = {}

def _build_lut(sigma, A, k, gamma, T, steps, Q, n_depths=80):
    """
    Pre-compute optimal bid/ask depths for each (step, inventory) pair
    using backward induction on a discrete state space.
    
    We discretize depth into a small grid and inventory into [-Q, Q].
    For each state (n, q), we choose bid_depth and ask_depth from the grid
    to maximize expected continuation value.
    
    Returns:
        bid_lut: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_lut: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    # Depth grid: from very small to some max
    # The depth should be positive. Let's use a log-spaced or linear grid.
    # Depths near 0 give high fill probability but low reward per fill.
    # Depths large give low fill probability but high reward.
    # Optimal depth is typically O(1/k).
    
    max_depth = 5.0 / k  # generous upper bound
    min_depth = 1e-4
    depth_grid = np.linspace(min_depth, max_depth, n_depths)
    
    dt = T / steps
    inv_range = range(-Q, Q + 1)
    inv_offset = Q
    n_inv = len(inv_range)
    
    # V[n, q] = optimal expected future reward from step n onward, given inventory q
    # We work backward from n = steps down to 0.
    # At the terminal step (n = steps), V = 0 (no terminal charge).
    
    V_next = np.zeros(n_inv)  # V at step n+1
    
    bid_lut = np.zeros((steps, n_inv))
    ask_lut = np.zeros((steps, n_inv))
    
    # Pre-compute probabilities for each depth
    # p_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    p_bid = np.zeros(n_depths)
    p_ask = np.zeros(n_depths)
    for i, d in enumerate(depth_grid):
        rate = A * np.exp(-k * d) * dt
        p = 1.0 - np.exp(-rate)
        p_bid[i] = p
        p_ask[i] = p  # same formula for ask
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for q_idx, q in enumerate(inv_range):
            best_val = -np.inf
            best_bid_d = depth_grid[0]
            best_ask_d = depth_grid[0]
            
            # If at boundary, one side is forced to no quote (inf)
            # But we can still choose depth for the other side.
            # Actually, the problem says "no bid fill at q=Q, no ask fill at q=-Q".
            # This means if q == Q, bid fill probability is 0.
            # If q == -Q, ask fill probability is 0.
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # Iterate over depth choices
            # For speed, we can vectorize over depth grid for each side
            # But with n_depths=80 and n_inv=61, steps=1000, that's 1000*61*80*80 = 388M ops
            # Too slow in Python. We need to vectorize.
            
            # Let's vectorize over depth pairs.
            # For each bid depth i and ask depth j, compute expected value.
            
            if can_bid and can_ask:
                # Both sides active
                # V = (1-pb)(1-pa) * V_next[q] 
                #   + pb*(1-pa) * (depth_i + V_next[q-1])
                #   + (1-pb)*pa * (depth_j + V_next[q+1])
                #   + pb*pa * (depth_i + depth_j + V_next[q])
                # Wait, inventory changes: bid fill decreases inventory by 1? 
                # "A fill earns its depth and moves inventory one unit"
                # Bid fill: we sell, so inventory decreases? Or increases?
                # Typically, if we are market maker, bid fill means we buy, so inventory increases.
                # Let's assume: bid fill -> q+1, ask fill -> q-1.
                # Objective: depths earned minus sum (gamma/2) sigma^2 q^2 dt
                # The cost is incurred based on q BEFORE fills.
                
                # So immediate reward = depth_earned - (gamma/2)*sigma^2*q^2*dt
                # Then next state inventory depends on fills.
                
                pb = p_bid  # shape (n_depths,)
                pa = p_ask  # shape (n_depths,)
                
                # Vectorized computation
                # For each i (bid depth) and j (ask depth):
                # val[i,j] = (1-pb[i])(1-pa[j]) * V_next[q_idx]
                #          + pb[i]*(1-pa[j]) * (depth_grid[i] + V_next[q_idx+1])
                #          + (1-pb[i])*pa[j] * (depth_grid[j] + V_next[q_idx-1])
                #          + pb[i]*pa[j] * (depth_grid[i] + depth_grid[j] + V_next[q_idx])
                #          - (gamma/2)*sigma^2*q^2*dt
                
                cost = (gamma / 2.0) * sigma**2 * q**2 * dt
                
                # V_next indices
                v0 = V_next[q_idx]
                v_plus = V_next[q_idx + 1] if q_idx + 1 < n_inv else V_next[q_idx]
                v_minus = V_next[q_idx - 1] if q_idx - 1 >= 0 else V_next[q_idx]
                
                # Build matrices
                # This is 80x80 = 6400 elements per (n, q), times 61*1000 = 388M
                # Still too slow in pure Python/NumPy if done naively.
                # Let's reduce n_depths to 30. 30*30*61*1000 = 5.5M, manageable.
                
                # Actually, let's use a smaller grid: n_depths = 30
                # And pre-compute for each q the best depth.
                
                # Alternative: use a coarser approach. Since the optimal depth doesn't change much
                # with inventory for small inventories, we can use a reduced set of candidate depths.
                
                # Let's try n_depths = 40 for now and see.
                # For this prototype, let's just do it with moderate grid size.
                
                # Vectorize over i for fixed j, or use full matrix.
                # Full matrix: 40x40 = 1600 per (n,q), 61*1000 = 61000 states, total 97.6M ops.
                # In NumPy, this should be fast enough if vectorized properly.
                
                pb_col = pb[:, None]  # (n_depths, 1)
                pa_row = pa[None, :]  # (1, n_depths)
                d_i = depth_grid[:, None]  # (n_depths, 1)
                d_j = depth_grid[None, :]  # (1, n_depths)
                
                term00 = (1 - pb_col) * (1 - pa_row) * v0
                term10 = pb_col * (1 - pa_row) * (d_i + v_plus)
                term01 = (1 - pb_col) * pa_row * (d_j + v_minus)
                term11 = pb_col * pa_row * (d_i + d_j + v0)
                
                vals = term00 + term10 + term01 + term11 - cost
                
                idx = np.argmax(vals)
                best_val = vals.flat[idx]
                i_best, j_best = divmod(idx, n_depths)
                best_bid_d = depth_grid[i_best]
                best_ask_d = depth_grid[j_best]
                
            elif can_bid:
                # q == Q, no ask fill
                pb = p_bid
                d_i = depth_grid
                v0 = V_next[q_idx]
                v_plus = V_next[q_idx]  # stays same if no fill? No, bid fill -> q+1 but q=Q so no fill
                # Actually if q=Q, bid fill prob is 0, so inventory stays Q.
                # So V_next[q_idx] is the only next state.
                
                # val[i] = (1-pb[i]) * V_next[q_idx] + pb[i] * (0 + V_next[q_idx]) - cost
                # = V_next[q_idx] - cost
                # So depth doesn't matter! Any depth gives same value.
                # But wait, if bid fill is impossible, then no depth is earned.
                # So val[i] = V_next[q_idx] - cost for all i.
                # We can pick any depth, say the one that minimizes risk? 
                # Since no fill, it doesn't matter. Let's pick a moderate depth.
                best_val = v0 - cost
                best_bid_d = depth_grid[len(depth_grid) // 2]
                best_ask_d = np.inf  # no ask quote? But can_ask is False, so we don't quote ask.
                # Actually, if can_ask is False, we should set ask depth to inf (no quote).
                # But the problem says "no ask fill at q=-Q", not "no ask quote".
                # Let me re-read: "no bid fill at q = Q, no ask fill at q = -Q".
                # This means the fill probability is zero, but we can still quote.
                # If we quote but no fill happens, we earn nothing.
                # So it's fine to quote, but it doesn't matter.
                
            elif can_ask:
                # q == -Q, no bid fill
                best_val = v0 - cost
                best_bid_d = np.inf
                best_ask_d = depth_grid[len(depth_grid) // 2]
                
            else:
                # Should not happen since -Q < Q
                best_val = -cost
                best_bid_d = np.inf
                best_ask_d = np.inf
            
            bid_lut[n, q_idx] = best_bid_d
            ask_lut[n, q_idx] = best_ask_d
            V_next[q_idx] = best_val
    
    return bid_lut, ask_lut


def quote(t, inventory, mid, params):
    """
    Quote using pre-computed HJB lookup table.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Create cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _LUT_CACHE:
        # Use a reduced depth grid for speed
        bid_lut, ask_lut = _build_lut(sigma, A, k, gamma, T, steps, Q, n_depths=40)
        _LUT_CACHE[key] = (bid_lut, ask_lut)
    else:
        bid_lut, ask_lut = _LUT_CACHE[key]
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to index
    inv_offset = Q
    inv_idx = np.clip(inventory + inv_offset, 0, 2 * Q)
    
    # Get depths from lookup table
    bid_depths = bid_lut[n, inv_idx]
    ask_depths = ask_lut[n, inv_idx]
    
    # Ensure valid values: clip to [min, max] and handle inf
    min_depth = 1e-4
    max_depth = 10.0 / k
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
