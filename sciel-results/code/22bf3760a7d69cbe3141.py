"""Optimal market making with inventory risk using backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # safety margin before 330s kill

_OPT_TABLES = {}


def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation.
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

    # Depth grid: use a log-spaced grid for depths
    # Depths should be positive; we'll search over a range
    # The optimal depth is typically around 1/k to a few times 1/k
    # Let's use a grid of depths from very small to large
    min_depth = 0.01
    max_depth = 10.0 / k  # upper bound
    n_depths = 60
    depth_grid = np.geomspace(min_depth, max_depth, n_depths)

    # Inventory range: -Q to Q
    inv_range = np.arange(-Q, Q + 1)
    n_inv = len(inv_range)

    # V[step, inv_idx] = optimal expected value from step to end given inventory
    # We work backwards from step=steps (terminal) to step=0
    # V[steps, :] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_inv))

    # Store optimal depths for each step and inventory
    # bid_depths[step, inv_idx] and ask_depths[step, inv_idx]
    bid_depths = np.full((steps, n_inv), np.inf)
    ask_depths = np.full((steps, n_inv), np.inf)

    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Time-to-go for the exponential hazard
        # P(bid hit) = 1 - exp(-A * exp(-k * bid_depth) * dt)
        # P(ask hit) = 1 - exp(-A * exp(-k * ask_depth) * dt)
        
        # For each inventory level, find optimal bid and ask depths
        for inv_idx in range(n_inv):
            q = inv_range[inv_idx]
            
            # Inventory cost for this step
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            best_bid_depth = np.inf
            best_ask_depth = np.inf
            best_val = -np.inf

            # Try all combinations of bid and ask depths
            # This is O(n_depths^2) per state, which could be slow
            # Let's try to be smarter: optimize bid and ask separately?
            # Actually, bid and ask are independent in terms of fill probability,
            # but the value function couples them through the inventory change.
            
            # For a given bid_depth and ask_depth:
            # p_bid = 1 - exp(-A * exp(-k * bid_depth) * dt)
            # p_ask = 1 - exp(-A * exp(-k * ask_depth) * dt)
            # 
            # Expected value:
            # - inv_cost
            # + p_bid * (bid_depth + V[n+1, q+1] if q < Q else bid_depth)
            # + p_ask * (-ask_depth + V[n+1, q-1] if q > -Q else -ask_depth)
            # + (1 - p_bid) * (1 - p_ask) * V[n+1, q]
            # + p_bid * p_ask * V[n+1, q]  # both fill, inventory unchanged
            # Wait, if bid fills, q -> q+1; if ask fills, q -> q-1
            # If both fill, q -> q (net zero change)
            
            # Let me rewrite:
            # E[V_next] = p_bid*(1-p_ask)*V[n+1, q+1'] + p_ask*(1-p_bid)*V[n+1, q-1'] + (p_bid*p_ask + (1-p_bid)*(1-p_ask))*V[n+1, q]
            # where q+1' is min(q+1, Q) and q-1' is max(q-1, -Q)
            
            # Plus the cash flows:
            # p_bid * bid_depth - p_ask * ask_depth
            
            q_plus = min(q + 1, Q)
            q_minus = max(q - 1, -Q)
            q_plus_idx = q_plus + Q
            q_minus_idx = q_minus + Q
            q_idx = q + Q
            
            V_next_q = V[n+1, q_idx]
            V_next_q_plus = V[n+1, q_plus_idx]
            V_next_q_minus = V[n+1, q_minus_idx]
            
            for i_b in range(n_depths):
                bd = depth_grid[i_b]
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bd) * dt)
                
                for i_a in range(n_depths):
                    ad = depth_grid[i_a]
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
                    
                    # Expected next value
                    p_both = p_bid * p_ask
                    p_bid_only = p_bid * (1.0 - p_ask)
                    p_ask_only = p_ask * (1.0 - p_bid)
                    p_none = (1.0 - p_bid) * (1.0 - p_ask)
                    
                    E_V_next = p_both * V_next_q + p_bid_only * V_next_q_plus + p_ask_only * V_next_q_minus + p_none * V_next_q
                    
                    # Cash flow
                    cash = p_bid * bd - p_ask * ad
                    
                    total = -inv_cost + cash + E_V_next
                    
                    if total > best_val:
                        best_val = total
                        best_bid_depth = bd
                        best_ask_depth = ad
            
            V[n, inv_idx] = best_val
            bid_depths[n, inv_idx] = best_bid_depth
            ask_depths[n, inv_idx] = best_ask_depth
            
            # Check deadline
            if time.time() > _DEADLINE - 30.0:
                break
        
        if time.time() > _DEADLINE - 30.0:
            break

    _OPT_TABLES[key] = (bid_depths, ask_depths, inv_range)
    return _OPT_TABLES[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return seed-like depths
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    # Determine current step
    dt = T / steps
    step = int(round(t / dt))
    step = max(0, min(step, steps - 1))

    # Get or compute optimal table
    bid_table, ask_table, inv_range = _get_optimal_table(params)

    n_paths = len(inventory)
    bid_depths = np.full(n_paths, np.inf)
    ask_depths = np.full(n_paths, np.inf)

    # For each path, look up the optimal depth
    for i in range(n_paths):
        q = inventory[i]
        if q < -Q or q > Q:
            continue
        inv_idx = q + Q
        bd = bid_table[step, inv_idx]
        ad = ask_table[step, inv_idx]
        
        # Boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
        if q >= Q:
            bd = np.inf
        if q <= -Q:
            ad = np.inf
        
        bid_depths[i] = bd
        ask_depths[i] = ad

    # Ensure depths are valid (above -3/k, not NaN, not -inf)
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
