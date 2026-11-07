"""Optimal market making with inventory risk via HJB backward induction lookup."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

# Cache for pre-computed optimal depths
# Key: (sigma, A, k, gamma, Q, steps)
# Value: dict with 'bid' and 'ask' arrays of shape (steps, 2*Q+1)
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

    # Depth grid: from very small to large
    # The optimal depth is bounded: at depth d, fill prob ~ A*exp(-k*d)*dt
    # For large d, fill prob ~ 0, so depth beyond ~10/k is useless
    # Use a log-spaced grid for finer resolution at small depths
    n_depths = 200
    min_depth = 1e-6
    max_depth = 15.0 / k
    depth_grid = np.logspace(np.log10(min_depth), np.log10(max_depth), n_depths)

    # Precompute fill probabilities for all depths
    # p_fill[d] = 1 - exp(-A * exp(-k * d) * dt)
    exp_neg_k_d = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_neg_k_d * dt)

    # Inventory range: -Q to Q, indexed 0 to 2Q
    n_inv = 2 * Q + 1

    # Value function: V[n, i] where i is inventory index (0 corresponds to -Q)
    # V[steps, i] = 0 (terminal, no terminal charge)
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)

    # Optimal depths: bid_depth[n, i], ask_depth[n, i]
    bid_depth = np.zeros((steps, n_inv), dtype=np.float64)
    ask_depth = np.zeros((steps, n_inv), dtype=np.float64)

    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 30:
            break

        # Inventory before this step's fills: q in [-Q, Q]
        # q index: i = q + Q, so q = i - Q
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)

        # Risk penalty for this step
        risk_pen = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt

        # For each inventory state, compute optimal bid and ask depths
        # Bid fill: prob p_b, earn depth_b, inventory increases by 1 (q -> q+1)
        #   But no bid fill at q = Q
        # Ask fill: prob p_a, earn depth_a, inventory decreases by 1 (q -> q-1)
        #   But no ask fill at q = -Q
        # Fills are independent

        # For state i (q = i - Q):
        # If bid fills: new_inv = q + 1 = i - Q + 1, index = i + 1 (if i < 2Q)
        # If ask fills: new_inv = q - 1 = i - Q - 1, index = i - 1 (if i > 0)
        # If both fill: new_inv = q, index = i
        # If neither: new_inv = q, index = i

        # Expected value for given bid depth index db and ask depth index da:
        # p_b = p_fill[db], p_a = p_fill[da]
        # E[V_next] = p_b * (1-p_a) * V[n+1, i+1] + (1-p_b) * p_a * V[n+1, i-1] 
        #            + p_b * p_a * V[n+1, i] + (1-p_b) * (1-p_a) * V[n+1, i]
        #           = p_b * (1-p_a) * V[n+1, i+1] + (1-p_b) * p_a * V[n+1, i-1]
        #            + (p_b*p_a + (1-p_b)*(1-p_a)) * V[n+1, i]
        # Reward: depth_b * p_b * (1-p_a) + depth_a * p_a * (1-p_b)
        #   (bid earns depth when it fills and ask doesn't? No - each fill earns its depth independently)
        # Actually: bid fill earns depth_b, ask fill earns depth_a, independently
        # So reward = depth_b * p_b + depth_a * p_a
        # But wait: if bid fills, inventory goes up. If ask fills, inventory goes down.
        # The reward is just the depths earned.

        # Actually re-reading: "A fill earns its depth and moves inventory one unit"
        # So bid fill: earn bid_depth, inventory +1
        # Ask fill: earn ask_depth, inventory -1
        # Both independent, so expected reward = bid_depth * p_b + ask_depth * p_a

        # Total for state (n, i) with depths (db, da):
        # reward = depth_grid[db] * p_fill[db] + depth_grid[da] * p_fill[da]
        # next_val = p_b*(1-p_a)*V[n+1,i+1] + (1-p_b)*p_a*V[n+1,i-1] + (1 - p_b*(1-p_a) - (1-p_b)*p_a)*V[n+1,i]
        #          = p_b*(1-p_a)*V[n+1,i+1] + (1-p_b)*p_a*V[n+1,i-1] + (p_b*p_a + (1-p_b)*(1-p_a))*V[n+1,i]
        # Note: p_b*p_a + (1-p_b)*(1-p_a) = 1 - p_b*(1-p_a) - (1-p_b)*p_a
        # So: next_val = V[n+1,i] + p_b*(1-p_a)*(V[n+1,i+1] - V[n+1,i]) + (1-p_b)*p_a*(V[n+1,i-1] - V[n+1,i])

        # We want to maximize: reward + next_val - risk_pen[i]
        # Since risk_pen[i] is constant for given i, we maximize reward + next_val

        # For efficiency, precompute:
        V_next = V[n + 1]
        
        # For each inventory state i:
        for i in range(n_inv):
            q = i - Q
            
            # Determine valid transitions
            # Bid fill only valid if q < Q (i.e., i < 2Q)
            # Ask fill only valid if q > -Q (i.e., i > 0)
            
            # Compute V_next values for the four cases
            V_ii = V_next[i]
            V_i1 = V_next[i + 1] if i < 2 * Q else V_ii  # bid fill: q+1
            V_im1 = V_next[i - 1] if i > 0 else V_ii     # ask fill: q-1
            
            # dV_bid = V_i1 - V_ii (change in value if bid fills)
            # dV_ask = V_im1 - V_ii (change in value if ask fills)
            dV_bid = V_i1 - V_ii
            dV_ask = V_im1 - V_ii
            
            # For each pair of depths (db, da), compute:
            # obj = depth_grid[db]*p_fill[db] + depth_grid[da]*p_fill[da]
            #     + V_ii + p_fill[db]*(1-p_fill[da])*dV_bid + (1-p_fill[db])*p_fill[da]*dV_ask
            # = depth_grid[db]*p_fill[db] + depth_grid[da]*p_fill[da]
            #   + V_ii + p_fill[db]*dV_bid - p_fill[db]*p_fill[da]*dV_bid
            #   + p_fill[da]*dV_ask - p_fill[db]*p_fill[da]*dV_ask
            # = V_ii + p_fill[db]*(depth_grid[db] + dV_bid) + p_fill[da]*(depth_grid[da] + dV_ask)
            #   - p_fill[db]*p_fill[da]*(dV_bid + dV_ask)
            
            # Let f_b[db] = p_fill[db] * (depth_grid[db] + dV_bid)
            # Let f_a[da] = p_fill[da] * (depth_grid[da] + dV_ask)
            # Let g[db, da] = p_fill[db] * p_fill[da] * (dV_bid + dV_ask)
            
            # obj[db, da] = V_ii + f_b[db] + f_a[da] - p_fill[db]*p_fill[da]*(dV_bid + dV_ask)
            
            # We can separate: for fixed da, maximize over db:
            # f_b[db] - p_fill[db]*p_fill[da]*(dV_bid + dV_ask)
            # = p_fill[db] * (depth_grid[db] + dV_bid - p_fill[da]*(dV_bid + dV_ask))
            
            # This is still coupled. Let's just do a grid search with vectorization.
            
            # For all db, da pairs, compute objective
            # Use broadcasting
            pf_b = p_fill  # (n_depths,)
            pf_a = p_fill  # (n_depths,)
            dg_b = depth_grid  # (n_depths,)
            dg_a = depth_grid  # (n_depths,)
            
            # f_b[db] = pf_b[db] * (dg_b[db] + dV_bid)
            f_b = pf_b * (dg_b + dV_bid)  # (n_depths,)
            f_a = pf_a * (dg_a + dV_ask)  # (n_depths,)
            
            # coupling[db, da] = pf_b[db] * pf_a[da] * (dV_bid + dV_ask)
            # obj[db, da] = V_ii + f_b[db] + f_a[da] - coupling[db, da]
            
            # obj = V_ii + f_b[:, None] + f_a[None, :] - np.outer(pf_b, pf_a) * (dV_bid + dV_ask)
            
            # But this is n_depths x n_depths = 200x200 = 40000 per state, times 61 states = 2.44M per step
            # Times 1000 steps = 2.44B operations. Too slow.
            
            # Alternative: since the coupling term is separable in a specific way,
            # note that for fixed da, the objective over db is:
            # f_b[db] - pf_b[db] * pf_a[da] * (dV_bid + dV_ask) + (f_a[da] + V_ii)
            # = pf_b[db] * (dg_b[db] + dV_bid - pf_a[da] * (dV_bid + dV_ask)) + (f_a[da] + V_ii)
            
            # For fixed da, the optimal db maximizes:
            # pf_b[db] * (dg_b[db] + dV_bid - pf_a[da] * (dV_bid + dV_ask))
            
            # Similarly for fixed db, optimal da maximizes:
            # pf_a[da] * (dg_a[da] + dV_ask - pf_b[db] * (dV_bid + dV_ask))
            
            # This doesn't fully separate. Let's use a coarser grid.
            
            # Use n_depths = 50 for speed
            # Actually, let's use a different approach: iterate over da, for each find best db.
            # n_depths=50, n_inv=61, steps=1000: 50*61*1000 = 3.05M inner iterations, each O(n_depths)=50
            # Total: 152.5M operations. Should be OK in ~10-20 seconds.
            
            pass  # We'll implement below

        # Vectorized approach: for each i, compute the full (n_depths, n_depths) grid
        # But do it in chunks to save memory
        pass

    # Fallback: use a simpler heuristic if HJB is too slow
    # For now, let's implement a simpler version with reduced grid

    # Reset and use n_depths = 50
    n_depths = 50
    min_depth = 1e-6
    max_depth = 15.0 / k
    depth_grid = np.logspace(np.log10(min_depth), np.log10(max_depth), n_depths)
    exp_neg_k_d = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_neg_k_d * dt)

    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    bid_depth = np.zeros((steps, n_inv), dtype=np.float64)
    ask_depth = np.zeros((steps, n_inv), dtype=np.float64)

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 30:
            break

        V_next = V[n + 1]
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)

        for i in range(n_inv):
            q = i - Q
            V_ii = V_next[i]
            V_i1 = V_next[i + 1] if i < 2 * Q else V_ii
            V_im1 = V_next[i - 1] if i > 0 else V_ii
            dV_bid = V_i1 - V_ii
            dV_ask = V_im1 - V_ii

            # For each ask depth da, find optimal bid depth db
            # obj(db, da) = V_ii + p_fill[db]*(depth_grid[db] + dV_bid) + p_fill[da]*(depth_grid[da] + dV_ask)
            #              - p_fill[db]*p_fill[da]*(dV_bid + dV_ask)
            
            # For fixed da, maximize over db:
            # p_fill[db] * (depth_grid[db] + dV_bid - p_fill[da]*(dV_bid + dV_ask))
            
            # Let's compute this for all da at once
            # best_obj_for_da[da] = max_db [ p_fill[db] * (depth_grid[db] + dV_bid - p_fill[da]*(dV_bid + dV_ask)) ]
            # + p_fill[da]*(depth_grid[da] + dV_ask) + V_ii
            
            # The term inside max over db: p_fill[db] * (depth_grid[db] + dV_bid - p_fill[da]*(dV_bid + dV_ask))
            # Let c[da] = p_fill[da]*(dV_bid + dV_ask)
            # Then: p_fill[db] * (depth_grid[db] + dV_bid - c[da])
            
            c = p_fill * (dV_bid + dV_ask)  # (n_depths,)
            
            # For each da, we need max over db of p_fill[db] * (depth_grid[db] + dV_bid - c[da])
            # This is equivalent to: for each da, compute array p_fill * (depth_grid + dV_bid - c[da]) and take max
            
            # Vectorize: create matrix (n_depths_da, n_depths_db)
            # M[da, db] = p_fill[db] * (depth_grid[db] + dV_bid - c[da])
            # = p_fill[db] * (depth_grid[db] + dV_bid) - p_fill[db] * c[da]
            # = (p_fill * (depth_grid + dV_bid))[db] - p_fill[db] * c[da]
            
            f_b_base = p_fill * (depth_grid + dV_bid)  # (n_depths,)
            # M[da, db] = f_b_base[db] - p_fill[db] * c[da]
            # = f_b_base[None, :] - np.outer(c, p_fill)
            
            M = f_b_base[None, :] - np.outer(c, p_fill)  # (n_depths, n_depths)
            # M[da, db] is the contribution from bid side given ask depth index da
            
            best_db_idx = np.argmax(M, axis=1)  # (n_depths,)
            best_bid_val = M[np.arange(n_depths), best_db_idx]  # (n_depths,)
            
            # Total obj for each da:
            f_a = p_fill * (depth_grid + dV_ask)  # (n_depths,)
            total_obj = V_ii + best_bid_val + f_a  # (n_depths,)
            
            best_da_idx = np.argmax(total_obj)
            best_db_idx_final = best_db_idx[best_da_idx]
            
            bid_depth[n, i] = depth_grid[best_db_idx_final]
            ask_depth[n, i] = depth_grid[best_da_idx]
            V[n, i] = total_obj[best_da_idx]

    _OPT_TABLES[key] = {'bid': bid_depth, 'ask': ask_depth}
    return _OPT_TABLES[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) for each path.
    """
    table = _get_optimal_table(params)
    steps = params["steps"]
    Q = params["Q"]
    
    # Determine current step from t
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # inventory: (4000,) ints in [-Q, Q]
    # Index into table: i = inventory + Q
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    bid_depths = table['bid'][n, inv_idx]
    ask_depths = table['ask'][n, inv_idx]
    
    # Ensure no bid fill at q=Q, no ask fill at q=-Q
    # At q=Q, set bid depth to inf (no bid quote)
    # At q=-Q, set ask depth to inf (no ask quote)
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    # Validate: values must be > -3/k, not -inf, not NaN
    min_val = -3.0 / params["k"]
    # Replace any invalid values
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths <= min_val), min_val + 1e-6, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths <= min_val), min_val + 1e-6, ask_depths)
    # But inf is allowed
    bid_depths = np.where(np.isinf(bid_depths) & (bid_depths > 0), np.inf, bid_depths)
    ask_depths = np.where(np.isinf(ask_depths) & (ask_depths > 0), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
