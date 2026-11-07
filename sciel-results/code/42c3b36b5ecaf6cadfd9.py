"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
from functools import lru_cache
import time

# Pre-compute the optimal policy table at module load time
_POLICY_CACHE = {}

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time HJB equation.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    # Value function: V[n, q] = optimal expected objective from step n with inventory q
    # We work backwards from n = steps-1 down to n = 0
    
    # Terminal condition: no terminal charge, so V[steps, q] = 0 for all q
    V_next = np.zeros(n_inv)
    
    # We need to find optimal bid_depth and ask_depth for each (step, inventory)
    # The depth must be >= -3/k (from the validity constraint)
    # Let's use a grid of candidate depths
    
    # Candidate depths: we'll use a log-spaced or linear grid
    # The optimal depth is likely in a reasonable range. Let's use a grid.
    # depth must be > -3/k. Let's use depths from a small positive value up to some max.
    # Actually, the constraint is depth >= -3/k, but depth represents distance from mid.
    # A bid_depth of d means we quote at mid - d. For a bid to be below mid, d > 0.
    # But the problem says "distances below and above mid to quote this step"
    # So bid_depth is the distance below mid (positive = below mid)
    # ask_depth is the distance above mid (positive = above mid)
    
    # The fill probability is 1 - exp(-A * exp(-k * depth) * dt)
    # For larger depth, exp(-k*depth) is smaller, so fill probability is lower.
    # The "earnings" from a fill is the depth itself.
    
    # Let's use a grid of candidate depths
    # The optimal depth balances the expected earnings from fills against the inventory risk
    
    # Let's use a fine grid of depths
    depth_min = 1e-6
    depth_max = 10.0 / k  # reasonable upper bound
    n_depths = 200
    depths = np.linspace(depth_min, depth_max, n_depths)
    
    # Precompute fill probabilities for each depth
    # p_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    exp_kd = np.exp(-k * depths)
    p_fill = 1.0 - np.exp(-A * exp_kd * dt)
    
    # Backward induction
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    for n in range(steps - 1, -1, -1):
        # For each inventory level, find optimal bid and ask depths
        # V[n, q] = max over bid_depth, ask_depth of:
        #   - (gamma/2) * sigma^2 * q^2 * dt
        #   + p_bid * bid_depth * (1 if q < Q else 0)  [earn bid_depth if bid fills and q < Q]
        #   + p_ask * ask_depth * (1 if q > -Q else 0)  [earn ask_depth if ask fills and q > -Q]
        #   + E[V[n+1, q']]
        # where q' depends on which fills occur
        
        # Actually, the mid moves by sigma*sqrt(dt)*N(0,1), but since we're looking at
        # expected value and the mid movement doesn't directly affect the depth earnings
        # (depths are relative to mid), we can ignore the mid movement for the depth optimization.
        # The inventory risk term depends on q, not on mid.
        
        # Wait, I need to reconsider. The objective is:
        # sum over steps of [depths earned] - sum over steps of (gamma/2)*sigma^2*q^2*dt
        # where q is the inventory BEFORE that step's fills.
        
        # So at step n with inventory q:
        # - We pay (gamma/2)*sigma^2*q^2*dt
        # - We may earn bid_depth if bid fills (and q < Q)
        # - We may earn ask_depth if ask fills (and q > -Q)
        # - Inventory changes: q' = q + 1 if bid fills, q - 1 if ask fills, q if neither or both
        
        # The mid movement doesn't affect the current step's earnings or the value function
        # in a way that depends on mid (since depths are relative). So we can ignore it.
        
        # For each inventory level q (indexed by i, where q = inv_range[i]):
        for i in range(n_inv):
            q = inv_range[i]
            
            # Inventory cost
            inv_cost = -0.5 * gamma * sigma**2 * q**2 * dt
            
            # Can we fill bid? Only if q < Q
            can_bid = q < Q
            # Can we fill ask? Only if q > -Q
            can_ask = q > -Q
            
            # If we can't bid, the optimal bid_depth is np.inf (no quote)
            # If we can't ask, the optimal ask_depth is np.inf (no quote)
            
            # We need to maximize over bid_depth and ask_depth
            # The problem is that bid and ask fills are independent, so we have 4 cases:
            # 1. No fills: prob (1-p_bid)(1-p_ask), q' = q
            # 2. Bid fill only: prob p_bid(1-p_ask), q' = q+1 (if can_bid)
            # 3. Ask fill only: prob (1-p_bid)p_ask, q' = q-1 (if can_ask)
            # 4. Both fills: prob p_bid*p_ask, q' = q (if both can fill)
            
            # The expected value is:
            # inv_cost + p_bid*bid_depth*(can_bid) + p_ask*ask_depth*(can_ask)
            #   + (1-p_bid)(1-p_ask) * V_next[q]
            #   + p_bid(1-p_ask) * V_next[q+1] * can_bid
            #   + (1-p_bid)p_ask * V_next[q-1] * can_ask
            #   + p_bid*p_ask * V_next[q] * (can_bid and can_ask)
            
            # Since bid and ask are independent, we can separate the optimization.
            # Actually, no, because the future value V_next depends on the combined outcome.
            # But we can still optimize jointly.
            
            # Let's do a grid search over bid_depth and ask_depth
            # For efficiency, we can precompute the contributions
            
            # If can_bid, we choose from depths; otherwise bid_depth = inf
            # If can_ask, we choose from depths; otherwise ask_depth = inf
            
            if can_bid and can_ask:
                # Both sides can fill
                # p_bid depends on bid_depth, p_ask depends on ask_depth
                # Expected value = inv_cost 
                #   + p_bid(bd)*bd + p_ask(ad)*ad
                #   + (1-p_bid)(1-p_ask)*V_next[i]
                #   + p_bid(1-p_ask)*V_next[i+1]
                #   + (1-p_bid)p_ask*V_next[i-1]
                #   + p_bid*p_ask*V_next[i]
                
                # Simplify:
                # = inv_cost
                #   + p_bid*bd + p_ask*ad
                #   + V_next[i] - p_bid*V_next[i] - p_ask*V_next[i] + p_bid*p_ask*V_next[i]
                #   + p_bid*V_next[i+1] - p_bid*p_ask*V_next[i+1]
                #   + p_ask*V_next[i-1] - p_bid*p_ask*V_next[i-1]
                # = inv_cost
                #   + p_bid*(bd - V_next[i] + V_next[i+1])
                #   + p_ask*(ad - V_next[i] + V_next[i-1])
                #   + p_bid*p_ask*(V_next[i] - V_next[i+1] - V_next[i-1])
                #   + V_next[i]
                
                # This is a 2D optimization. Let's do a grid search.
                # For each bid_depth, find the best ask_depth, and vice versa.
                
                # Actually, let's just do a full grid search since n_depths is small enough.
                # But 200x200 = 40000 per inventory level, and 61 inventory levels, and 1000 steps = 2.44 billion. Too slow.
                
                # Let's reduce n_depths or use a smarter approach.
                # Let's use n_depths = 50 for now.
                
                # Actually, let's think about this differently.
                # The term p_bid*(bd - V_next[i] + V_next[i+1]) + p_ask*(ad - V_next[i] + V_next[i-1])
                # + p_bid*p_ask*(V_next[i] - V_next[i+1] - V_next[i-1])
                
                # Let c1 = V_next[i+1] - V_next[i]
                # Let c2 = V_next[i-1] - V_next[i]
                # Let c3 = V_next[i] - V_next[i+1] - V_next[i-1] = -c1 - c2
                
                # So the expression is:
                # p_bid*(bd + c1) + p_ask*(ad + c2) + p_bid*p_ask*(-c1 - c2)
                # = p_bid*bd + p_ask*ad + p_bid*c1 + p_ask*c2 - p_bid*p_ask*(c1+c2)
                
                # For a fixed bid_depth (and thus p_bid), this is linear in p_ask and ad.
                # But p_ask is a function of ad, so it's not separable.
                
                # Let's just do a 1D grid search for each side, assuming the other is fixed,
                # and iterate. Or just do a coarse grid.
                
                # For now, let's use a smaller grid: n_depths = 30
                depths_small = np.linspace(depth_min, depth_max, 30)
                exp_kd_s = np.exp(-k * depths_small)
                p_fill_s = 1.0 - np.exp(-A * exp_kd_s * dt)
                
                c1 = V_next[i+1] - V_next[i]
                c2 = V_next[i-1] - V_next[i]
                c3 = -c1 - c2
                
                # For each bid_depth, find best ask_depth
                best_val = -np.inf
                best_bd = depths_small[0]
                best_ad = depths_small[0]
                
                # This is O(n_depths^2) per inventory level
                # 30^2 = 900 per inventory, 61 inventory, 1000 steps = 54.9 million. Should be OK.
                
                for j, bd in enumerate(depths_small):
                    pb = p_fill_s[j]
                    # For each ask_depth
                    for l, ad in enumerate(depths_small):
                        pa = p_fill_s[l]
                        val = (pb * bd + pa * ad + pb * c1 + pa * c2 + pb * pa * c3)
                        if val > best_val:
                            best_val = val
                            best_bd = bd
                            best_ad = ad
                
                bid_table[n, i] = best_bd
                ask_table[n, i] = best_ad
                V_cur = inv_cost + V_next[i] + best_val
                # Store in a temporary array
                if not hasattr(_compute_policy_table, '_V_cur'):
                    _compute_policy_table._V_cur = np.zeros(n_inv)
                _compute_policy_table._V_cur[i] = V_cur
                
            elif can_bid and not can_ask:
                # Only bid can fill
                # Expected value = inv_cost + p_bid*bd + (1-p_bid)*V_next[i] + p_bid*V_next[i+1]
                # = inv_cost + V_next[i] + p_bid*(bd - V_next[i] + V_next[i+1])
                # = inv_cost + V_next[i] + p_bid*(bd + c1) where c1 = V_next[i+1] - V_next[i]
                
                c1 = V_next[i+1] - V_next[i]
                best_val = -np.inf
                best_bd = depths[0]
                
                for j, bd in enumerate(depths):
                    pb = p_fill[j]
                    val = pb * (bd + c1)
                    if val > best_val:
                        best_val = val
                        best_bd = bd
                
                bid_table[n, i] = best_bd
                ask_table[n, i] = np.inf
                V_cur = inv_cost + V_next[i] + best_val
                _compute_policy_table._V_cur[i] = V_cur
                
            elif can_ask and not can_bid:
                # Only ask can fill
                c2 = V_next[i-1] - V_next[i]
                best_val = -np.inf
                best_ad = depths[0]
                
                for j, ad in enumerate(depths):
                    pa = p_fill[j]
                    val = pa * (ad + c2)
                    if val > best_val:
                        best_val = val
                        best_ad = ad
                
                bid_table[n, i] = np.inf
                ask_table[n, i] = best_ad
                V_cur = inv_cost + V_next[i] + best_val
                _compute_policy_table._V_cur[i] = V_cur
                
            else:
                # Neither can fill
                bid_table[n, i] = np.inf
                ask_table[n, i] = np.inf
                V_cur = inv_cost + V_next[i]
                _compute_policy_table._V_cur[i] = V_cur
        
        V_next = _compute_policy_table._V_cur.copy()
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths based on pre-computed policy table."""
    steps = params['steps']
    Q = params['Q']
    
    # Get current step
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index: q in [-Q, Q] -> i in [0, 2Q]
    inv_idx = inventory + Q
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get depths
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    # Ensure values are valid: >= -3/k, not -inf, not NaN
    k = params['k']
    min_depth = -3.0 / k
    
    # Replace any values below min_depth with min_depth
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Replace NaN with a default value
    bid_depths = np.nan_to_num(bid_depths, nan=1.0/k, posinf=np.inf, neginf=min_depth)
    ask_depths = np.nan_to_num(ask_depths, nan=1.0/k, posinf=np.inf, neginf=min_depth)
    
    return bid_depths, ask_depths
