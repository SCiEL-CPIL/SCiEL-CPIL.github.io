"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
import time

# Module-level cache for policy tables
_policy_cache = {}
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via backward induction on the discrete-time model.
    
    The state is (step n, inventory q). At each step we choose bid_depth b >= 0 and ask_depth a >= 0.
    
    Objective per path: sum of (depths earned from fills) - sum over steps of (gamma/2)*sigma^2*q^2*dt
    
    Fill probability for bid: 1 - exp(-A * exp(-k * b) * dt)
    Fill probability for ask: 1 - exp(-A * exp(-k * a) * dt)
    
    When a bid fill occurs: earn b, inventory decreases by 1 (q -> q-1)
    When an ask fill occurs: earn a, inventory increases by 1 (q -> q+1)
    No bid fill at q = Q (can't sell more), no ask fill at q = -Q (can't buy more).
    
    Mid moves but doesn't affect the depth choice in this formulation since we quote relative to mid.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _policy_cache:
        return _policy_cache[cache_key]
    
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function: V[n, q] = optimal expected objective from step n with inventory q
    # We work backwards from n = steps-1 to n = 0
    # At n = steps, V = 0 (no terminal charge)
    
    # V_next[q] for q in range(-Q, Q+1), indexed as 0..2Q
    V_next = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # We need to search over possible depths. The optimal depth is bounded.
    # For large depths, fill probability -> 0, so depth contribution -> 0.
    # For small depths, fill probability is high but depth earned is small.
    # Let's use a grid of candidate depths.
    
    # The fill prob is 1 - exp(-A * exp(-k*d) * dt). Let p = A * exp(-k*d) * dt.
    # For p >> 1, prob ~ 1. For p << 1, prob ~ p.
    # The expected depth earned is d * (1 - exp(-p)).
    # d * (1 - exp(-A*exp(-k*d)*dt))
    
    # Let's parameterize by x = A * exp(-k*d), so d = -ln(x/A)/k for x in (0, A].
    # Expected depth = (-ln(x/A)/k) * (1 - exp(-x*dt))
    # We want to maximize this minus the inventory cost impact.
    
    # Actually, let's just use a grid of depths. The relevant range:
    # When d is very small, exp(-k*d) ~ 1, so p ~ A*dt. 
    # When d is large, exp(-k*d) is tiny, p is tiny, expected depth ~ d*p = d*A*exp(-k*d)*dt -> 0.
    # The maximum of d*(1-exp(-A*exp(-k*d)*dt)) occurs at some moderate d.
    
    # Let's use a logarithmic grid of depths from a small value to a large value.
    # Min depth: 1e-6 (essentially 0)
    # Max depth: where A*exp(-k*d)*dt ~ 0.01, i.e., d ~ ln(100*A*dt)/k
    # For A=140, dt=0.001, k=1.5: A*dt = 0.14, so d_max ~ ln(100*0.14)/1.5 ~ ln(14)/1.5 ~ 1.96
    # For A=50, dt=0.001, k=0.75: A*dt = 0.05, d_max ~ ln(100*0.05)/0.75 ~ ln(5)/0.75 ~ 1.99
    
    # Let's use a grid from 1e-4 to about 5.0 with ~200 points.
    d_min = 1e-4
    d_max = 5.0
    n_d = 200
    depth_grid = np.linspace(d_min, d_max, n_d)
    
    # Precompute fill probabilities and expected depths for each depth
    # fill_prob[d] = 1 - exp(-A * exp(-k*d) * dt)
    # exp_depth[d] = d * fill_prob[d]
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    exp_depths = depth_grid * fill_probs
    
    # Now backward induction
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            # Return what we have so far (zero for remaining steps)
            break
        
        for q_idx in range(n_inv):
            q = q_idx - Q  # inventory value, q in [-Q, Q]
            
            # Inventory cost for this step
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # Can we place a bid? (q < Q, since no bid fill at q=Q means we can't sell)
            # Actually: "no bid fill at q = Q" means if q == Q, bid won't fill.
            # "no ask fill at q = -Q" means if q == -Q, ask won't fill.
            # So at q = Q, bid is effectively not placed (or doesn't fill).
            # At q = -Q, ask is effectively not placed.
            
            best_val = -np.inf
            best_bid = 0.0
            best_ask = 0.0
            
            if q < Q:
                # Can place bid
                for i_d in range(n_d):
                    b = depth_grid[i_d]
                    p_b = fill_probs[i_d]
                    ed_b = exp_depths[i_d]
                    
                    # After bid fill: q -> q-1, earn b
                    # After no fill: q stays
                    # But ask can also fill independently
                    
                    # We need to consider all combinations of bid/ask fills.
                    # This is expensive if we do it naively. Let's think about it.
                    
                    # Actually, for each (b, a) pair, the expected value is:
                    # - inv_cost
                    # + p_b * (b + V_next[q-1]) * (1 - p_a)  # bid fills, ask doesn't
                    # + p_a * (a + V_next[q+1]) * (1 - p_b)  # ask fills, bid doesn't
                    # + p_b * p_a * (b + a + V_next[q])       # both fill
                    # + (1-p_b)*(1-p_a) * V_next[q]           # neither fills
                    # Wait, when both fill: bid fill gives q->q-1, then ask fill gives (q-1)->q. So net q unchanged.
                    # Actually the fills happen simultaneously in this model? Let me re-read.
                    # "bid hit with probability ..., ask likewise, independently"
                    # "A fill earns its depth and moves inventory one unit"
                    # I think fills are independent and both can happen in the same step.
                    # If both fill: bid makes q->q-1, ask makes q->q+1, net q unchanged.
                    # Earned: b + a
                    
                    # Let me restructure:
                    # E[V] = -inv_cost + 
                    #   (1-p_b)(1-p_a) * V_next[q] +
                    #   p_b(1-p_a) * (b + V_next[q-1]) +
                    #   (1-p_b)p_a * (a + V_next[q+1]) +
                    #   p_b*p_a * (b + a + V_next[q])
                    
                    # This is for a fixed (b, a). We need to maximize over b and a.
                    # This is a 2D optimization which is expensive.
                    
                    # Let's separate: the cross term p_b*p_a*(b+a+V_next[q]) couples them.
                    # Let me expand:
                    # = -inv_cost + V_next[q] + 
                    #   p_b(1-p_a)*b + p_b(1-p_a)*V_next[q-1] +
                    #   (1-p_b)p_a*a + (1-p_b)p_a*V_next[q+1] +
                    #   p_b*p_a*(b+a+V_next[q]) - V_next[q]
                    # Hmm, let me just compute it directly.
                    
                    # For efficiency, let's precompute for each bid depth the contribution
                    # and for each ask depth the contribution, then combine.
                    
                    # Actually this is getting complex. Let's use a simpler approach:
                    # Grid search over (b, a) pairs. With 200x200 = 40000 pairs per (n, q),
                    # and 1000 steps * 61 inventory levels = 61000 states, that's 61000 * 40000 = 2.44e9 operations.
                    # Too slow.
                    
                    # Better approach: separate the optimization.
                    # Let's define for bid:
                    #   f_b(p_b, ed_b) = p_b * (1-p_a) * (b + V_next[q-1]) + p_b*p_a*(b+a+V_next[q])
                    # This still couples with a.
                    
                    # Alternative: iterate. For a fixed a, optimize b, then for fixed b, optimize a.
                    # Or just use a coarser grid.
                    
                    pass  # We'll handle this below with a different approach
            
            # Let's use a different approach: vectorized computation over depth grid for each q
            # For each q, compute the optimal (b, a) by considering the joint probability space.
            
            # Let's restructure to be vectorized over the depth grid.
            # For a given q, we have:
            # For bid depth b_i and ask depth a_j:
            #   val = -inv_cost + (1-pb_i)(1-pa_j)*V_next[q] + pb_i(1-pa_j)*(b_i + V_next[q-1]) + (1-pb_i)pa_j*(a_j + V_next[q+1]) + pb_i*pa_j*(b_i+a_j+V_next[q])
            
            # where pb_i = fill_probs[i], b_i = depth_grid[i], etc.
            
            # Let's compute this as a matrix of shape (n_d, n_d) for each q.
            # That's 200*200 = 40000 values per q, times 61 q's = 2.44M per step, times 1000 steps = 2.44B.
            # Still too slow in Python. Need to vectorize with NumPy.
            
            # Let's do it with NumPy broadcasting.
            if q < Q and q > -Q:
                # Both bid and ask can be placed
                # pb shape: (n_d,), pa shape: (n_d,)
                # We want val[i,j] for bid depth i, ask depth j
                pb = fill_probs  # (n_d,)
                pa = fill_probs  # (n_d,)
                b_vals = depth_grid  # (n_d,)
                a_vals = depth_grid  # (n_d,)
                
                # V_next indices
                V_q = V_next[q_idx]
                V_qm1 = V_next[q_idx - 1]  # q-1
                V_qp1 = V_next[q_idx + 1]  # q+1
                
                # Compute the value matrix
                # val[i,j] = -inv_cost + (1-pb[i])(1-pa[j])*V_q + pb[i](1-pa[j])*(b[i]+V_qm1) + (1-pb[i])pa[j]*(a[j]+V_qp1) + pb[i]pa[j]*(b[i]+a[j]+V_q)
                
                # Let's expand and separate:
                # = -inv_cost + V_q + 
                #   pb[i](1-pa[j])*(b[i]+V_qm1 - V_q) +
                #   (1-pb[i])pa[j]*(a[j]+V_qp1 - V_q) +
                #   pb[i]pa[j]*(b[i]+a[j]+V_q - V_q)
                # = -inv_cost + V_q +
                #   pb[i](1-pa[j])*(b[i]+V_qm1 - V_q) +
                #   (1-pb[i])pa[j]*(a[j]+V_qp1 - V_q) +
                #   pb[i]pa[j]*(b[i]+a[j])
                
                # Let's compute using broadcasting:
                # pb[i] is (n_d, 1), pa[j] is (1, n_d)
                pb_col = pb[:, np.newaxis]  # (n_d, 1)
                pa_row = pa[np.newaxis, :]  # (1, n_d)
                b_col = b_vals[:, np.newaxis]  # (n_d, 1)
                a_row = a_vals[np.newaxis, :]  # (1, n_d)
                
                term1 = pb_col * (1 - pa_row) * (b_col + V_qm1 - V_q)
                term2 = (1 - pb_col) * pa_row * (a_row + V_qp1 - V_q)
                term3 = pb_col * pa_row * (b_col + a_row)
                
                val_matrix = -inv_cost + V_q + term1 + term2 + term3
                
                # Find the max
                max_idx = np.unravel_index(np.argmax(val_matrix), val_matrix.shape)
                best_bid = depth_grid[max_idx[0]]
                best_ask = depth_grid[max_idx[1]]
                best_val = val_matrix[max_idx[0], max_idx[1]]
                
            elif q < Q:
                # Can only place bid (q == -Q, no ask fill)
                pb = fill_probs
                b_vals = depth_grid
                V_q = V_next[q_idx]
                V_qm1 = V_next[q_idx - 1]
                
                # val[i] = -inv_cost + (1-pb[i])*V_q + pb[i]*(b[i]+V_qm1)
                # = -inv_cost + V_q + pb[i]*(b[i]+V_qm1 - V_q)
                val_vec = -inv_cost + V_q + pb * (b_vals + V_qm1 - V_q)
                best_idx = np.argmax(val_vec)
                best_bid = depth_grid[best_idx]
                best_ask = 0.0  # No ask
                best_val = val_vec[best_idx]
                
            elif q > -Q:
                # Can only place ask (q == Q, no bid fill)
                pa = fill_probs
                a_vals = depth_grid
                V_q = V_next[q_idx]
                V_qp1 = V_next[q_idx + 1]
                
                # val[j] = -inv_cost + (1-pa[j])*V_q + pa[j]*(a[j]+V_qp1)
                # = -inv_cost + V_q + pa[j]*(a[j]+V_qp1 - V_q)
                val_vec = -inv_cost + V_q + pa * (a_vals + V_qp1 - V_q)
                best_idx = np.argmax(val_vec)
                best_ask = depth_grid[best_idx]
                best_bid = 0.0  # No bid
                best_val = val_vec[best_idx]
            else:
                # q == Q and q == -Q, impossible since Q > 0
                best_bid = 0.0
                best_ask = 0.0
                best_val = -inv_cost + V_next[q_idx]
            
            bid_table[n, q_idx] = best_bid
            ask_table[n, q_idx] = best_ask
            V_next[q_idx] = best_val
    
    _policy_cache[cache_key] = (bid_table, ask_table)
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """Quote bid/ask depths for each path based on optimal policy table."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Determine step index
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Get or compute policy table
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _policy_cache:
        if time.time() > _DEADLINE:
            # Not enough time, return default
            depth = np.full(len(inventory), 1.0 / k)
            return depth, depth.copy()
        _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
    
    bid_table, ask_table = _policy_cache[cache_key]
    
    # Map inventory to index
    # inventory in [-Q, Q], index = inventory + Q
    inv_idx = inventory + Q
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    # Ensure valid values: must be >= -3/k, finite
    min_valid = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    
    # Replace any NaN or -inf with 0
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, 0.0)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, 0.0)
    
    return bid_depths, ask_depths
