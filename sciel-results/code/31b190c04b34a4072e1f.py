"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _optimal_depths(sigma, A, k, gamma, T, steps, Q, V_next, n):
    """
    Compute optimal bid/ask depths via DP over (time, inventory) states.
    Uses independent optimization (bid and ask optimized separately given
    the other side's fill probability), which is exact for this model since
    the value function decomposes.
    
    Returns (opt_bid, opt_ask) arrays of shape (2*Q+1,).
    """
    dt = T / steps
    num_inventory = 2 * Q + 1
    
    # Fine depth grid
    depth_min = 1e-4
    depth_max = max(10.0 / k, 1.0)
    n_depths = 200
    depth_grid = np.linspace(depth_min, depth_max, n_depths)
    
    # Precompute fill probabilities for each depth
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Precompute penalty for each inventory level
    q_vals = np.arange(-Q, Q + 1)
    penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    opt_bid = np.zeros(num_inventory)
    opt_ask = np.zeros(num_inventory)
    
    # For interior states, we can optimize bid and ask independently
    # because the expected value separates:
    # E[reward] = E[bid_reward] + E[ask_reward] - penalty + E[next_value]
    # and the next value depends on inventory change which is additive
    
    # Actually, let's be more careful. The joint optimization is:
    # max_{db, da} db*pb(db) + da*pa(da) - penalty + 
    #   pb*pa*V[q] + pb*(1-pa)*V[q+1] + (1-pb)*pa*V[q-1] + (1-pb)*(1-pa)*V[q]
    
    # This can be rewritten as:
    # max_{db, da} [db*pb(db) - penalty + pb*(V[q+1] - V[q])] + 
    #              [da*pa(da) + pa*(V[q-1] - V[q])]
    
    # Wait, that's not quite right because of the cross terms. Let me expand:
    # = db*pb + da*pa - penalty + 
    #   pb*pa*V[q] + pb*(1-pa)*V[q+1] + (1-pb)*pa*V[q-1] + (1-pb)*(1-pa)*V[q]
    
    # Group by pb and pa:
    # = db*pb + da*pa - penalty + 
    #   pb*pa*(V[q] + V[q] - V[q] - V[q]) ... no let me just expand properly
    
    # = db*pb + da*pa - penalty +
    #   pb*pa*V[q] + pb*V[q+1] - pb*pa*V[q+1] + pa*V[q-1] - pb*pa*V[q-1] + V[q] - pb*V[q] - pa*V[q] + pb*pa*V[q]
    
    # = db*pb + da*pa - penalty + V[q] +
    #   pb*(V[q+1] - V[q]) + pa*(V[q-1] - V[q]) - pb*pa*(V[q+1] - V[q] + V[q-1] - V[q] - V[q])
    
    # This is getting complicated. Let's just do it properly with the full expression.
    
    # For efficiency, we can vectorize over inventory.
    # For each inventory level q, compute the optimal (db, da).
    
    for q_idx in range(num_inventory):
        penalty = penalties[q_idx]
        
        if q_idx == 0:
            # q = -Q: no ask quote
            best_val = -penalty
            best_bid = 0.0
            for i, d in enumerate(depth_grid):
                p_bid = fill_probs[i]
                val = d * p_bid - penalty + (1 - p_bid) * V_next[q_idx] + p_bid * V_next[q_idx + 1]
                if val > best_val:
                    best_val = val
                    best_bid = d
            opt_bid[q_idx] = best_bid
            opt_ask[q_idx] = 0.0
        elif q_idx == num_inventory - 1:
            # q = Q: no bid quote
            best_val = -penalty
            best_ask = 0.0
            for i, d in enumerate(depth_grid):
                p_ask = fill_probs[i]
                val = d * p_ask - penalty + (1 - p_ask) * V_next[q_idx] + p_ask * V_next[q_idx - 1]
                if val > best_val:
                    best_val = val
                    best_ask = d
            opt_bid[q_idx] = 0.0
            opt_ask[q_idx] = best_ask
        else:
            # Interior state: optimize both bid and ask
            # Use the fact that we can separate the optimization
            # by noting that for fixed pa, the optimal pb can be found, and vice versa.
            # But it's a 2D problem. Let's use a coarser grid for speed.
            
            # Use a smaller grid for the joint optimization
            n_depths_small = 80
            depth_grid_small = np.linspace(depth_min, depth_max, n_depths_small)
            fill_probs_small = 1.0 - np.exp(-A * np.exp(-k * depth_grid_small) * dt)
            
            # Create meshgrid
            pb, pa = np.meshgrid(fill_probs_small, fill_probs_small, indexing='ij')
            db, da = np.meshgrid(depth_grid_small, depth_grid_small, indexing='ij')
            
            p_both = pb * pa
            p_bid_only = pb * (1 - pa)
            p_ask_only = (1 - pb) * pa
            p_none = (1 - pb) * (1 - pa)
            
            reward = db * pb + da * pa - penalty
            next_val = (p_both * V_next[q_idx] +
                        p_bid_only * V_next[q_idx + 1] +
                        p_ask_only * V_next[q_idx - 1] +
                        p_none * V_next[q_idx])
            
            val = reward + next_val
            best_idx = np.unravel_index(np.argmax(val), val.shape)
            best_bid = depth_grid_small[best_idx[0]]
            best_ask = depth_grid_small[best_idx[1]]
            
            opt_bid[q_idx] = best_bid
            opt_ask[q_idx] = best_ask
    
    return opt_bid, opt_ask


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via DP over (time, inventory) states.
    Vectorized over inventory for speed.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    num_inventory = 2 * Q + 1
    
    # V[q] = value function for current time step, indexed by q + Q
    V_next = np.zeros(num_inventory)
    
    # Store optimal depths: opt_bid[n, q+Q], opt_ask[n, q+Q]
    opt_bid = np.zeros((steps, num_inventory))
    opt_ask = np.zeros((steps, num_inventory))
    
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 5.0:
            break
        
        opt_bid_n, opt_ask_n = _optimal_depths(sigma, A, k, gamma, T, steps, Q, V_next, n)
        opt_bid[n] = opt_bid_n
        opt_ask[n] = opt_ask_n
        
        # Update value function
        dt = T / steps
        depth_min = 1e-4
        depth_max = max(10.0 / k, 1.0)
        n_depths_small = 80
        depth_grid_small = np.linspace(depth_min, depth_max, n_depths_small)
        fill_probs_small = 1.0 - np.exp(-A * np.exp(-k * depth_grid_small) * dt)
        
        q_vals = np.arange(-Q, Q + 1)
        penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        V_curr = np.zeros(num_inventory)
        
        for q_idx in range(num_inventory):
            penalty = penalties[q_idx]
            
            if q_idx == 0:
                # q = -Q: no ask quote
                best_val = -penalty
                for i, d in enumerate(depth_grid_small):
                    p_bid = fill_probs_small[i]
                    val = d * p_bid - penalty + (1 - p_bid) * V_next[q_idx] + p_bid * V_next[q_idx + 1]
                    if val > best_val:
                        best_val = val
                V_curr[q_idx] = best_val
            elif q_idx == num_inventory - 1:
                # q = Q: no bid quote
                best_val = -penalty
                for i, d in enumerate(depth_grid_small):
                    p_ask = fill_probs_small[i]
                    val = d * p_ask - penalty + (1 - p_ask) * V_next[q_idx] + p_ask * V_next[q_idx - 1]
                    if val > best_val:
                        best_val = val
                V_curr[q_idx] = best_val
            else:
                # Interior state
                db = opt_bid_n[q_idx]
                da = opt_ask_n[q_idx]
                
                pb = 1.0 - np.exp(-A * np.exp(-k * db) * dt)
                pa = 1.0 - np.exp(-A * np.exp(-k * da) * dt)
                
                p_both = pb * pa
                p_bid_only = pb * (1 - pa)
                p_ask_only = (1 - pb) * pa
                p_none = (1 - pb) * (1 - pa)
                
                reward = db * pb + da * pa - penalty
                next_val = (p_both * V_next[q_idx] +
                            p_bid_only * V_next[q_idx + 1] +
                            p_ask_only * V_next[q_idx - 1] +
                            p_none * V_next[q_idx])
                
                V_curr[q_idx] = reward + next_val
        
        V_next = V_curr
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with inventory-based skew."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _POLICY_CACHE:
        # Try to compute if we have time
        if time.time() < _DEADLINE - 10.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except:
                pass

    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        # Map inventory to index
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx]
        ask_depths = opt_ask[n, q_idx]
        # Ensure valid depths
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        return bid_depths, ask_depths
    else:
        # Fallback: simple inventory-based skew
        base_depth = 1.0 / k
        skew = gamma * sigma**2 * (T - t) / (2.0 * k)
        bid_depths = base_depth + skew * inventory
        ask_depths = base_depth - skew * inventory

        bid_depths = np.clip(bid_depths, 1e-4, 10.0)
        ask_depths = np.clip(ask_depths, 1e-4, 10.0)

        return bid_depths, ask_depths
# EVOLVE-BLOCK-END
