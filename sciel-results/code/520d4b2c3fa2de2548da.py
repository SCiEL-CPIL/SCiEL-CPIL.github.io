import numpy as np
import time

_DEADLINE = time.time() + 290
_POLICY_CACHE = {}

def _compute_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Grid for depth search
    # Max depth: 5/k is reasonable. Let's use 10/k to be safe.
    max_depth = 10.0 / k
    grid_size = 100
    depths = np.linspace(0, max_depth, grid_size)
    
    # V_next[q_index] where q_index = q + Q
    V_next = np.zeros(2 * Q + 1)
    
    bid_table = np.full((steps, 2 * Q + 1), np.inf)
    ask_table = np.full((steps, 2 * Q + 1), np.inf)
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2 * Q + 1)
        
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            
            # Inventory penalty
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Next state values
            # Bid fill: q -> q+1. If q == Q, no fill allowed.
            if q < Q:
                V_next_q_plus = V_next[q_idx + 1]
            else:
                V_next_q_plus = -np.inf # Effectively prevents selection if it leads to invalid state? 
                # Actually, if q==Q, we set bid_depth to inf. 
                # In DP, if q==Q, we should not consider bid fills.
                # So we maximize over b such that p(b) is small? 
                # The prompt says "no bid fill at q=Q". This implies the transition probability is 0.
                # So the term p(b)(...) becomes 0.
                # We still get the reward 0? No, if no fill, we don't earn depth.
                # So Val(b) = 0 * (b + V) + 1 * V_next(q) = V_next(q).
                # This is constant w.r.t b. So any b is optimal?
                # But we want to minimize risk or something? 
                # Actually, if no fill, we earn 0. 
                # So Val(b) = V_next(q).
                # We can set b = 0 or inf. 
                # Convention: inf means no quote. 
                # If we quote, we might get a fill? No, "no bid fill". 
                # So quoting is useless. 
                # Let's set b = inf for q=Q.
                
            if q > -Q:
                V_next_q_minus = V_next[q_idx - 1]
            else:
                V_next_q_minus = -np.inf
                
            V_next_q = V_next[q_idx]
            
            # Optimize Bid
            if q < Q:
                # Maximize p(b)(b + V_next_q_plus) + (1-p(b))V_next_q
                # p(b) = 1 - exp(-A exp(-k b) dt)
                # Let's compute for all depths in grid
                p_b = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
                # Value = p_b * (depths + V_next_q_plus) + (1 - p_b) * V_next_q
                val_b = p_b * (depths + V_next_q_plus) + (1 - p_b) * V_next_q
                idx_b = np.argmax(val_b)
                bid_table[n, q_idx] = depths[idx_b]
                max_val_b = val_b[idx_b]
            else:
                bid_table[n, q_idx] = np.inf
                max_val_b = V_next_q # If no fill, value is just next state value
                
            # Optimize Ask
            if q > -Q:
                # Ask fill: q -> q-1
                # Earns ask_depth
                p_a = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
                val_a = p_a * (depths + V_next_q_minus) + (1 - p_a) * V_next_q
                idx_a = np.argmax(val_a)
                ask_table[n, q_idx] = depths[idx_a]
                max_val_a = val_a[idx_a]
            else:
                ask_table[n, q_idx] = np.inf
                max_val_a = V_next_q
                
            V_curr[q_idx] = max_val_b + max_val_a - cost
            
        V_next = V_curr
        
    return bid_table, ask_table

def quote(t, inventory, mid, params):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple heuristic
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        Q = params['Q']
        steps = params['steps']
        T = params['T']
        dt = T / steps
        remaining_time = T - t
        time_fraction = remaining_time / T
        q = inventory.astype(np.float64)
        base_depth = 1.0 / k
        beta = 0.02 * (1.0 + 0.5 * sigma) * (0.5 + 0.5 * time_fraction)
        q_scaled = q / Q
        q_adjust = np.tanh(1.5 * q_scaled)
        bid_depth = base_depth + beta * q_adjust
        ask_depth = base_depth - beta * q_adjust
        bid_depth = np.maximum(bid_depth, 0.01)
        ask_depth = np.maximum(ask_depth, 0.01)
        bid_depth[inventory >= Q] = np.inf
        ask_depth[inventory <= -Q] = np.inf
        return bid_depth, ask_depth

    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _POLICY_CACHE:
        # Check if we have time to compute
        if time.time() < _DEADLINE - 10: # 10s margin
            _POLICY_CACHE[key] = _compute_policy(params)
        else:
            # Fallback
            sigma = params['sigma']
            A = params['A']
            k = params['k']
            gamma = params['gamma']
            Q = params['Q']
            steps = params['steps']
            T = params['T']
            dt = T / steps
            remaining_time = T - t
            time_fraction = remaining_time / T
            q = inventory.astype(np.float64)
            base_depth = 1.0 / k
            beta = 0.02 * (1.0 + 0.5 * sigma) * (0.5 + 0.5 * time_fraction)
            q_scaled = q / Q
            q_adjust = np.tanh(1.5 * q_scaled)
            bid_depth = base_depth + beta * q_adjust
            ask_depth = base_depth - beta * q_adjust
            bid_depth = np.maximum(bid_depth, 0.01)
            ask_depth = np.maximum(ask_depth, 0.01)
            bid_depth[inventory >= Q] = np.inf
            ask_depth[inventory <= -Q] = np.inf
            return bid_depth, ask_depth
            
    bid_table, ask_table = _POLICY_CACHE[key]
    
    # Determine current step index
    # t = n * dt. n = t / dt.
    # t is float. 
    steps = params['steps']
    T = params['T']
    dt = T / steps
    n = int(t / dt)
    # Clamp n to [0, steps-1]
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    Q = params['Q']
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    # index = inventory + Q
    indices = inventory + Q
    
    # Get depths
    bid_depths = bid_table[n, indices]
    ask_depths = ask_table[n, indices]
    
    # Ensure shapes are correct
    # inventory is (4000,)
    # bid_depths should be (4000,)
    
    return bid_depths, ask_depths
