# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_global_start_time = None

def _get_deadline():
    """Return the global deadline (330s wall-clock limit)."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    return _global_start_time + 330  # 330s wall-clock limit

def _compute_optimal_policy(params):
    """
    Compute optimal bid and ask depths for all (t, q) states using vectorized DP.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    lam = A * dt
    num_q = 2 * Q + 1
    
    # V[step, q_idx] - value function
    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    
    # Depth tables
    bid_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    ask_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    
    # Grid for depths
    # Optimal depth is around ln(A*dt)/k. 
    # Max depth: if k*d is large, prob is small. 
    # Let's use 0 to 6/k.
    max_depth = 6.0 / k
    num_grid = 200
    grid = np.linspace(0, max_depth, num_grid)
    
    deadline = _get_deadline()
    
    # Initialize b and a for alternating optimization
    # Start with a reasonable guess
    b_curr = np.full(num_q, 1.0 / k, dtype=np.float64)
    a_curr = np.full(num_q, 1.0 / k, dtype=np.float64)
    
    for t in range(steps - 1, -1, -1):
        # Check deadline
        if t % 10 == 0 and time.time() > deadline - 10:
            # Fill remaining with heuristic
            for tt in range(t - 1, -1, -1):
                bid_depths[tt, :] = 1.0 / k
                ask_depths[tt, :] = 1.0 / k
                V[tt, :] = 0.0
            return bid_depths, ask_depths
        
        V_next = V[t + 1]
        cost = (gamma / 2.0) * sigma**2 * dt * np.arange(num_q - 2 * Q, num_q + 1) ** 2
        # Note: q ranges from -Q to Q. q_idx = q + Q.
        # q = q_idx - Q.
        # cost = 0.5 * gamma * sigma^2 * (q_idx - Q)^2 * dt
        
        # Boundary masks
        # q = -Q -> index 0. Cannot ask (q > -Q is false). Can bid.
        # q = Q -> index 2Q. Cannot bid (q < Q is false). Can ask.
        
        can_bid = np.ones(num_q, dtype=bool)
        can_bid[-1] = False  # q = Q
        
        can_ask = np.ones(num_q, dtype=bool)
        can_ask[0] = False   # q = -Q
        
        # Alternating Optimization
        # We will iterate a few times to converge b and a
        num_iter = 10
        
        for _ in range(num_iter):
            # 1. Optimize b given a
            # For each q, we have a_curr[q].
            # P_a = 1 - exp(-lam * exp(-k * a_curr))
            P_a = 1.0 - np.exp(-lam * np.exp(-k * a_curr))
            
            # C1(q) = (1-P_a)*V_next[q+1] + P_a*(a_curr + V_next[q])
            # C2(q) = P_a*(a_curr + V_next[q-1]) + (1-P_a)*V_next[q]
            
            # V_next indices:
            # q_idx ranges 0..2Q
            # q+1 corresponds to q_idx+1. For q_idx=2Q (q=Q), q+1 is out of bounds.
            # But if q=Q, can_bid is False, so we don't care about b for that q.
            # We can handle boundaries by masking or padding.
            
            # Let's compute C1 and C2 for all q, using safe indexing.
            # For q_idx = 2Q, q+1 is 2Q+1 (out of bounds). 
            # But can_bid[2Q] is False.
            # We can just compute for all and mask later, or use np.roll/clip.
            # Simpler: create V_next_ext with padding.
            V_next_ext = np.zeros(num_q + 2)
            V_next_ext[1:-1] = V_next
            
            q_indices = np.arange(num_q)
            V_q = V_next_ext[1:num_q+1]
            V_q_plus = V_next_ext[2:num_q+2]
            V_q_minus = V_next_ext[0:num_q]
            
            C1 = (1.0 - P_a) * V_q_plus + P_a * (a_curr + V_q)
            C2 = P_a * (a_curr + V_q_minus) + (1.0 - P_a) * V_q
            
            # Now find best b for each q
            # J(b, q) = P_b(b) * (b + C1(q)) + (1-P_b(b)) * C2(q)
            # P_b_grid = 1 - exp(-lam * exp(-k * grid))
            P_b_grid = 1.0 - np.exp(-lam * np.exp(-k * grid))
            
            # Broadcasting:
            # P_b_grid: (num_grid, 1)
            # grid: (num_grid, 1)
            # C1: (1, num_q)
            # C2: (1, num_q)
            
            J_b = P_b_grid * (grid + C1) + (1.0 - P_b_grid) * C2
            
            # For q where can_bid is False, set J_b to -inf so argmax picks 0 (or any, but we mask later)
            # Actually, we just want to update b_curr for valid q.
            # We can compute argmax for all, then mask.
            
            best_b_idx = np.argmax(J_b, axis=0)
            best_b_vals = grid[best_b_idx]
            
            # Mask invalid q
            best_b_vals[~can_bid] = 0.0 # Or keep previous? 0 is safe (no bid)
            
            b_curr = best_b_vals
            
            # 2. Optimize a given b
            P_b = 1.0 - np.exp(-lam * np.exp(-k * b_curr))
            
            # C3(q) = P_b*(b_curr + V_q) + (1-P_b)*V_q_minus  (Reward if ask fills? No.)
            # Let's re-derive for a.
            # J(a) = P_a * (a + C3(q)) + (1-P_a) * C4(q)
            # From symmetry:
            # If ask fills: gain a, q -> q-1.
            # If no ask: q stays or bid fills.
            
            # J(a) = P_a * [ P_b(b+a+V_q) + (1-P_b)(b+V_q_plus) ] + (1-P_a) * [ P_b(b+V_q_plus) + (1-P_b)V_q ]
            # This is getting messy. Let's stick to the previous form but swap roles.
            
            # Term for a:
            # If ask fills (prob P_a):
            #   If bid also fills (prob P_b): gain a+b, next V_q
            #   If bid no fill (prob 1-P_b): gain a, next V_{q-1}
            # If ask no fill (prob 1-P_a):
            #   If bid fills (prob P_b): gain b, next V_{q+1}
            #   If bid no fill (prob 1-P_b): gain 0, next V_q
            
            # J(a) = P_a P_b (a+b+V_q) + P_a (1-P_b) (a+V_{q-1}) + (1-P_a) P_b (b+V_{q+1}) + (1-P_a)(1-P_b) V_q
            # = P_a [ P_b (a+b+V_q) + (1-P_b) (a+V_{q-1}) ] + (1-P_a) [ P_b (b+V_{q+1}) + (1-P_b) V_q ]
            # = P_a [ a + P_b (b+V_q) + (1-P_b) V_{q-1} ] + (1-P_a) [ P_b (b+V_{q+1}) + (1-P_b) V_q ]
            
            C3 = P_b * (b_curr + V_q) + (1.0 - P_b) * V_q_minus
            C4 = P_b * (b_curr + V_q_plus) + (1.0 - P_b) * V_q
            
            J_a = P_b_grid * (grid + C3) + (1.0 - P_b_grid) * C4
            
            best_a_idx = np.argmax(J_a, axis=0)
            best_a_vals = grid[best_a_idx]
            
            best_a_vals[~can_ask] = 0.0
            
            a_curr = best_a_vals
        
        # After convergence, compute V_t
        P_b_final = 1.0 - np.exp(-lam * np.exp(-k * b_curr))
        P_a_final = 1.0 - np.exp(-lam * np.exp(-k * a_curr))
        
        # J = P_b P_a (b+a+V_q) + P_b (1-P_a) (b+V_q_plus) + (1-P_b) P_a (a+V_q_minus) + (1-P_b)(1-P_a) V_q
        # Note: For boundary q, one of P_b or P_a is 0 (effectively, as depth is 0 or inf? 
        # We set depth to 0 for invalid. P(0) = 1-exp(0) = 0. So it works out.)
        
        V_t = (P_b_final * P_a_final * (b_curr + a_curr + V_q) + 
               P_b_final * (1.0 - P_a_final) * (b_curr + V_q_plus) + 
               (1.0 - P_b_final) * P_a_final * (a_curr + V_q_minus) + 
               (1.0 - P_b_final) * (1.0 - P_a_final) * V_q) - cost
        
        # Handle boundaries explicitly just in case
        # q=Q (idx 2Q): can_bid False. b=0. P_b=0.
        # V_t[2Q] = P_a (a + V_{2Q-1}) + (1-P_a) V_{2Q} - cost[2Q]
        # Our formula: P_b=0 -> P_b P_a ... = 0. P_b(1-P_a)... = 0.
        # Remaining: (1-0) P_a (a + V_{2Q-1}) + (1-0)(1-P_a) V_{2Q} - cost.
        # = P_a (a + V_{2Q-1}) + (1-P_a) V_{2Q} - cost. Correct.
        
        # q=-Q (idx 0): can_ask False. a=0. P_a=0.
        # V_t[0] = P_b (b + V_1) + (1-P_b) V_0 - cost[0].
        # Formula: P_a=0 -> terms with P_a are 0.
        # Remaining: P_b (1-0) (b + V_1) + (1-P_b)(1-0) V_0 - cost.
        # = P_b (b + V_1) + (1-P_b) V_0 - cost. Correct.
        
        V[t] = V_t
        bid_depths[t] = b_curr
        ask_depths[t] = a_curr
        
    return bid_depths, ask_depths

def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for the current state.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        deadline = _get_deadline()
        if time.time() < deadline - 10:
            bid_table, ask_table = _compute_optimal_policy(params)
            _policy_cache[key] = (bid_table, ask_table)
        else:
            k = params['k']
            depth = np.full(len(inventory), 1.0 / k)
            return depth, depth.copy()
    
    bid_table, ask_table = _policy_cache[key]
    
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = np.clip(step, 0, params['steps'] - 1)
    
    Q = params['Q']
    q_indices = inventory + Q
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    bid_depths = bid_table[step, q_indices]
    ask_depths = ask_table[step, q_indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
