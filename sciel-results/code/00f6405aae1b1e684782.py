# EVOLVE-BLOCK-START
import numpy as np

# Global cache for pre-computed policies
_policy_cache = {}

def _compute_policy(params):
    """
    Compute the optimal policy using backward induction (Dynamic Programming).
    Returns a dictionary with 'bid' and 'ask' arrays of shape (steps, 2*Q+1).
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    num_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    # Discretize depths
    # The optimal depth is typically small. 
    # Let's use a grid that covers the likely range.
    # d = 0.01 to 5.0 seems reasonable for k=1.5, A=140.
    # If k is small, d might be larger.
    # Let's adapt the max depth based on k.
    # d_opt ~ 1/k * log(A * dt / (gamma * sigma^2 * t_rem)) ? 
    # Let's just use a sufficiently large grid.
    max_d = 10.0
    min_d = 0.01
    num_d = 50
    depths = np.linspace(min_d, max_d, num_d)
    
    # Precompute fill probabilities for all depths
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * depths)
    rates = A * exp_term * dt
    probs = 1.0 - np.exp(-rates)
    
    # V_next[q_idx] for t+1
    V_next = np.zeros(num_q)
    
    # Storage for optimal actions
    opt_bid = np.zeros((steps, num_q))
    opt_ask = np.zeros((steps, num_q))
    
    # Backward induction
    for t in range(steps - 1, -1, -1):
        t_rem = T - t * dt
        
        # V_current[q_idx]
        V_current = np.zeros(num_q)
        
        # For each inventory level q, we need to find the best (d_b, d_a)
        # We can vectorize over q.
        # For a fixed q, the value is:
        # J(d_b, d_a) = 
        #   (1-pb)(1-pa) * (V_next[q] + reward_neither) 
        #   + pb(1-pa) * (V_next[q+1] + reward_bid)
        #   + (1-pb)pa * (V_next[q-1] + reward_ask)
        #   + pb*pa * (V_next[q] + reward_both)
        # 
        # Reward:
        # r = d_b * I[bid] + d_a * I[ask] - 0.5 * gamma * sigma^2 * q^2 * dt
        # Note: The penalty is incurred regardless of fills? 
        # "sum over steps of (gamma / 2) sigma^2 q^2 dt, q the inventory before that step's fills"
        # Yes, the penalty is fixed for the step, independent of fills.
        # So we can subtract the penalty from the total J once.
        # J = E[Fill Rewards] + E[Next Value] - Penalty
        
        # Let's vectorize the computation of J for all (d_b, d_a) pairs for all q.
        # This is a 4D array: (num_d, num_d, num_q) -> might be large.
        # 50 * 50 * 61 = 152,500 floats. This is fine.
        
        # Grid of actions
        d_b_grid, d_a_grid = np.meshgrid(depths, depths, indexing='ij')
        # Shape: (num_d, num_d)
        
        p_b_grid = probs[np.clip(np.round((d_b_grid - min_d) / (max_d - min_d) * (num_d - 1)), 0, num_d - 1).astype(int)]
        p_a_grid = probs[np.clip(np.round((d_a_grid - min_d) / (max_d - min_d) * (num_d - 1)), 0, num_d - 1).astype(int)]
        
        # Actually, easier to just use the depths directly to index probs?
        # No, probs is computed for `depths`.
        # d_b_grid values are from `depths`.
        # We can just use `depths` indices.
        
        # Let's iterate over q to be safe and simple, vectorizing over actions.
        # 61 iterations * (50x50 matrix ops) is very fast.
        
        for q_idx in range(num_q):
            q = q_indices[q_idx]
            
            # Determine valid actions based on inventory limits
            # If q == Q, no bid fill -> p_b = 0
            # If q == -Q, no ask fill -> p_a = 0
            
            pb = p_b_grid.copy()
            pa = p_a_grid.copy()
            
            if q == Q:
                pb[:] = 0.0
            if q == -Q:
                pa[:] = 0.0
            
            # Next inventory states
            # q+1, q-1, q
            # Need to handle boundaries for V_next indexing
            # q_idx + 1, q_idx - 1
            
            V_q = V_next[q_idx]
            
            V_q_plus = V_next[q_idx + 1] if q_idx < num_q - 1 else 0.0
            V_q_minus = V_next[q_idx - 1] if q_idx > 0 else 0.0
            
            # If q=Q, q+1 is invalid, but p_b=0 so it doesn't matter.
            # If q=-Q, q-1 is invalid, but p_a=0 so it doesn't matter.
            
            # Rewards
            # r_bid = d_b
            # r_ask = d_a
            # r_both = d_b + d_a
            # r_neither = 0
            
            # Expected Reward from Fills
            # E[R] = pb * (1-pa) * d_b + (1-pb) * pa * d_a + pb * pa * (d_b + d_a)
            # E[R] = pb * d_b + pa * d_a
            
            # Expected Next Value
            # E[V] = (1-pb)(1-pa) V_q + pb(1-pa) V_q_plus + (1-pb)pa V_q_minus + pb pa V_q
            # E[V] = V_q * (1 - pb - pa + 2*pb*pa) + V_q_plus * pb * (1-pa) + V_q_minus * pa * (1-pb)
            
            # Total J
            # J = pb * d_b + pa * d_a + V_q * (1 - pb - pa + 2*pb*pa) + V_q_plus * pb * (1-pa) + V_q_minus * pa * (1-pb)
            #    - 0.5 * gamma * sigma^2 * q^2 * dt
            
            # We want to maximize J.
            
            term_const = V_q - 0.5 * gamma * sigma**2 * q**2 * dt
            term_pb = d_b_grid + V_q_plus - V_q  # Coeff of pb when pa=0? No.
            
            # Let's just compute the full expression element-wise.
            # J = term_const + pb * (d_b + V_q_plus - V_q + pa * (V_q - V_q_plus - V_q_minus + d_a)) 
            # This is getting complicated to factor. Just compute directly.
            
            J = (pb * d_b_grid + pa * d_a_grid) + \
                (1 - pb) * (1 - pa) * V_q + \
                pb * (1 - pa) * V_q_plus + \
                (1 - pb) * pa * V_q_minus + \
                pb * pa * V_q - \
                0.5 * gamma * sigma**2 * q**2 * dt
            
            # Find max
            max_idx = np.argmax(J)
            d_b_opt = d_b_grid.flat[max_idx]
            d_a_opt = d_a_grid.flat[max_idx]
            
            opt_bid[t, q_idx] = d_b_opt
            opt_ask[t, q_idx] = d_a_opt
            V_current[q_idx] = J.flat[max_idx]
            
        V_next = V_current
        
    return {
        'bid': opt_bid,
        'ask': opt_ask
    }

def quote(t, inventory, mid, params):
    # Check cache
    cache_key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if cache_key not in _policy_cache:
        _policy_cache[cache_key] = _compute_policy(params)
    
    policy = _policy_cache[cache_key]
    
    steps = params["steps"]
    Q = params["Q"]
    
    # t is float, but we need integer step index
    # t = n * dt, so n = t / dt
    dt = params["T"] / steps
    n = int(round(t / dt))
    # Clamp n to valid range
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    inv = np.asarray(inventory, dtype=np.int64)
    # Ensure inv is within bounds for indexing
    inv_idx = np.clip(inv + Q, 0, 2 * Q)
    
    # Fetch optimal depths
    bid_depths = policy['bid'][n, inv_idx]
    ask_depths = policy['ask'][n, inv_idx]
    
    # Apply inventory limit logic (redundant if DP handled it, but safe)
    # If q == Q, bid should be inf (no quote)
    # In our DP, if q==Q, pb=0, so d_b doesn't affect J, but we might pick a small d_b.
    # The task says "np.inf means no quote".
    # If we quote with finite depth but prob 0, it's effectively no quote.
    # However, to be precise:
    bid_depths = np.where(inv >= Q, np.inf, bid_depths)
    ask_depths = np.where(inv <= -Q, np.inf, ask_depths)
    
    # Ensure valid values
    # The task says values below -3/k make result invalid.
    # Our depths are >= 0.01.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
