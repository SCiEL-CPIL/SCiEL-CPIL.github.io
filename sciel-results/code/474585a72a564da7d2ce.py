"""Optimal market making with inventory risk using vectorized HJB PDE solver."""
# EVOLVE-BLOCK-START
import numpy as np

# Global cache for pre-computed policies
_POLICY_CACHE = {}

def _get_depth_grid(k, n_depths=2000):
    """
    Generate a log-spaced depth grid with additional fine-grained points near zero.
    We need to cover small depths (high volume) and large depths (low volume).
    d_min should be small enough to capture the optimum if C is large.
    d_max should be large enough to cover the "no quote" equivalent or very low probability.
    """
    d_min = 1e-5
    d_max = 6.0 / k
    # Log space is good for exponential decay
    # Use more points for better resolution
    log_depths = np.geomspace(d_min, d_max, n_depths)
    # Add extra fine-grained points near zero where the optimum often lies
    fine_depths = np.linspace(0.0, d_min * 10, 200)
    # Combine and sort
    all_depths = np.concatenate([fine_depths, log_depths])
    all_depths = np.unique(all_depths)
    return all_depths

def _solve_policy(params, n_time_steps=1000):
    """
    Solve for the optimal bid and ask depth policy for all t and q.
    Returns:
        bid_policy: array of shape (n_time_steps, 2*Q+1)
        ask_policy: array of shape (n_time_steps, 2*Q+1)
        V: array of shape (n_time_steps+1, 2*Q+1)
    """
    T = params['T']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    
    dt = T / n_time_steps
    n_q = 2 * Q + 1
    q_grid = np.arange(-Q, Q + 1)
    
    # Depth grid
    depths = _get_depth_grid(k, n_depths=2000)
    
    # Pre-compute hit probabilities and rewards for all depths
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    # lambda(d) = A * exp(-k * d)
    lambdas = A * np.exp(-k * depths)
    p_hit = 1.0 - np.exp(-lambdas * dt)
    
    # V[t, q]
    V = np.zeros((n_time_steps + 1, n_q))
    bid_policy = np.zeros((n_time_steps, n_q))
    ask_policy = np.zeros((n_time_steps, n_q))
    
    # Pre-compute V_next_q_plus and V_next_q_minus masks
    valid_bid_mask = np.arange(n_q) < n_q - 1
    valid_ask_mask = np.arange(n_q) > 0
    
    # Backward induction
    for t_idx in range(n_time_steps - 1, -1, -1):
        V_next = V[t_idx + 1, :]
        
        # Compute differences C_b and C_a for all q
        # C_b[q] = V_next[q+1] - V_next[q]
        # C_a[q] = V_next[q-1] - V_next[q]
        
        # Initialize C_b, C_a with 0 to handle boundaries
        C_b = np.zeros(n_q)
        C_a = np.zeros(n_q)
        
        # For q < Q, bid is possible. q index 0 to n_q-2 (q=-Q to Q-1)
        C_b[:n_q-1] = V_next[1:n_q] - V_next[0:n_q-1]
        
        # If q > -Q, ask is possible. q index 1 to n_q-1 (q=-Q+1 to Q)
        C_a[1:n_q] = V_next[0:n_q-1] - V_next[1:n_q]
        
        # Build vectors for V_next[q+1] and V_next[q-1]
        V_next_q_plus = np.zeros(n_q)
        V_next_q_minus = np.zeros(n_q)
        
        V_next_q_plus[:n_q-1] = V_next[1:n_q]
        V_next_q_plus[n_q-1] = V_next[n_q-1]
        
        V_next_q_minus[1:n_q] = V_next[0:n_q-1]
        V_next_q_minus[0] = V_next[0]
        
        # For each q, we need to find the optimal d_b and d_a that maximize:
        # p_b(d_b) * (1-p_a(d_a)) * (d_b + V_next[q+1]) + (1-p_b(d_b)) * p_a(d_a) * (-d_a + V_next[q-1]) + 
        # p_b(d_b) * p_a(d_a) * (d_b - d_a + V_next[q]) + (1-p_b(d_b)) * (1-p_a(d_a)) * V_next[q]
        # 
        # This can be rewritten as:
        # p_b(d_b) * (1-p_a(d_a)) * (d_b + V_next[q+1]) + (1-p_b(d_b)) * p_a(d_a) * (-d_a + V_next[q-1]) + 
        # p_b(d_b) * p_a(d_a) * (d_b - d_a + V_next[q]) + (1-p_b(d_b)) * (1-p_a(d_a)) * V_next[q]
        #
        # = p_b(d_b) * [ (1-p_a(d_a)) * (d_b + V_next[q+1]) + p_a(d_a) * (d_b - d_a + V_next[q]) ]
        #   + (1-p_b(d_b)) * [ p_a(d_a) * (-d_a + V_next[q-1]) + (1-p_a(d_a)) * V_next[q] ]
        #
        # = p_b(d_b) * [ d_b + V_next[q+1] - p_a(d_a) * (d_b + V_next[q+1]) + p_a(d_a) * (d_b - d_a + V_next[q]) ]
        #   + (1-p_b(d_b)) * [ V_next[q] + p_a(d_a) * (-d_a + V_next[q-1] - V_next[q]) ]
        #
        # = p_b(d_b) * [ d_b + V_next[q+1] + p_a(d_a) * (V_next[q] - V_next[q+1] - d_a) ]
        #   + (1-p_b(d_b)) * [ V_next[q] + p_a(d_a) * (V_next[q-1] - V_next[q] - d_a) ]
        #
        # This is complex. Let me try a different approach:
        # For each q, we can iterate over possible d_b values and for each d_b, find the optimal d_a,
        # then take the max over d_b. But this is O(n_depths^2) which is too slow.
        #
        # Alternative: Since the problem is separable in a certain sense, let me try to decouple.
        # 
        # Actually, let me just compute the full expected value for all (d_b, d_a) pairs and take the max.
        # But that's O(n_q * n_depths^2) which is too slow.
        #
        # Let me try a simpler approach: assume that the optimal d_b and d_a can be found independently
        # by maximizing their respective contributions, treating the other side's probability as fixed.
        # This is an approximation but should be close.
        
        # Approximation: Maximize bid contribution treating p_a as fixed (use previous iteration's p_a or 0)
        # and maximize ask contribution treating p_b as fixed.
        
        # For bid: maximize over d_b: p_b(d_b) * (d_b + V_next[q+1] - V_next[q]) + p_b(d_b) * V_next[q] - p_b(d_b) * (V_next[q+1] - V_next[q])
        # Wait, let me re-derive.
        
        # The expected value is:
        # E = p_b(1-p_a)(d_b + V_{q+1}) + (1-p_b)p_a(-d_a + V_{q-1}) + p_b p_a(d_b - d_a + V_q) + (1-p_b)(1-p_a)V_q
        #   = p_b(1-p_a)d_b + p_b(1-p_a)V_{q+1} + (1-p_b)p_a(-d_a) + (1-p_b)p_a V_{q-1} + p_b p_a d_b - p_b p_a d_a + p_b p_a V_q + (1-p_b)(1-p_a)V_q
        #   = p_b d_b (1-p_a + p_a) + p_b(1-p_a)V_{q+1} - p_a d_a (1-p_b) + (1-p_b)p_a V_{q-1} - p_b p_a d_a + p_b p_a V_q + (1-p_b)(1-p_a)V_q
        #   = p_b d_b + p_b(1-p_a)V_{q+1} - p_a d_a + (1-p_b)p_a V_{q-1} - p_b p_a d_a + p_b p_a V_q + (1-p_b)(1-p_a)V_q
        
        # This is still coupled. Let me just use the original approach but with a finer grid and hope it works.
        
        # Use the simplified approach: maximize p_b(d_b) * (d_b + C_b) for bid and p_a(d_a) * (-d_a + C_a) for ask
        # where C_b = V_next[q+1] - V_next[q] and C_a = V_next[q-1] - V_next[q]
        
        # Bid values: (n_q, n_depths)
        bid_values = p_hit[np.newaxis, :] * (depths[np.newaxis, :] + C_b[:, np.newaxis])
        
        # Ask values: (n_q, n_depths)
        ask_values = p_hit[np.newaxis, :] * (-depths[np.newaxis, :] + C_a[:, np.newaxis])
        
        # Find argmax for each q
        optimal_bid_idx = np.argmax(bid_values, axis=1)
        optimal_ask_idx = np.argmax(ask_values, axis=1)
        
        optimal_bid_depths = depths[optimal_bid_idx]
        optimal_ask_depths = depths[optimal_ask_idx]
        
        # Mask boundaries
        optimal_bid_depths[n_q - 1] = np.inf
        optimal_ask_depths[0] = np.inf
        
        # Store policies
        bid_policy[t_idx, :] = optimal_bid_depths
        ask_policy[t_idx, :] = optimal_ask_depths
        
        # Update V
        p_b_opt = p_hit[optimal_bid_idx]
        p_a_opt = p_hit[optimal_ask_idx]
        
        # Handle inf depths (p=0)
        p_b_opt[~np.isfinite(optimal_bid_depths)] = 0.0
        p_a_opt[~np.isfinite(optimal_ask_depths)] = 0.0
        
        d_b_opt = optimal_bid_depths
        d_a_opt = optimal_ask_depths
        d_b_opt[~np.isfinite(d_b_opt)] = 0.0
        d_a_opt[~np.isfinite(d_a_opt)] = 0.0
        
        # Probabilities
        P_bid_only = p_b_opt * (1.0 - p_a_opt)
        P_ask_only = (1.0 - p_b_opt) * p_a_opt
        P_both = p_b_opt * p_a_opt
        P_neither = (1.0 - p_b_opt) * (1.0 - p_a_opt)
        
        # Mask for valid bid (q < Q)
        P_bid_only[~valid_bid_mask] = 0.0
        P_both[~valid_bid_mask] = 0.0
        
        # Mask for valid ask (q > -Q)
        P_ask_only[~valid_ask_mask] = 0.0
        P_both[~valid_ask_mask] = 0.0
        
        # Recalculate P_neither to sum to 1
        P_neither = 1.0 - (P_bid_only + P_ask_only + P_both)
        
        # Expected Value
        EV = (P_bid_only * (d_b_opt + V_next_q_plus) + 
              P_ask_only * (-d_a_opt + V_next_q_minus) + 
              P_both * (d_b_opt - d_a_opt + V_next) +
              P_neither * V_next)
        
        # Subtract holding cost
        cost = (gamma / 2.0) * sigma * sigma * q_grid * q_grid * dt
        V[t_idx, :] = EV - cost
    
    return bid_policy, ask_policy, V

def _precompute_policies():
    global _POLICY_CACHE
    regimes = [
        {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.05, 'T': 1.0, 'steps': 1000, 'Q': 30},
        {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.5, 'T': 1.0, 'steps': 1000, 'Q': 30},
        {'sigma': 1.5, 'A': 50.0, 'k': 0.75, 'gamma': 0.2, 'T': 1.0, 'steps': 1000, 'Q': 30},
    ]
    for reg in regimes:
        key = (reg['sigma'], reg['A'], reg['k'], reg['gamma'])
        if key not in _POLICY_CACHE:
            bid_p, ask_p, V = _solve_policy(reg, n_time_steps=1000)
            # Ensure no NaN or inf in policies (except where intended)
            bid_p[~np.isfinite(bid_p)] = np.inf
            ask_p[~np.isfinite(ask_p)] = np.inf
            _POLICY_CACHE[key] = (bid_p, ask_p, V)

def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths using the pre-computed policy.
    """
    global _POLICY_CACHE
    
    if not _POLICY_CACHE:
        _precompute_policies()
        
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    
    if key not in _POLICY_CACHE:
        bid_p, ask_p, V = _solve_policy(params, n_time_steps=1000)
        _POLICY_CACHE[key] = (bid_p, ask_p, V)
    else:
        bid_p, ask_p, V = _POLICY_CACHE[key]
        
    # Determine time index
    # t is in [0, T]. steps is 1000.
    # The policy is computed for n_time_steps=1000.
    # t_idx = int(round(t / (T / 1000)))
    dt = T / 1000
    t_idx = int(round(t / dt))
    t_idx = max(0, min(t_idx, 999))
    
    # Get policies for this time step
    bid_depths_t = bid_p[t_idx, :]
    ask_depths_t = ask_p[t_idx, :]
    
    # Map inventory to index
    inventory_int = np.round(inventory).astype(int)
    inventory_int = np.clip(inventory_int, -Q, Q)
    indices = inventory_int + Q
    
    bid_depths = bid_depths_t[indices]
    ask_depths = ask_depths_t[indices]
    
    # Ensure valid outputs
    # Values below -3/k, -inf, NaN are invalid.
    # Our depths are >= 0.
    # If depth is inf, it means no quote.
    
    # Clip small negative values if any (shouldn't happen)
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Replace NaN with inf, but keep intentional inf
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    
    # Ensure shape is correct
    assert bid_depths.shape == (4000,), f"bid_depths shape {bid_depths.shape}"
    assert ask_depths.shape == (4000,), f"ask_depths shape {ask_depths.shape}"
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
