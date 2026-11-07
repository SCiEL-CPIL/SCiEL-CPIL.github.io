"""Optimal market making with inventory risk: Separable 1D HJB Solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

_policy_cache = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precomputes the optimal policy using separable 1D backward induction.
    
    The HJB equation for this problem allows us to decompose the joint optimization
    of (bid_depth, ask_depth) into two independent 1D optimizations because the
    immediate reward and the future value terms are separable with respect to the
    bid and ask decisions, given the inventory state.
    
    Specifically, for a given state (n, q):
    V(n, q) = max_{d_b, d_a} { R(d_b, d_a, q) - C(q) + E[V(n+1, q')] }
    
    Since the next inventory q' depends on whether bid/ask fills occur, and the
    fill events are independent given the depths, the expectation can be expanded.
    However, a simpler and very effective approximation (often exact in these
    linear-quadratic-like settings or close to it) is to note that the optimal
    bid depth depends only on the value difference between q and q+1, and the
    optimal ask depth depends only on the value difference between q and q-1.
    
    Let's derive the 1D conditions.
    The objective is to maximize:
    J(d_b, d_a) = p_b(d_b)(1-p_a(d_a))d_b + (1-p_b(d_b))p_a(d_a)d_a + p_b(d_b)p_a(d_a)(d_b+d_a) 
                  - (gamma/2)sigma^2 q^2 dt 
                  + p_b(d_b)(1-p_a(d_a))V_{n+1}(q+1) + (1-p_b(d_b))p_a(d_a)V_{n+1}(q-1) 
                  + (p_b(d_b)p_a(d_a) + (1-p_b(d_b))(1-p_a(d_a)))V_{n+1}(q)
    
    This coupling makes exact separability tricky if we look at the full expectation.
    However, we can use a "greedy" or "myopic" decoupling which is standard in
    market making literature (e.g., Avellaneda-Stoikov logic applied to discrete time).
    
    Actually, a more robust approach for the discrete case:
    Notice that the term involving d_b appears in:
    1. Immediate reward: d_b * P(bid fills) = d_b * p_b * (1 - p_a + p_a) = d_b * p_b
       Wait, if bid fills, we get d_b. If both fill, we get d_b + d_a.
       So contribution of d_b is d_b * p_b.
    2. Future value: 
       If bid fills (prob p_b(1-p_a) + p_b p_a = p_b), inventory goes to q+1 (if only bid) or q (if both).
       This coupling is the issue.
       
    Let's look at the structure again.
    q' = q + 1_{bid_fill} - 1_{ask_fill}
    
    If we assume that the optimal d_b and d_a are chosen such that they don't interact
    strongly in the future value term (or we approximate the interaction), we can solve
    for d_b maximizing the "bid leg" value and d_a maximizing the "ask leg" value.
    
    A common and effective heuristic in these discrete HJB problems, which is often
    very close to optimal, is to treat the bid and ask as independent "options".
    
    Let's try a different decomposition that is exact for the *immediate* reward and
    uses a "linear approximation" for the future value impact, or simply performs
    the 2D search but with a much coarser grid for one dimension and fine for the other?
    No, the directive asks for a different strategy.
    
    Let's stick to the "Separable 1D" idea but implement it carefully.
    
    Alternative Interpretation of Separability:
    If we fix d_a, the optimal d_b can be found. But they are coupled.
    
    However, there is a known result that for this specific reward structure (linear in depth)
    and independent Poisson fills, the optimal bid and ask spreads can be found by solving
    two separate 1D problems if we use the "effective" inventory impact.
    
    Let's try a pragmatic approach:
    We will perform the backward induction. At each step (n, q), we will find the optimal
    bid depth by maximizing a 1D function that assumes the ask depth is fixed at a
    "neutral" or "optimal from previous iteration" value? No, that's iterative and slow.
    
    Let's look at the "Dual-Variable" hint in the prompt.
    "exploiting the convexity... pre-compute optimal reservation price shift and spread width"
    
    This suggests using the Avellaneda-Stoikov closed-form solution as a *starting point*
    or *approximation*, but the current program is already a discrete HJB.
    
    Let's re-read the current program's weakness: "discrete grid search... O(N_depth^2)".
    The new strategy should be "1D Newton-Raphson" or "1D Grid Search".
    
    How to make it 1D?
    We can exploit the symmetry or the structure.
    
    Actually, let's look at the objective function again.
    Maximize: 
    d_b * p_b + d_a * p_a - Cost + E[Future]
    
    If we assume that the future value V(q+1) - V(q) is the "marginal value of inventory",
    we can define:
    Delta_plus = V(n+1, q+1) - V(n+1, q)
    Delta_minus = V(n+1, q-1) - V(n+1, q)
    
    Then the expected future value is:
    V(n+1, q) + p_b(1-p_a)Delta_plus + p_a(1-p_b)Delta_minus + p_b p_a (Delta_plus + Delta_minus - Delta_plus - Delta_minus? No)
    
    If both fill: q' = q. So change is 0.
    If bid only: q' = q+1. Change is Delta_plus.
    If ask only: q' = q-1. Change is Delta_minus.
    
    So E[Future] = V(n+1, q) + p_b(1-p_a)Delta_plus + p_a(1-p_b)Delta_minus.
    
    The total value to maximize is:
    V(n+1, q) - Cost + [ d_b * p_b + Delta_plus * p_b(1-p_a) ] + [ d_a * p_a + Delta_minus * p_a(1-p_b) ]
    
    Notice that the term in the first bracket depends on d_b and d_a (via p_a).
    The term in the second bracket depends on d_a and d_b (via p_b).
    They are coupled through the cross-terms p_b p_a.
    
    However, if we approximate p_b p_a as negligible (which is true if dt is small and A is not huge, making fill probabilities small), then:
    p_b(1-p_a) ≈ p_b
    p_a(1-p_b) ≈ p_a
    
    Then the problem becomes:
    Maximize: [ d_b * p_b(d_b) + Delta_plus * p_b(d_b) ] + [ d_a * p_a(d_a) + Delta_minus * p_a(d_a) ]
    
    This is fully separable!
    Optimal d_b maximizes: p_b(d_b) * (d_b + Delta_plus)
    Optimal d_a maximizes: p_a(d_a) * (d_a + Delta_minus)
    
    This is a 1D optimization problem for each side.
    Given that dt = 0.001 and A=140, p_fill = 1-exp(-140*exp(-1.5*d)*0.001).
    For d=0, p ≈ 1-exp(-0.14) ≈ 0.13.
    For d=1, p ≈ 1-exp(-0.14*0.22) ≈ 0.03.
    The product p_b p_a is at most 0.016, which is small compared to 1.
    So the separable approximation is likely very good.
    
    We will implement this separable 1D optimization.
    """
    q_min = -Q
    q_max = Q
    n_q = q_max - q_min + 1
    
    dt = T / steps
    
    # Depth grid for 1D optimization
    # We need a fine grid to be accurate.
    # Let's use 2000 points from 0 to 8/k (which is large enough).
    max_depth = 8.0 / k
    n_d = 2000
    d_grid = np.linspace(0.0, max_depth, n_d)
    
    # Precompute p_fill for this grid
    p_fill = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
    
    # Precompute log_p_fill for soft-max
    # Where p_fill is 0, log is -inf. But p_fill > 0 for finite d.
    # For very large d, p_fill underflows to 0.
    # We use log(1 - exp(-x)) = -log(exp(x)-1) for stability? No, x = A*exp(-kd)*dt.
    # For small x, 1-exp(-x) ~ x.
    # Let's just compute log safely.
    log_p_fill = np.full(n_d, -np.inf)
    mask_p = p_fill > 0
    log_p_fill[mask_p] = np.log(p_fill[mask_p])
    
    # V_next: shape (n_q,)
    V_next = np.zeros(n_q)
    
    # Optimal depths tables
    optimal_bid = np.full((steps, n_q), 0.0)
    optimal_ask = np.full((steps, n_q), 0.0)
    
    # Precompute the 1D objective function components?
    # For a given Delta (which is V(q+1)-V(q)), the optimal d maximizes p(d) * (d + Delta).
    # Let's precompute a lookup table for optimal d given Delta?
    # Delta can vary. But we can just compute the max for each state.
    # Since n_d is 500 and n_q is 61, and steps is 1000:
    # 1000 * 61 * 2 * 500 = 61,000,000 operations. Very fast in NumPy.
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_q)
        
        # Calculate Deltas for all q at once?
        # Delta_plus[q_idx] = V_next[q_idx+1] - V_next[q_idx]
        # Delta_minus[q_idx] = V_next[q_idx-1] - V_next[q_idx]
        
        # Handle boundaries for Deltas
        # For q=Q, we don't care about Delta_plus (no bid fill).
        # For q=-Q, we don't care about Delta_minus (no ask fill).
        
        # Create arrays of Deltas
        # q_idx from 0 to n_q-1 corresponds to q from -Q to Q
        # V_next is indexed 0..n_q-1
        
        # Delta_plus for q_idx: needs V_next[q_idx+1]. Valid for q_idx < n_q-1.
        # Delta_minus for q_idx: needs V_next[q_idx-1]. Valid for q_idx > 0.
        
        # We will loop over q_idx to handle boundaries cleanly, but vectorize the d_grid part.
        
        # Vectorized computation for all q at once where possible
        # Delta_plus for q_idx 0..n_q-2
        # Delta_minus for q_idx 1..n_q-1
        
        # We need to compute optimal d and value for each q.
        # obj = p * (d + delta)
        # max over d.
        
        # Since n_q is small (61), the loop is fine, but we can vectorize the argmax part.
        # However, delta varies per q_idx.
        
        # Let's stick to the loop but optimize the inner logic.
        for q_idx in range(n_q):
            q = q_idx + q_min
            
            hold_cost = (gamma / 2.0) * (sigma ** 2) * (q ** 2) * dt
            
            # Base value
            base_val = V_next[q_idx]
            
            # Bid Optimization
            if q < Q:
                # Delta_plus = V_next[q+1] - V_next[q]
                delta_plus = V_next[q_idx + 1] - V_next[q_idx]
                
                # Objective: p(d) * (d + delta_plus)
                # We want to maximize this over d_grid.
                # Using log-sum-exp style soft max? No, we need the exact max index for the policy.
                # But we can use the log trick to find the max more stably if values are tiny?
                # Actually, argmax of f(x) is argmax of log(f(x)) if f(x)>0.
                # f(d) = p(d) * (d + delta).
                # If d + delta <= 0, f(d) <= 0.
                # If d + delta > 0, log(f(d)) = log(p(d)) + log(d + delta).
                
                # Let's compute the objective directly.
                obj_bid = p_fill * (d_grid + delta_plus)
                
                # Handle case where obj_bid might be negative or zero.
                # We want the max.
                max_idx_bid = np.argmax(obj_bid)
                optimal_bid[n, q_idx] = d_grid[max_idx_bid]
                val_bid = obj_bid[max_idx_bid]
            else:
                # No bid allowed
                optimal_bid[n, q_idx] = np.inf
                val_bid = 0.0
            
            # Ask Optimization
            if q > -Q:
                # Delta_minus = V_next[q-1] - V_next[q]
                delta_minus = V_next[q_idx - 1] - V_next[q_idx]
                
                # Objective: p(d) * (d + delta_minus)
                obj_ask = p_fill * (d_grid + delta_minus)
                
                max_idx_ask = np.argmax(obj_ask)
                optimal_ask[n, q_idx] = d_grid[max_idx_ask]
                val_ask = obj_ask[max_idx_ask]
            else:
                # No ask allowed
                optimal_ask[n, q_idx] = np.inf
                val_ask = 0.0
            
            # Total Value
            # We are ignoring the cross-term p_b p_a (d_b+d_a) and the cross-term in future value.
            # This is the separable approximation.
            V_curr[q_idx] = base_val - hold_cost + val_bid + val_ask
        
        V_next = V_curr
    
    return {
        'bid_depths': optimal_bid,
        'ask_depths': optimal_ask,
        'q_min': q_min
    }

def _get_policy_for_params(params):
    """
    Retrieves or computes the policy for the given parameters.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key in _policy_cache:
        return _policy_cache[key]
    
    # Check time budget
    if time.time() > _DEADLINE - 10.0:
        return None
    
    # Precompute
    policy = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
    _policy_cache[key] = policy
    return policy

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed HJB policy.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Get step index
    dt = T / steps
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    # Get policy
    policy = _get_policy_for_params(params)
    
    if policy is None:
        # Fallback
        dt_rem = T - t
        R = gamma * (sigma ** 2) * inventory * dt_rem
        delta_star = 1.0 / k
        
        bid_depths = R + delta_star
        ask_depths = -R + delta_star
        
        min_depth = 1e-6
        bid_depths = np.maximum(bid_depths, min_depth)
        ask_depths = np.maximum(ask_depths, min_depth)
        
        return bid_depths, ask_depths
    
    q_min = policy['q_min']
    bid_table = policy['bid_depths']
    ask_table = policy['ask_depths']
    
    # Map inventory to indices
    idxs = inventory - q_min
    idxs = np.clip(idxs, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = bid_table[n, idxs]
    ask_depths = ask_table[n, idxs]
    
    # Ensure validity
    min_valid = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    
    # Handle inf if any (shouldn't happen with our grid, but good for safety)
    # If np.inf is returned, it's valid per problem statement.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
