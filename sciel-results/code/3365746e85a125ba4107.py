# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0
_POLICY_CACHE = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal policy via dynamic programming.
    Returns a dictionary mapping (step, inventory) to (bid_depth, ask_depth).
    """
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventories from -Q to Q
    
    # V will be (n_steps+1, n_inv)
    # V[step, inv_idx] = optimal value from step to end, given inventory
    V = np.zeros((n_steps + 1, n_inv))
    
    # Precompute fill probabilities for a range of depths
    # We need to find optimal depth for bid and ask given the value difference
    # For bid: max over d >= 0 of p(d) * (d + V(q+1) - V(q)) - 0 (no cost for depth, just earned)
    # Actually, the objective is sum of (depths earned) - sum (gamma/2) sigma^2 q^2 dt
    # So at each step, we choose bid_depth b and ask_depth a.
    # If bid fills, we earn b and inventory increases by 1.
    # If ask fills, we earn a and inventory decreases by 1.
    # They are independent.
    
    # Bellman equation:
    # V(s, q) = max_{b, a >= 0} [ p_b * b + p_a * a - (gamma/2) sigma^2 q^2 dt
    #             + E_mid[ 0.5*(1-p_b)*(1-p_a)*V(s+1, q) 
    #                      + 0.5*p_b*(1-p_a)*V(s+1, q+1)  # wait, mid moves, but V is defined per step
    #                      ... ]
    
    # Wait, the mid moves by sigma sqrt(dt) N(0,1). The value function V(s, q) does not depend on mid explicitly?
    # The problem says: "Objective per path: depths earned minus sum over steps of ... q the inventory before that step's fills"
    # The mid movement does not directly enter the objective, except through the fact that the next step's mid is different.
    # But the quote function is called with mid, and depths are relative to mid.
    # The fill probabilities depend on depths, not mid.
    # The mid movement is independent of the fills and inventory.
    # So V(s, q) should be independent of mid. The mid just shifts the quote levels, but depths are what matter.
    
    # Let's re-verify.
    # Bid fill prob: 1 - exp(-A exp(-k bid_depth) dt)
    # Ask fill prob: 1 - exp(-A exp(-k ask_depth) dt)
    # These depend only on depths.
    # The objective is sum of depths earned - sum of inventory costs.
    # The mid movement does not affect the objective directly.
    # So V(s, q) is indeed independent of mid.
    
    # However, the quote function returns depths. The evaluator simulates the fills and mid movement.
    # Since V(s, q) is independent of mid, the policy (bid_depth, ask_depth) depends only on (s, q).
    
    # Bellman:
    # V(s, q) = - (gamma/2) sigma^2 q^2 dt + max_{b, a} [ p_b(b) * b + p_a(a) * a
    #             + (1-p_b(b))(1-p_a(a)) V(s+1, q)
    #             + p_b(b)(1-p_a(a)) V(s+1, q+1)  # if q < Q
    #             + (1-p_b(b))p_a(a) V(s+1, q-1)  # if q > -Q
    #             + p_b(b)p_a(a) V(s+1, q) ]  # wait, if both fill, inventory changes by +1 -1 = 0?
    
    # "A fill earns its depth and moves inventory one unit"
    # Bid fill: inventory increases by 1 (buy)
    # Ask fill: inventory decreases by 1 (sell)
    # If both fill: inventory increases by 1 and decreases by 1, so net change 0.
    # But do we earn both depths? Yes.
    
    # So:
    # V(s, q) = - (gamma/2) sigma^2 q^2 dt + max_{b, a} [ p_b b + p_a a
    #             + (1-p_b)(1-p_a) V(s+1, q)
    #             + p_b(1-p_a) V(s+1, q+1)  [if q < Q]
    #             + (1-p_b)p_a V(s+1, q-1)  [if q > -Q]
    #             + p_b p_a V(s+1, q) ]
    
    # Note: if q = Q, no bid fill. So p_b = 0 effectively, or we set b = inf (no quote).
    # If q = -Q, no ask fill. So p_a = 0 effectively.
    
    # This is a 2D maximization over b and a. But it's separable?
    # Let's expand:
    # p_b b + p_a a + V(s+1, q) + p_b(1-p_a)(V(s+1, q+1) - V(s+1, q)) + (1-p_b)p_a(V(s+1, q-1) - V(s+1, q))
    # = V(s+1, q) + p_b [ b + (1-p_a)(V(s+1, q+1) - V(s+1, q)) ] + p_a [ a + (1-p_b)(V(s+1, q-1) - V(s+1, q)) ]
    
    # This is not separable because p_a appears in the b term and vice versa.
    # However, for small dt, p_b and p_a are small.
    # To first order in dt, p_b ~ A exp(-k b) dt, p_a ~ A exp(-k a) dt.
    # Then the cross terms p_b p_a are O(dt^2) and can be neglected for small dt.
    # But dt = 0.001, so dt^2 = 1e-6. This might be acceptable.
    
    # Let's assume separability for now:
    # V(s, q) ≈ - (gamma/2) sigma^2 q^2 dt + V(s+1, q)
    #             + max_b [ p_b(b) ( b + V(s+1, q+1) - V(s+1, q) ) ]  [if q < Q]
    #             + max_a [ p_a(a) ( a + V(s+1, q-1) - V(s+1, q) ) ]  [if q > -Q]
    
    # This is a good approximation. Let's use it.
    
    # Precompute optimal depth for a given C = V(s+1, q+1) - V(s+1, q) (for bid)
    # or C = V(s+1, q-1) - V(s+1, q) (for ask)
    # We want to maximize f(x) = (1 - exp(-alpha exp(-k x))) (x + C)
    # where alpha = A dt.
    
    # Let's precompute a table of optimal depths for a range of C values.
    # C can be negative or positive.
    # If C is very negative, we want to avoid fills, so x should be large (p small).
    # If C is very positive, we want to maximize fills, so x should be small (p large).
    
    # Let's compute the optimal depth for a grid of C values.
    C_min = -10.0
    C_max = 10.0
    C_grid = np.linspace(C_min, C_max, 100)
    optimal_depths = np.zeros_like(C_grid)
    
    for i, C in enumerate(C_grid):
        alpha = A * dt
        if alpha <= 0:
            optimal_depths[i] = 0.0
            continue
        
        # Maximize f(x) = (1 - exp(-alpha exp(-k x))) (x + C)
        # f'(x) = alpha k exp(-k x) exp(-alpha exp(-k x)) (x + C) + (1 - exp(-alpha exp(-k x)))
        # Set to 0.
        # This is hard to solve analytically. Use numerical optimization.
        
        def neg_obj(x):
            if x < 0:
                return 1e10
            p = 1.0 - np.exp(-alpha * np.exp(-k * x))
            return -p * (x + C)
        
        # Use a simple grid search followed by refinement
        x_grid = np.linspace(0, 10.0/k, 100)
        vals = [neg_obj(x) for x in x_grid]
        best_x = x_grid[np.argmin(vals)]
        
        # Refine around best_x
        x_refine = np.linspace(max(0, best_x - 0.1/k), best_x + 0.1/k, 100)
        vals_refine = [neg_obj(x) for x in x_refine]
        optimal_depths[i] = x_refine[np.argmin(vals_refine)]
    
    # Now compute V backwards
    policy = {}
    
    for s in range(n_steps - 1, -1, -1):
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            V_s_q = -0.5 * gamma * sigma**2 * q**2 * dt
            
            bid_depth = np.inf
            ask_depth = np.inf
            
            if q < Q:
                C_bid = V[s + 1, q + 1 + Q] - V[s + 1, q + Q]
                # Interpolate optimal depth
                if C_bid < C_min:
                    C_bid = C_min
                elif C_bid > C_max:
                    C_bid = C_max
                idx = (C_bid - C_min) / (C_max - C_min) * (len(C_grid) - 1)
                idx0 = int(np.floor(idx))
                idx1 = min(idx0 + 1, len(C_grid) - 1)
                frac = idx - idx0
                bid_depth = (1 - frac) * optimal_depths[idx0] + frac * optimal_depths[idx1]
                
                alpha = A * dt
                p_b = 1.0 - np.exp(-alpha * np.exp(-k * bid_depth))
                V_s_q += p_b * (bid_depth + V[s + 1, q + 1 + Q] - V[s + 1, q + Q])
            
            if q > -Q:
                C_ask = V[s + 1, q - 1 + Q] - V[s + 1, q + Q]
                if C_ask < C_min:
                    C_ask = C_min
                elif C_ask > C_max:
                    C_ask = C_max
                idx = (C_ask - C_min) / (C_max - C_min) * (len(C_grid) - 1)
                idx0 = int(np.floor(idx))
                idx1 = min(idx0 + 1, len(C_grid) - 1)
                frac = idx - idx0
                ask_depth = (1 - frac) * optimal_depths[idx0] + frac * optimal_depths[idx1]
                
                alpha = A * dt
                p_a = 1.0 - np.exp(-alpha * np.exp(-k * ask_depth))
                V_s_q += p_a * (ask_depth + V[s + 1, q - 1 + Q] - V[s + 1, q + Q])
            
            V[s, q_idx] = V_s_q
            policy[(s, q)] = (bid_depth, ask_depth)
    
    return policy

def _get_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        if time.time() > _DEADLINE - 10:
            return None
        _POLICY_CACHE[key] = _compute_policy(*key)
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE - 5:
        # Fallback to seed policy
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    policy = _get_policy(params)
    if policy is None:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    n = int(round(t / (params['T'] / params['steps'])))
    n = min(n, params['steps'] - 1)
    
    bid_depths = np.zeros(len(inventory))
    ask_depths = np.zeros(len(inventory))
    
    for i, q in enumerate(inventory):
        bd, ad = policy.get((n, q), (np.inf, np.inf))
        bid_depths[i] = bd
        ask_depths[i] = ad
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
