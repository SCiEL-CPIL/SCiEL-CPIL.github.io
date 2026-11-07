"""Optimal market making with inventory risk: Vectorized HJB Backward Induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

# Global cache for policies
# Key: (sigma, A, k, gamma, T, steps, Q)
# Value: dict with 'bid_depths' and 'ask_depths' arrays of shape (steps, 2*Q+1)
_policy_cache = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precomputes the optimal policy for a given regime using HJB backward induction.
    Uses a vectorized approach over the depth grid to maximize performance.
    """
    dt = T / steps
    
    # Inventory grid: -Q to Q
    q_min = -Q
    q_max = Q
    n_q = q_max - q_min + 1
    q_values = np.arange(q_min, q_max + 1)
    
    # Depth grid for search
    # We need to cover a reasonable range. 
    # Prob ~ 1 - exp(-A exp(-k d) dt).
    # If d is very large, prob -> 0.
    # If d is small (near 0), prob ~ A*dt.
    # Optimal d is usually around 1/k.
    # Let's use a grid from a small epsilon to a max depth.
    # Max depth: beyond this, probability is negligible.
    # exp(-k d) dt A < 1e-4 => k d > ln(A dt / 1e-4)
    # For A=140, dt=0.001, A*dt = 0.14. ln(1400) ~ 7.2.
    # d_max ~ 7.2/k. If k=0.75, d_max ~ 9.6.
    # Let's use d_max = 15.0 to be safe, but fine-grained near 0.
    
    # Non-uniform grid: dense near 0, sparse far away.
    # Log-spaced from 1e-4 to 15.0
    n_depths = 200
    d_min = 1e-4
    d_max = 15.0
    depths = np.geomspace(d_min, d_max, n_depths)
    
    # Precompute probabilities for all depths
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    # Shape: (n_depths,)
    p_depths = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Value function V[t, q]
    # t: 0..steps (steps+1 values, V[steps] = 0)
    # q: 0..n_q-1
    V = np.zeros((steps + 1, n_q))
    
    # Policy tables
    bid_policy = np.zeros((steps, n_q))
    ask_policy = np.zeros((steps, n_q))
    
    # Backward induction
    for t in range(steps - 1, -1, -1):
        # Inventory risk cost for current step
        # Cost = (gamma / 2) * sigma^2 * q^2 * dt
        # q_values is the inventory BEFORE the step's fills?
        # "inventory (4000,) ints ... q the inventory before that step's fills"
        # Yes, cost is based on current q.
        risk_cost = 0.5 * gamma * (sigma ** 2) * (q_values ** 2) * dt
        
        # V_next = V[t+1, :]
        # We need to evaluate V[t+1, q+1] and V[t+1, q-1] for each q.
        # q+1 corresponds to index q+1 (if q < Q)
        # q-1 corresponds to index q-1 (if q > -Q)
        
        # V[t+1, q+1]:
        # For q < Q, q+1 is valid. Index q+1.
        # For q == Q, no bid fill, so q stays Q. But wait, the transition is:
        # If bid hit, q -> q-1 (buy).
        # If ask hit, q -> q+1 (sell).
        
        # Let's trace carefully:
        # Current inventory q.
        # Bid hit (buy): inventory decreases by 1 -> q-1.
        # Ask hit (sell): inventory increases by 1 -> q+1.
        
        # Constraints:
        # "no bid fill at q = Q" -> If q == Q, bid hit prob is 0.
        # "no ask fill at q = -Q" -> If q == -Q, ask hit prob is 0.
        
        # So:
        # If q == Q:
        #   Bid prob = 0.
        #   Ask prob = p_ask.
        #   Next states: q (no fill), q+1 (ask fill).
        #   But q+1 > Q is invalid? The problem says inventory in [-Q, Q].
        #   Usually in these models, if you are at Q, you can't sell more? 
        #   "no ask fill at q = -Q" implies symmetric: no bid fill at Q.
        #   Does it imply no ask fill at Q? 
        #   "A fill earns its depth and moves inventory one unit".
        #   If q=Q and ask hits, q becomes Q+1. Is that allowed?
        #   "inventory (4000,) ints in [-Q, Q]". This implies the state space is bounded.
        #   Usually, this means you cannot quote on the side that would violate bounds, OR the fill doesn't happen.
        #   The prompt explicitly says: "no bid fill at q = Q, no ask fill at q = -Q".
        #   It does NOT say "no ask fill at q = Q". 
        #   However, if q becomes Q+1, it's outside the defined state space for the next step's input.
        #   Let's assume the standard interpretation: You can only quote if the resulting inventory is within bounds, 
        #   OR the "no fill" constraint is specific to the side that would push you OUT of bounds?
        #   Actually, "no bid fill at q=Q" prevents q -> Q-1? No, bid fill is BUY, so q -> q-1. 
        #   Wait. 
        #   Bid is below mid. Buyer hits bid. Market maker SELLs? 
        #   No, Market Maker places Bid. If hit, MM BUYS. Inventory (long position) INCREASES?
        #   Let's check standard convention.
        #   Inventory q. 
        #   If MM buys, q increases (more long).
        #   If MM sells, q decreases.
        #   
        #   Let's re-read: "A fill earns its depth and moves inventory one unit".
        #   Usually:
        #   Bid quote: MM sells at bid? No, MM offers to buy at bid. If hit, MM buys. q -> q+1.
        #   Ask quote: MM offers to sell at ask. If hit, MM sells. q -> q-1.
        #   
        #   Let's check the risk term: (gamma / 2) sigma^2 q^2. Symmetric.
        #   Let's check the constraint: "no bid fill at q = Q".
        #   If q=Q (max long), and bid fills (MM buys), q -> Q+1. This is out of bounds.
        #   So "no bid fill at q=Q" makes sense if Bid Fill -> q+1.
        #   Similarly, "no ask fill at q = -Q".
        #   If q=-Q (max short), and ask fills (MM sells), q -> -Q-1. Out of bounds.
        #   So Ask Fill -> q-1.
        #   
        #   Therefore:
        #   Bid Fill -> q -> q+1
        #   Ask Fill -> q -> q-1
        #   
        #   Let's verify this interpretation.
        #   If I am long (q>0), I want to sell. Selling happens via Ask fill.
        #   Ask fill reduces q. This reduces my long position. Correct.
        #   If I am short (q<0), I want to buy. Buying happens via Bid fill.
        #   Bid fill increases q (towards 0). Correct.
        
        # So:
        # p_bid = probability of bid fill.
        # p_ask = probability of ask fill.
        #
        # Expected Value for state q at time t:
        # EV = -risk_cost + p_bid * (depth_bid + V[t+1, q+1]) + p_ask * (depth_ask + V[t+1, q-1]) + (1 - p_bid - p_ask) * V[t+1, q]
        #
        # Note: If q=Q, p_bid must be 0 (or we don't quote bid).
        # If q=-Q, p_ask must be 0 (or we don't quote ask).
        
        # We need to choose d_bid and d_ask to maximize EV.
        # EV = -risk_cost + V_next_q + p_bid * (d_bid + V_next_q_plus) - p_bid * V_next_q + p_ask * (d_ask + V_next_q_minus) - p_ask * V_next_q
        # EV = -risk_cost + V_next_q + p_bid * (d_bid + V_next_q_plus - V_next_q) + p_ask * (d_ask + V_next_q_minus - V_next_q)
        
        # Let Delta_plus(q) = V[t+1, q+1] - V[t+1, q]
        # Let Delta_minus(q) = V[t+1, q-1] - V[t+1, q]
        
        # For each q, we can choose d_bid and d_ask independently because p_bid and p_ask are independent 
        # and the terms separate?
        # EV = const + p_bid(d_bid) * (d_bid + Delta_plus) + p_ask(d_ask) * (d_ask + Delta_minus)
        # Yes, separable!
        
        # We need to maximize f(d) = p(d) * (d + Delta) over d in depth_grid.
        # Where p(d) = 1 - exp(-A exp(-k d) dt).
        
        # Precompute p_depths again? It's the same for all q.
        # p_depths shape (n_depths,)
        
        # For each q, we have Delta_plus and Delta_minus.
        # We want to find d_bid* that maximizes p_depths * (depths + Delta_plus[q])
        # And d_ask* that maximizes p_depths * (depths + Delta_minus[q])
        
        # Vectorized approach:
        # Create a matrix M of shape (n_q, n_depths)
        # M[q, j] = p_depths[j] * (depths[j] + Delta[q])
        # Find argmax along axis 1.
        
        # Calculate Deltas
        # V_next = V[t+1, :]
        V_next = V[t + 1, :]
        
        # Delta_plus[q] = V_next[q+1] - V_next[q]
        # For q < Q, q+1 is valid index.
        # For q == Q, we don't quote bid, so d_bid doesn't matter (p=0). 
        # But we must handle indices carefully.
        
        # Indices:
        # q_idx 0..n_q-1 corresponds to q_min..q_max.
        # q_val = q_min + q_idx.
        
        # Delta_plus for q_idx:
        # If q_val < Q: V_next[q_idx+1] - V_next[q_idx]
        # If q_val == Q: Undefined/Not used.
        
        # Delta_minus for q_idx:
        # If q_val > -Q: V_next[q_idx-1] - V_next[q_idx]
        # If q_val == -Q: Undefined/Not used.
        
        # Let's compute arrays of shape (n_q,)
        Delta_plus = np.zeros(n_q)
        Delta_minus = np.zeros(n_q)
        
        # q_idx 0..n_q-2 correspond to q < Q (since q_max is at n_q-1)
        # So for q_idx in 0..n_q-2:
        Delta_plus[:n_q-1] = V_next[1:n_q] - V_next[0:n_q-1]
        # For q_idx n_q-1 (q=Q), Delta_plus is not used.
        
        # q_idx 1..n_q-1 correspond to q > -Q (since q_min is at 0)
        # So for q_idx in 1..n_q-1:
        Delta_minus[1:n_q] = V_next[0:n_q-1] - V_next[1:n_q]
        # For q_idx 0 (q=-Q), Delta_minus is not used.
        
        # Now, construct the objective matrices.
        # Obj_bid[q, j] = p_depths[j] * (depths[j] + Delta_plus[q])
        # Obj_ask[q, j] = p_depths[j] * (depths[j] + Delta_minus[q])
        
        # To save memory/time, we can compute argmax for each q separately? 
        # Or vectorize fully.
        # n_q = 61 (Q=30). n_depths = 200.
        # Matrix size 61x200 = 12200 floats. Very small.
        
        # Obj_bid = np.outer(Delta_plus, np.ones(n_depths)) + depths # (n_q, n_depths)
        # Actually: p_depths[j] * (depths[j] + Delta_plus[q])
        # = p_depths * depths + p_depths * Delta_plus[q]
        # This is not separable in a simple outer product way for argmax?
        # Let's just compute the full matrix.
        
        # Term 1: p_depths * (depths + Delta_plus[:, None])
        # Shape: (n_depths,) * (n_q, n_depths) -> (n_q, n_depths)
        
        obj_bid = p_depths[None, :] * (depths[None, :] + Delta_plus[:, None])
        obj_ask = p_depths[None, :] * (depths[None, :] + Delta_minus[:, None])
        
        # For q=Q (last index), we should not quote bid. 
        # How to enforce? 
        # If we set the objective to -inf for all depths, argmax will pick 0 (first index).
        # Then we can check if q==Q and set to inf (no quote).
        # Or, we can just set the chosen depth to inf later.
        
        # Similarly for q=-Q (first index) and ask.
        
        # Find optimal depth indices
        best_bid_idx = np.argmax(obj_bid, axis=1)
        best_ask_idx = np.argmax(obj_ask, axis=1)
        
        # Get optimal depths
        opt_bid_depths = depths[best_bid_idx]
        opt_ask_depths = depths[best_ask_idx]
        
        # Enforce boundary conditions:
        # If q == Q (q_idx == n_q-1), bid_depth -> inf (no quote)
        # If q == -Q (q_idx == 0), ask_depth -> inf (no quote)
        
        opt_bid_depths[n_q-1] = np.inf
        opt_ask_depths[0] = np.inf
        
        # Store in policy tables
        bid_policy[t, :] = opt_bid_depths
        ask_policy[t, :] = opt_ask_depths
        
        # Update V[t, :]
        # V[t, q] = -risk_cost[q] + V_next[q] + max_obj_bid[q] + max_obj_ask[q]
        # Wait, the max_obj includes the p*(d+Delta) term.
        # EV = -risk_cost + V_next + p_bid*(d_bid + V_next[q+1] - V_next[q]) + p_ask*(d_ask + V_next[q-1] - V_next[q])
        # max_obj_bid[q] is the max over d_bid of p_bid*(d_bid + Delta_plus[q]).
        # max_obj_ask[q] is the max over d_ask of p_ask*(d_ask + Delta_minus[q]).
        
        # So V[t, q] = -risk_cost[q] + V_next[q] + obj_bid[q, best_bid_idx[q]] + obj_ask[q, best_ask_idx[q]]
        
        # Note: For boundary cases, we set depth to inf, which means p=0?
        # If depth is inf, p(inf) = 0.
        # In our grid, max depth is 15. p(15) is small but not 0.
        # We forced opt_bid_depths[n_q-1] = inf.
        # But obj_bid[n_q-1, best_idx] was calculated using the grid depth, not inf.
        # We need to ensure that for boundary states, the contribution of the forbidden side is 0.
        
        # For q=Q (n_q-1):
        # We forced bid_depth = inf. This implies p_bid = 0.
        # So the term p_bid*(...) should be 0.
        # But obj_bid[n_q-1, best_idx] is non-zero (calculated with finite depth).
        # We must zero out the obj for the forbidden side.
        
        # For q=Q, bid is forbidden. Set obj_bid[n_q-1, :] = 0? 
        # No, we already picked best_idx. 
        # Let's just manually set the contribution to 0 for the forbidden side.
        
        max_bid_contrib = obj_bid[np.arange(n_q), best_bid_idx]
        max_ask_contrib = obj_ask[np.arange(n_q), best_ask_idx]
        
        # Zero out forbidden contributions
        max_bid_contrib[n_q-1] = 0.0
        max_ask_contrib[0] = 0.0
        
        V[t, :] = -risk_cost + V_next + max_bid_contrib + max_ask_contrib
        
    # Convert policies to lookup tables indexed by inventory value
    # Inventory range [-Q, Q]. Index = q - q_min.
    # We need arrays of shape (steps, 2*Q+1)
    # bid_policy is already (steps, n_q) where n_q = 2*Q+1.
    # So we can return directly if we map q to index.
    
    return {
        'bid_depths': bid_policy,
        'ask_depths': ask_policy,
        'q_min': q_min
    }

def _get_policy_for_params(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        if time.time() > _DEADLINE - 5.0:
            return None
        _policy_cache[key] = _precompute_regime(
            params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q']
        )
    return _policy_cache[key]

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed HJB policy.
    """
    # Get or compute policy
    policy = _get_policy_for_params(params)
    
    if policy is None:
        # Fallback to simple heuristic if precomputation failed or timed out
        sigma = params["sigma"]
        A = params["A"]
        k = params["k"]
        gamma = params["gamma"]
        T = params["T"]
        
        dt_rem = T - t
        R = gamma * (sigma ** 2) * inventory * dt_rem
        delta_star = 1.0 / k
        
        bid_depths = R + delta_star
        ask_depths = -R + delta_star
        
        min_depth = 1e-6
        bid_depths = np.maximum(bid_depths, min_depth)
        ask_depths = np.maximum(ask_depths, min_depth)
        
        return bid_depths, ask_depths

    Q = params['Q']
    steps = params['steps']
    q_min = policy['q_min']
    
    # Determine current step index
    # t = n * dt. dt = T / steps.
    # n = t / dt = t * steps / T
    # Since t is passed as n * dt, we can recover n.
    # However, floating point might be an issue.
    # t is given. dt = params['T'] / params['steps'].
    # n = round(t / dt)
    
    dt = params['T'] / steps
    n = int(round(t / dt))
    
    # Clamp n to valid range [0, steps-1]
    if n < 0:
        n = 0
    if n >= steps:
        n = steps - 1
        
    # Inventory is an array of ints.
    # Map inventory values to indices.
    # idx = inventory - q_min
    # q_min is -Q. So idx = inventory + Q.
    # inventory is in [-Q, Q], so idx is in [0, 2Q].
    
    idx = inventory - q_min
    
    # Fetch depths from policy tables
    # policy['bid_depths'] has shape (steps, 2*Q+1)
    # We need row n.
    
    bid_depths = policy['bid_depths'][n, idx]
    ask_depths = policy['ask_depths'][n, idx]
    
    # Ensure shapes are correct (4000,)
    # inventory is (4000,), so idx is (4000,), so bid_depths is (4000,).
    
    # Check for NaNs or invalid values
    # If any depth is negative (shouldn't happen with grid), clip to min.
    # np.inf is allowed.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
