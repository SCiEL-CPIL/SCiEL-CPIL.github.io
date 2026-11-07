"""Optimal market making with inventory risk: Decomposed 1D HJB Optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

# Global cache for policies
_policy_cache = {}

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precomputes the optimal policy using decomposed 1D backward induction.
    Exploits the independence of bid and ask decisions to reduce complexity
    from O(N_depth^2) to O(N_depth) per state.
    """
    q_min = -Q
    q_max = Q
    n_q = q_max - q_min + 1
    
    dt = T / steps
    
    # Depth grid: Use a finer grid since the operation is now linear in N_depth
    # instead of quadratic. 1000 points is fast and high resolution.
    # Max depth: 5.0 is usually sufficient as exp(-k*5) is very small for k>0.5
    max_depth = 5.0
    n_d = 1000
    d_grid = np.linspace(0.0, max_depth, n_d)
    
    # Precompute fill probabilities for each depth
    # p[d] = 1 - exp(-A * exp(-k * d) * dt)
    exp_term = np.exp(-k * d_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    
    # V_next[q_idx] is the value at step n+1 for inventory q
    V_next = np.zeros(n_q)
    
    # Store optimal depths
    # We only need to store the depths for the current step to return in quote()
    # But quote() is called at every step, so we need to store all steps.
    # Memory: 1000 steps * 61 inventory * 2 (bid/ask) * 8 bytes = ~1 MB. Fine.
    optimal_bid = np.full((steps, n_q), np.inf)
    optimal_ask = np.full((steps, n_q), np.inf)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() > _DEADLINE - 5.0:
            # If we run out of time, we can't finish. 
            # However, the evaluator calls quote() 1000 times. 
            # If we haven't finished precomputation, quote() will fail or fallback.
            # We must ensure we finish. 
            # 1000 steps * 61 states * (1000 ops for bid + 1000 ops for ask) = 122M ops.
            # NumPy vectorization makes this very fast.
            pass
        
        V_curr = np.full(n_q, -np.inf)
        
        # Vectorized calculation for all inventory states simultaneously?
        # The transition probabilities depend on q (boundary conditions), 
        # but the core reward structure is similar.
        # Let's loop over q_idx but vectorize over d_grid.
        
        for q_idx in range(n_q):
            q = q_idx + q_min
            
            # Holding cost
            hold_cost = (gamma / 2.0) * (sigma ** 2) * (q ** 2) * dt
            
            # --- Optimize Bid Depth ---
            # Reward from bid: p_bid * d_bid
            # Next inventory:
            # If bid fills: q+1 (unless q==Q, then no fill)
            # If no bid fill: q (ask fill handled separately? No, we decomposed)
            # Wait, the decomposition relies on:
            # Total Reward = R_bid(b) + R_ask(a)
            # Total Future Value = E[V_next(q')]
            # Since q' depends on both, we must be careful.
            # q' = q + (1 if bid fills else 0) - (1 if ask fills else 0)
            # E[V_next(q')] = p_b p_a V(q) + p_b (1-p_a) V(q+1) + (1-p_b) p_a V(q-1) + (1-p_b)(1-p_a) V(q)
            # This does NOT decompose linearly into f(b) + g(a) because V_next is not linear in the probabilities.
            # HOWEVER, if we assume the value function is smooth, or if we use the *linear* approximation of the value function slope?
            # No, the current program (score 1.35) did the full 2D search.
            # The "Decomposition" strategy is only valid if the value function is linear or if we ignore the interaction term.
            # The interaction term is p_b * p_a * (V(q) - V(q+1) - V(q-1) + V(q)).
            # For small dt, this interaction term is small (order dt^2).
            # Given dt = 0.001, the error from ignoring the interaction might be acceptable and much faster.
            # Let's try the approximation:
            # E[V_next] ≈ V(q) + p_b (V(q+1) - V(q)) + p_a (V(q-1) - V(q))
            # This makes the bid and ask optimizations independent!
            # Bid Obj: max_b [ p_b * d_b + p_b * (V(q+1) - V(q)) - hold_cost_bid_part ]
            # Ask Obj: max_a [ p_a * d_a + p_a * (V(q-1) - V(q)) - hold_cost_ask_part ]
            # Where hold_cost is split or just subtracted once.
            
            # Let's implement this "Linearized Value" approximation.
            
            if q == Q:
                # No bid fill
                bid_val = 0.0
                bid_depth = 0.0 # or inf? Problem says inf means no quote.
                # If we don't quote, depth is inf.
                # But if we are at Q, we *should* not quote bid to avoid invalid state? 
                # The model says "no bid fill at q=Q". It doesn't say we can't quote.
                # But quoting a bid at Q is useless.
                # Let's set bid_depth to a large value or inf.
                # However, the prompt says "np.inf means no quote".
                # If we return inf, the evaluator might treat it as no quote.
                # Let's return 0.0 for safety? No, 0.0 is a valid quote.
                # The prompt says "Values below -3/k ... make the whole result invalid".
                # Inf is allowed.
                # Let's use inf for "no quote".
                bid_depth = np.inf
                
                # Ask optimization
                # p_a = p_fill
                # Reward: p_a * d_a
                # Value change: p_a * (V(q-1) - V(q))
                # Objective: p_a * (d_a + V(q-1) - V(q))
                val_diff_ask = V_next[q_idx - 1] - V_next[q_idx]
                obj_ask = p_fill * (d_grid + val_diff_ask)
                max_idx_ask = np.argmax(obj_ask)
                ask_depth = d_grid[max_idx_ask]
                ask_val = obj_ask[max_idx_ask]
                
                V_curr[q_idx] = ask_val - hold_cost
                optimal_bid[n, q_idx] = bid_depth
                optimal_ask[n, q_idx] = ask_depth
                
            elif q == -Q:
                # No ask fill
                ask_depth = np.inf
                
                # Bid optimization
                val_diff_bid = V_next[q_idx + 1] - V_next[q_idx]
                obj_bid = p_fill * (d_grid + val_diff_bid)
                max_idx_bid = np.argmax(obj_bid)
                bid_depth = d_grid[max_idx_bid]
                bid_val = obj_bid[max_idx_bid]
                
                V_curr[q_idx] = bid_val - hold_cost
                optimal_bid[n, q_idx] = bid_depth
                optimal_ask[n, q_idx] = ask_depth
                
            else:
                # Normal case
                # Bid
                val_diff_bid = V_next[q_idx + 1] - V_next[q_idx]
                obj_bid = p_fill * (d_grid + val_diff_bid)
                max_idx_bid = np.argmax(obj_bid)
                bid_depth = d_grid[max_idx_bid]
                bid_val = obj_bid[max_idx_bid]
                
                # Ask
                val_diff_ask = V_next[q_idx - 1] - V_next[q_idx]
                obj_ask = p_fill * (d_grid + val_diff_ask)
                max_idx_ask = np.argmax(obj_ask)
                ask_depth = d_grid[max_idx_ask]
                ask_val = obj_ask[max_idx_ask]
                
                V_curr[q_idx] = bid_val + ask_val - hold_cost
                optimal_bid[n, q_idx] = bid_depth
                optimal_ask[n, q_idx] = ask_depth
                
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
    
    dt = T / steps
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    policy = _get_policy_for_params(params)
    
    if policy is None:
        # Fallback heuristic
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
    
    idxs = inventory - q_min
    idxs = np.clip(idxs, 0, 2 * Q)
    
    bid_depths = bid_table[n, idxs]
    ask_depths = ask_table[n, idxs]
    
    # Ensure validity
    min_valid = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    
    # Replace inf with a large valid number if necessary? 
    # The prompt says "np.inf means no quote". 
    # But it also says "Values below -3/k, -inf, NaN ... make the whole result invalid".
    # So +inf is likely valid for "no quote".
    # However, some evaluators might not handle inf well in arithmetic.
    # Let's check the prompt: "np.inf means no quote".
    # So returning inf is the correct way to signal no quote.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
