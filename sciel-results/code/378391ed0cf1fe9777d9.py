"""Optimal market making with inventory risk: Fast 1D Optimized HJB Solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin

# Precomputation storage
_REGIME_CACHE = {}

def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solves the discrete HJB by decoupling the action space into two 1D optimizations.
    Returns policy (steps, 2*Q+1, 2) with optimal [bid_depth, ask_depth].
    """
    dt = T / steps
    num_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Precompute fill probabilities for a wide range of depths? 
    # No, we need to evaluate P(d) on the fly during optimization.
    # P(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    # V_next shape (num_q,)
    V_next = np.zeros(num_q)
    
    # Policy storage
    policy = np.zeros((steps, num_q, 2))
    
    # We iterate backwards from n = steps-1 down to 0
    for n in range(steps - 1, -1, -1):
        # Risk cost is independent of action, so we can add it at the end or ignore it for optimization
        # (it shifts the value but not the argmax)
        
        # We need to optimize bid_depth and ask_depth independently for each q.
        # Let f_bid(d) = E[V_next | bid_depth=d, ask_depth=opt_ask]
        # Actually, since bid and ask are independent in the fill process and the value function is linear in the immediate reward,
        # the optimization separates.
        
        # Let's define the objective for bid depth d_b, given optimal ask depth d_a*:
        # J_b(d_b) = P_b(d_b) * (d_b + V_next[q+1]) + (1 - P_b(d_b)) * (V_next[q] + P_a(d_a*) * (d_a* + V_next[q-1] - V_next[q]))
        # Wait, the ask part is constant with respect to d_b.
        # So we just maximize: P_b(d_b) * (d_b + V_next[q+1] - V_next[q]) + (1-P_b(d_b)) * V_next[q]
        # Let Delta_plus = V_next[q+1] - V_next[q]
        # Maximize: P_b(d_b) * (d_b + Delta_plus)
        
        # Similarly for ask depth d_a:
        # Let Delta_minus = V_next[q-1] - V_next[q]
        # Maximize: P_a(d_a) * (d_a + Delta_minus)
        
        # Note: At boundaries q=Q, bid is disabled (P_b=0). At q=-Q, ask is disabled.
        
        # We can vectorize this over q.
        # For each q, we need to find d_b that maximizes g_b(d_b) = P_b(d_b) * (d_b + Delta_plus[q])
        # And d_a that maximizes g_a(d_a) = P_a(d_a) * (d_a + Delta_minus[q])
        
        # P(d) = 1 - exp(-lambda(d)) where lambda(d) = A * exp(-k*d) * dt
        # g(d) = (1 - exp(-lambda(d))) * (d + Delta)
        
        # We can use a simple grid search or Newton's method.
        # Given the smoothness, a small grid search with refinement or just a decent grid is fast.
        # Let's try a grid search with 100 points for speed and accuracy balance.
        # Range: [0, 5/k]
        
        max_depth = 5.0 / k
        min_depth = 0.0
        
        # Create a depth grid
        # To be efficient, we can precompute a grid of depths and corresponding P values?
        # But Delta depends on q, so we can't precompute the full objective.
        # However, we can precompute lambda(d) and P(d) for a grid of d.
        
        d_grid = np.linspace(min_depth, max_depth, 200)
        lambda_grid = A * np.exp(-k * d_grid) * dt
        P_grid = 1.0 - np.exp(-lambda_grid)
        
        # Delta_plus[q] = V_next[q+1] - V_next[q]
        # Delta_minus[q] = V_next[q-1] - V_next[q]
        
        # Handle boundaries
        # q=Q (index Q): no bid fill. d_b can be anything, effectively 0 reward from bid.
        # Actually, if q=Q, we don't quote bid. So d_b = inf (or 0 prob).
        # In our model, we return depths. If we return a large depth, prob is ~0.
        # Let's just let the optimizer find the max. If Delta_plus is negative (which it should be for high q),
        # the max will be at small d or d=0?
        # Wait, if q=Q, we cannot increase inventory. So bid fill is impossible.
        # The term P_b * (d_b + Delta) is 0 because P_b is forced to 0 by the boundary condition.
        # How to implement?
        # If q=Q, set Delta_plus = -inf? Or just mask the result.
        # Let's mask the result: for q=Q, bid_depth = 0 (or large).
        # Actually, the prompt says "no bid fill at q=Q". It doesn't say we can't quote.
        # But if we quote and get filled, inventory would go to Q+1 which is invalid.
        # So effectively, we must not quote bid (or quote at infinite depth).
        # Let's set optimal depth to a large value (e.g., max_depth) for disabled sides.
        
        # Compute Deltas
        # V_next shape (num_q,)
        # Indices: 0..num_q-1 corresponding to q=-Q..Q
        
        # Delta_plus[i] = V_next[i+1] - V_next[i] for i < num_q-1
        # Delta_minus[i] = V_next[i-1] - V_next[i] for i > 0
        
        Delta_plus = np.zeros(num_q)
        Delta_minus = np.zeros(num_q)
        
        Delta_plus[:-1] = V_next[1:] - V_next[:-1]
        Delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        # For q=Q (last index), Delta_plus is undefined/irrelevant.
        # For q=-Q (first index), Delta_minus is undefined/irrelevant.
        
        # Objective for bid: O_b(d, Delta) = P(d) * (d + Delta)
        # We want argmax over d for each Delta.
        
        # Vectorize over d_grid (200) and q (num_q)
        # P_grid shape (200,)
        # d_grid shape (200,)
        # Delta_plus shape (num_q,)
        
        # O_b shape (num_q, 200)
        # O_b[i, j] = P_grid[j] * (d_grid[j] + Delta_plus[i])
        
        # But we must handle boundaries.
        # If q=Q, bid is disabled. We want O_b to be -inf or just pick a fixed depth.
        # Let's compute the max, then override boundaries.
        
        # Note: P(d) is decreasing in d. (d+Delta) is increasing in d.
        # The product is unimodal.
        
        # Compute O_b for all q and d
        # To save memory, we can do this in chunks or just let NumPy handle it.
        # num_q ~ 61, d_grid ~ 200. 61*200 = 12200 floats. Tiny.
        
        O_b = P_grid[None, :] * (d_grid[None, :] + Delta_plus[:, None])
        O_a = P_grid[None, :] * (d_grid[None, :] + Delta_minus[:, None])
        
        # Find argmax
        best_bid_idx = np.argmax(O_b, axis=1)
        best_ask_idx = np.argmax(O_a, axis=1)
        
        bid_depths = d_grid[best_bid_idx]
        ask_depths = d_grid[best_ask_idx]
        
        # Handle boundaries
        # q=Q (index Q): bid_depth should be large (no fill).
        # q=-Q (index 0): ask_depth should be large (no fill).
        # Actually, if we look at the objective:
        # If q=Q, we shouldn't take bid risk. 
        # In the HJB, the transition q->q+1 is forbidden.
        # So the term P_b * (d_b + V_next[q+1]) should be removed.
        # This is equivalent to setting V_next[q+1] = -inf? No.
        # It's equivalent to setting P_b = 0.
        # If P_b = 0, the objective is 0. Any d is optimal.
        # But we want to "not quote". A large depth implies low prob.
        # Let's just set the depth to max_depth for disabled sides.
        
        bid_depths[Q] = max_depth
        ask_depths[0] = max_depth
        
        # Store policy
        policy[n, :, 0] = bid_depths
        policy[n, :, 1] = ask_depths
        
        # Update V_next
        # V_curr[q] = max_action (Reward + Expected Future Value) - Risk Cost
        # We already found the max action. Now compute the value.
        
        # Reward + Future Value for bid side (using optimal d_b):
        # P_b * (d_b + V_next[q+1]) + (1-P_b) * V_next[q] ... wait.
        # The total value is:
        # P_b * P_a * (d_b + d_a + V_next[q])
        # + P_b * (1-P_a) * (d_b + V_next[q+1])
        # + (1-P_b) * P_a * (d_a + V_next[q-1])
        # + (1-P_b) * (1-P_a) * V_next[q]
        
        # This can be rewritten as:
        # V_next[q] + P_b * (d_b + V_next[q+1] - V_next[q]) + P_a * (d_a + V_next[q-1] - V_next[q])
        # - P_b * P_a * (V_next[q+1] + V_next[q-1] - 2*V_next[q])? No.
        
        # Let's just calculate it directly.
        P_b_opt = P_grid[best_bid_idx]
        P_a_opt = P_grid[best_ask_idx]
        
        # Handle boundaries for P
        # If q=Q, P_b should be 0.
        # If q=-Q, P_a should be 0.
        P_b_eff = P_b_opt.copy()
        P_a_eff = P_a_opt.copy()
        P_b_eff[Q] = 0.0
        P_a_eff[0] = 0.0
        
        # Terms
        # V_next[q+1]
        V_q_plus = np.zeros(num_q)
        V_q_plus[:-1] = V_next[1:]
        V_q_plus[-1] = V_next[-1] # Doesn't matter, P_b=0
        
        V_q_minus = np.zeros(num_q)
        V_q_minus[1:] = V_next[:-1]
        V_q_minus[0] = V_next[0] # Doesn't matter, P_a=0
        
        # Expected Value
        # E[V_next] = P_b*P_a*V_q + P_b*(1-P_a)*V_q_plus + (1-P_b)*P_a*V_q_minus + (1-P_b)*(1-P_a)*V_q
        #             = V_q + P_b*(V_q_plus - V_q) + P_a*(V_q_minus - V_q) - P_b*P_a*(V_q_plus + V_q_minus - 2*V_q)?
        # Let's use the expanded form to be safe.
        
        p_bb = P_b_eff * P_a_eff
        p_ba = P_b_eff * (1 - P_a_eff)
        p_ab = (1 - P_b_eff) * P_a_eff
        p_aa = (1 - P_b_eff) * (1 - P_a_eff)
        
        E_V = p_bb * V_next + p_ba * V_q_plus + p_ab * V_q_minus + p_aa * V_next
        
        # Reward
        # Bid reward: d_b if bid fills.
        # Ask reward: d_a if ask fills.
        # If both fill, we get d_b + d_a.
        # E[Reward] = p_bb * (d_b + d_a) + p_ba * d_b + p_ab * d_a
        
        # Note: d_b and d_a are the optimal depths we found.
        # But for q=Q, d_b is max_depth. P_b=0. So reward 0.
        
        E_R = p_bb * (bid_depths + ask_depths) + p_ba * bid_depths + p_ab * ask_depths
        
        # Risk Cost
        risk_cost = (gamma / 2.0) * (sigma ** 2) * (q_vals ** 2) * dt
        
        V_curr = E_R + E_V - risk_cost
        
        V_next = V_curr
        
    return policy

def quote(t, inventory, mid, params):
    """
    Returns optimal depths from precomputed HJB policy.
    """
    # Get policy
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _REGIME_CACHE:
        if time.time() > _DEADLINE - 10:
            # Fallback: AS heuristic
            sigma = params["sigma"]
            A = params["A"]
            k = params["k"]
            gamma = params["gamma"]
            T = params["T"]
            dt_rem = T - t
            R = gamma * (sigma ** 2) * inventory * dt_rem
            delta_star = 1.0 / k
            bid_depths = np.maximum(R + delta_star, 1e-6)
            ask_depths = np.maximum(-R + delta_star, 1e-6)
            return bid_depths, ask_depths
        
        # Precompute
        policy = _solve_regime(
            params['sigma'], params['A'], params['k'], 
            params['gamma'], params['T'], params['steps'], params['Q']
        )
        _REGIME_CACHE[key] = policy
    
    policy = _REGIME_CACHE[key]
    
    # Determine step index n
    steps = params['steps']
    T = params['T']
    dt = T / steps
    
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    # Inventory to index
    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    # Get depths
    depths = policy[n, q_idx, :]
    
    bid_depths = depths[:, 0]
    ask_depths = depths[:, 1]
    
    # Ensure validity
    bid_depths = np.maximum(bid_depths, 1e-6)
    ask_depths = np.maximum(ask_depths, 1e-6)
    
    return bid_depths, ask_depths

# Precompute known regimes at import time
_KNOWN_REGIMES = [
    (2, 140, 1.5, 0.05),
    (2, 140, 1.5, 0.5),
    (1.5, 50, 0.75, 0.2)
]

for sig, A, k, gam in _KNOWN_REGIMES:
    if time.time() > _DEADLINE - 30:
        break
    try:
        p = {'sigma': sig, 'A': A, 'k': k, 'gamma': gam, 'T': 1.0, 'steps': 1000, 'Q': 30}
        # Trigger precomputation
        _ = quote(0, np.zeros(1, dtype=int), np.array([100.0]), p)
    except Exception:
        pass
# EVOLVE-BLOCK-END
