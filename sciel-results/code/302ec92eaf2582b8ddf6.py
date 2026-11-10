"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via DP over (time, inventory) states.
    Vectorized over inventory for speed.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    # Adaptive depth grid: optimal depth is around 1/k, search wider
    depth_min = 1e-4
    depth_max = max(8.0 / k, 1.0)
    n_depths = 80
    depth_grid = np.linspace(depth_min, depth_max, n_depths)
    
    # Precompute fill probabilities for each depth
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # V_next[q_idx] where q_idx = q + Q, q in [-Q, Q]
    V_next = np.zeros(2 * Q + 1)
    
    # Store optimal depths: shape (steps, 2*Q+1)
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 3.0:
            # Fill remaining steps with current best estimate
            for nn in range(n - 1, -1, -1):
                opt_bid[nn] = opt_bid[n]
                opt_ask[nn] = opt_ask[n]
            break
        
        V_curr = np.zeros(2 * Q + 1)
        
        # Precompute penalty for each q
        q_vals = np.arange(-Q, Q + 1)
        penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        # For each inventory level, find optimal bid and ask depths
        # Since bid and ask are independent given V_next, we can optimize each separately
        # but the cross-term (both fill) couples them. However, for small dt,
        # P(both) is very small, so we can approximate by optimizing independently.
        
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            penalty = penalties[q_idx]
            
            if q == Q:
                # Only ask quote allowed (bid fills prevented)
                # Ask fill: q -> q-1
                # Objective: d * p_ask - penalty + (1-p_ask)*V_next[q_idx] + p_ask*V_next[q_idx-1]
                # = d*p_ask - penalty + V_next[q_idx] + p_ask*(V_next[q_idx-1] - V_next[q_idx])
                # Maximize over d: d*p_ask + p_ask * delta - penalty + V_next[q_idx]
                # = p_ask * (d + delta) - penalty + V_next[q_idx]
                # where delta = V_next[q_idx-1] - V_next[q_idx]
                # Since p_ask is increasing in d (deeper = more likely to fill),
                # and d + delta: if delta < 0 (V decreases as q decreases), 
                # we want to balance.
                delta = V_next[q_idx - 1] - V_next[q_idx]
                # Maximize p_ask(d) * (d + delta)
                # Try all depths
                vals = fill_probs * (depth_grid + delta)
                best_i = np.argmax(vals)
                best_val = vals[best_i] - penalty + V_next[q_idx]
                
                V_curr[q_idx] = best_val
                opt_ask[n, q_idx] = depth_grid[best_i]
                # No bid quote
                opt_bid[n, q_idx] = np.inf
                
            elif q == -Q:
                # Only bid quote allowed
                # Bid fill: q -> q+1
                # Objective: d * p_bid - penalty + (1-p_bid)*V_next[q_idx] + p_bid*V_next[q_idx+1]
                # = p_bid * (d + delta) - penalty + V_next[q_idx]
                # where delta = V_next[q_idx+1] - V_next[q_idx]
                delta = V_next[q_idx + 1] - V_next[q_idx]
                vals = fill_probs * (depth_grid + delta)
                best_i = np.argmax(vals)
                best_val = vals[best_i] - penalty + V_next[q_idx]
                
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = depth_grid[best_i]
                opt_ask[n, q_idx] = np.inf
                
            else:
                # Both sides allowed
                # Bid fill: q -> q+1, Ask fill: q -> q-1
                # Both fill: q -> q (net zero)
                # Objective: db*p_bid + da*p_ask - penalty 
                #   + p_bid*p_ask*V_next[q_idx]
                #   + p_bid*(1-p_ask)*V_next[q_idx+1]
                #   + (1-p_bid)*p_ask*V_next[q_idx-1]
                #   + (1-p_bid)*(1-p_ask)*V_next[q_idx]
                # = db*p_bid + da*p_ask - penalty
                #   + p_bid*V_next[q_idx+1] + p_ask*V_next[q_idx-1]
                #   + (V_next[q_idx] - V_next[q_idx+1])*p_bid*(1-p_ask)  -- wait, let me redo
                # 
                # Let me expand properly:
                # next_val = p_bid*p_ask*V[q] + p_bid*(1-p_ask)*V[q+1] + (1-p_bid)*p_ask*V[q-1] + (1-p_bid)*(1-p_ask)*V[q]
                # = p_bid*p_ask*V[q] + p_bid*V[q+1] - p_bid*p_ask*V[q+1] + p_ask*V[q-1] - p_bid*p_ask*V[q-1] + V[q] - p_bid*V[q] - p_ask*V[q] + p_bid*p_ask*V[q]
                # = V[q] + p_bid*(V[q+1] - V[q]) + p_ask*(V[q-1] - V[q]) - p_bid*p_ask*(V[q+1] - 2*V[q] + V[q-1])
                # 
                # For small dt, the cross term p_bid*p_ask is O(dt^2) and small.
                # Approximate: next_val ≈ V[q] + p_bid*delta_plus + p_ask*delta_minus
                # where delta_plus = V[q+1] - V[q], delta_minus = V[q-1] - V[q]
                #
                # So total: db*p_bid + da*p_ask - penalty + V[q] + p_bid*delta_plus + p_ask*delta_minus
                # = p_bid*(db + delta_plus) + p_ask*(da + delta_minus) - penalty + V[q]
                #
                # Since bid and ask are independent in this approximation, optimize each separately!
                
                delta_plus = V_next[q_idx + 1] - V_next[q_idx]
                delta_minus = V_next[q_idx - 1] - V_next[q_idx]
                
                # Optimize bid: maximize p_bid(d) * (d + delta_plus)
                vals_bid = fill_probs * (depth_grid + delta_plus)
                best_bi = np.argmax(vals_bid)
                
                # Optimize ask: maximize p_ask(d) * (d + delta_minus)
                vals_ask = fill_probs * (depth_grid + delta_minus)
                best_ai = np.argmax(vals_ask)
                
                best_val = vals_bid[best_bi] + vals_ask[best_ai] - penalty + V_next[q_idx]
                
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = depth_grid[best_bi]
                opt_ask[n, q_idx] = depth_grid[best_ai]
        
        V_next = V_curr
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _analytical_quote(t, inventory, mid, params):
    """
    Avellaneda-Stoikov style analytical quote.
    depth = 1/k * (1 + gamma * sigma^2 * (T-t) * q / (2 * something))
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    remaining = T - t
    
    # Base depth from risk-neutral pricing
    base = 1.0 / k
    
    # Inventory skew: more inventory -> push bid deeper (less willing to buy), ask shallower (more willing to sell)
    # The optimal skew coefficient from AS model: gamma * sigma^2 * (T-t) / 2
    skew_coeff = gamma * sigma**2 * remaining / 2.0
    
    # Skew in depth space: the optimal depth shift is proportional to inventory
    # In the AS model, the spread widens and the mid shifts.
    # For our model, a good approximation:
    # bid_depth = base + c * q, ask_depth = base - c * q
    # where c captures the inventory risk aversion
    
    # Scale c to be in the right range. The penalty is (gamma/2)*sigma^2*q^2*dt
    # The marginal cost of one more unit of inventory is gamma*sigma^2*q*dt
    # This should balance against the expected profit from filling.
    # A reasonable heuristic: c = gamma * sigma^2 * remaining / (2 * k)
    c = skew_coeff / (2.0 * k)
    
    bid_depths = base + c * inventory
    ask_depths = base - c * inventory
    
    # Ensure valid depths
    bid_depths = np.clip(bid_depths, 1e-4, 10.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 10.0 / k)
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with analytical fallback."""
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
    if n < 0:
        n = 0
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    # Try to compute policy if not cached and we have time
    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 15.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except Exception:
                pass
    
    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        # Map inventory to index
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        # Replace 0 with inf (no quote)
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        # Ensure shape is correct
        bid_depths = np.asarray(bid_depths, dtype=np.float64)
        ask_depths = np.asarray(ask_depths, dtype=np.float64)
        return bid_depths, ask_depths
    else:
        # Fallback: analytical quote
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
