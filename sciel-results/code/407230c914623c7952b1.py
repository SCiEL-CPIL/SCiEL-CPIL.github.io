# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 265.0
_SEARCH_DONE = False
_POLICY_CACHE = {}

def _get_regime_key(params):
    return (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])

def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    sigma2 = sigma * sigma
    sigma2_dt = sigma2 * dt
    
    # Precompute fill probability table
    # depth range: 0.01 to 5.0, 500 levels
    depth_min = 0.01
    depth_max = 5.0
    n_depths = 500
    depth_grid = np.linspace(depth_min, depth_max, n_depths)
    # p_fill(depth) = 1 - exp(-A * exp(-k * depth)) * dt
    p_fill_grid = (1.0 - np.exp(-A * np.exp(-k * depth_grid))) * dt
    p_fill_grid = np.clip(p_fill_grid, 0.0, 1.0)
    
    # Inventory states: 0 to 2*Q
    n_states = 2 * Q + 1
    q_mid = Q  # index for q=0
    
    # Value function: V[step, q_index]
    # We go backwards from step = steps to step = 0
    V_next = np.zeros(n_states, dtype=np.float64)
    bid_depths = np.zeros((steps, n_states), dtype=np.float64)
    ask_depths = np.zeros((steps, n_states), dtype=np.float64)
    
    penalty = 0.5 * gamma * sigma2_dt
    
    for s in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states, dtype=np.float64)
        
        # For each inventory state q (index i), choose bid_depth and ask_depth to maximize:
        # p_bid * (depth_bid - (q-1) penalty adjustment) + p_ask * (depth_ask - (q+1) penalty adjustment) + V_next - penalty * q^2
        # Actually: earnings - penalty where penalty uses q BEFORE fills
        # So: p_bid * depth_bid + p_ask * depth_ask + (1-p_bid)*(1-p_ask)*V_next[q] 
        #     + p_bid*(1-p_ask)*V_next[q-1] + (1-p_bid)*p_ask*V_next[q+1]
        #     + p_bid*p_ask*V_next[q]  -- wait, can both fill? Task says "independently"
        # Actually if both fill: bid fill (q-1) and ask fill (q+1), net change 0
        # So: p_bid*p_ask*V_next[q] + p_bid*(1-p_ask)*V_next[q-1] + (1-p_bid)*p_ask*V_next[q+1] + (1-p_bid)*(1-p_ask)*V_next[q]
        # But we need to handle boundaries: no bid fill at q=Q, no ask fill at q=-Q
        
        for qi in range(n_states):
            q = qi - Q
            
            # Try all combinations of bid_depth and ask_depth
            best_val = -np.inf
            best_bid = 0.0
            best_ask = 0.0
            
            # For speed, we can vectorize over depth choices
            # But let's do a smarter approach: the optimal depth for each side given the other
            
            # Actually, let's just do a coarse-to-fine search
            # First pass: coarse grid
            bid_probs = p_fill_grid
            ask_probs = p_fill_grid
            
            # Compute expected future value for each combination
            # This is O(n_depths^2) per state which is too slow
            # Let's use the fact that optimal depth separates in some cases
            
            # For now, let's use a greedy approach: optimize bid given ask=mu, then ask given bid=mu
            mu = 1.0 / k
            
            # Actually, let's do a 1D optimization assuming symmetric base
            # Try: bid = mu - delta, ask = mu + delta for various delta
            # Or independent: bid = mu * (1 - c*q), ask = mu * (1 + c*q)
            
            # Let's parameterize: bid = base_b - skew*q, ask = base_a + skew*q
            # Try a grid of (base_b, base_a, skew)
            
            # Simpler: for each q, the optimal policy should have bid <= ask
            # Let's try: bid = mu*(1 - alpha*q/Q), ask = mu*(1 + alpha*q/Q)
            # And also vary mu slightly
            
            # Given time constraints, let's use a parametric family
            # depth_bid = max(0.01, mu_b - skew * q)
            # depth_ask = max(0.01, mu_a + skew * q)
            
            # Optimize mu_b, mu_a, skew for each q... still expensive
            
            # Let's do a different approach: precompute optimal depths using a coarser state space
            # Or just use the analytical approximation
            
            # Given the 330s limit and 4000 paths, we need something fast
            # Let's use the continuous-time approximation:
            # Optimal bid: depth_b = (1/k) * log(A/k) + (gamma*sigma2*time_left*q)/k * something
            # From market making theory: skew = (gamma*sigma2*time_left)/k * q
            
            time_left = T - s * dt
            skew_coeff = gamma * sigma2 * time_left / k
            base = 1.0 / k
            
            # For this state, compute optimal depths using parametric family
            # Try several skew values
            skews = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0])
            
            best_val = -np.inf
            for skew in skews:
                bid_d = base - skew * skew_coeff * q
                ask_d = base + skew * skew_coeff * q
                bid_d = max(0.01, bid_d)
                ask_d = max(0.01, ask_d)
                
                # Compute fill probabilities
                p_b = (1.0 - np.exp(-A * np.exp(-k * bid_d))) * dt
                p_a = (1.0 - np.exp(-A * np.exp(-k * ask_d))) * dt
                p_b = np.clip(p_b, 0.0, 1.0)
                p_a = np.clip(p_a, 0.0, 1.0)
                
                # Boundary conditions
                if qi == n_states - 1:  # q = Q, no bid fill
                    p_b = 0.0
                if qi == 0:  # q = -Q, no ask fill
                    p_a = 0.0
                
                # Expected value
                # Both fill: q unchanged (bid: q-1, ask: q+1, net 0)
                # Bid only: q-1
                # Ask only: q+1
                # Neither: q
                ev = (p_b * (1.0 - p_a) * V_next[max(0, qi-1)] +
                      (1.0 - p_b) * p_a * V_next[min(n_states-1, qi+1)] +
                      (1.0 - p_b) * (1.0 - p_a) * V_next[qi] +
                      p_b * p_a * V_next[qi])
                
                # Earnings
                earnings = p_b * bid_d + p_a * ask_d
                # Penalty
                pen = penalty * q * q
                
                total = earnings + ev - pen
                
                if total > best_val:
                    best_val = total
                    best_bid = bid_d
                    best_ask = ask_d
            
            V_curr[qi] = best_val
            bid_depths[s, qi] = best_bid
            ask_depths[s, qi] = best_ask
        
        V_next = V_curr
        
        if time.time() > _DEADLINE - 10.0:
            break
    
    return bid_depths, ask_depths

def _run_search():
    global _SEARCH_DONE
    if _SEARCH_DONE:
        return
    _SEARCH_DONE = True
    
    regimes = [
        {"sigma": 2.0, "A": 140.0, "k": 1.5, "gamma": 0.05},
        {"sigma": 2.0, "A": 140.0, "k": 1.5, "gamma": 0.5},
        {"sigma": 1.5, "A": 50.0, "k": 0.75, "gamma": 0.2},
    ]
    
    for reg in regimes:
        key = (reg["sigma"], reg["A"], reg["k"], reg["gamma"], 1.0, 1000, 30)
        if key not in _POLICY_CACHE:
            if time.time() > _DEADLINE - 5.0:
                break
            bid_tab, ask_tab = _compute_optimal_policy(
                reg["sigma"], reg["A"], reg["k"], reg["gamma"], 1.0, 1000, 30
            )
            _POLICY_CACHE[key] = (bid_tab, ask_tab)

def _fallback_quote(t, inventory, mid, params):
    sigma = params["sigma"]
    gamma = params["gamma"]
    k = params["k"]
    Q = params["Q"]
    T = params["T"]
    time_left = max(T - t, 0.0)
    
    base = 1.0 / k
    skew_coeff = gamma * sigma * sigma * time_left / k
    
    bid_depths = base - skew_coeff * inventory.astype(np.float64)
    ask_depths = base + skew_coeff * inventory.astype(np.float64)
    
    bid_depths = np.clip(bid_depths, 0.01, 10.0)
    ask_depths = np.clip(ask_depths, 0.01, 10.0)
    
    return (bid_depths, ask_depths)

def quote(t, inventory, mid, params):
    global _SEARCH_DONE
    if not _SEARCH_DONE:
        _run_search()
    
    key = _get_regime_key(params)
    
    if key in _POLICY_CACHE:
        bid_tab, ask_tab = _POLICY_CACHE[key]
        steps = params["steps"]
        Q = params["Q"]
        
        dt = params["T"] / steps
        t_idx = int(round(t / dt))
        t_idx = max(0, min(steps - 1, t_idx))
        
        inv_clipped = np.clip(inventory, -Q, Q)
        idx_q = inv_clipped + Q
        
        bid_depths = bid_tab[t_idx, idx_q]
        ask_depths = ask_tab[t_idx, idx_q]
        
        return (bid_depths.astype(np.float64), ask_depths.astype(np.float64))
    else:
        return _fallback_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
