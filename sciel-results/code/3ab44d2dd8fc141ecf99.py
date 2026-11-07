import numpy as np
import time
from scipy.optimize import brentq

_DEADLINE = time.time() + 290
_POLICY_CACHE = {}

def _solve_dp(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    alpha = A * dt
    
    # Precompute optimal depth for a range of delta_v
    # delta_v = V_next[q'] - V_next[q]
    # We need to cover the possible range of delta_v.
    # V is roughly O(Q^2 * gamma * sigma^2 * T).
    # delta_v is O(Q * gamma * sigma^2 * T).
    # Let's use a wide range, e.g., [-1000, 1000]
    # And interpolate.
    
    dv_min = -1000.0
    dv_max = 1000.0
    dv_grid = np.linspace(dv_min, dv_max, 2000)
    d_opt_grid = np.zeros(len(dv_grid))
    
    # Function to solve for d given dv
    def solve_d(dv):
        # h(u) = alpha * u * exp(-alpha * u) * (dv - (1/k) * ln(u)) - (1 - exp(-alpha * u))
        # u in (0, 1]
        # If dv is very large positive, we want small d (large u)?
        # If dv is very large negative, we want large d (small u)?
        
        # Let's check limits.
        # If u -> 1 (d -> 0):
        # h(1) = alpha * 1 * e^{-alpha} * (dv - 0) - (1 - e^{-alpha})
        # If u -> 0 (d -> inf):
        # alpha * u * e^{-alpha u} ~ alpha * u
        # ln(u) -> -inf
        # If dv > 0: (dv - 1/k ln u) -> inf. h(u) -> inf?
        # If dv < 0: (dv - 1/k ln u) -> inf. h(u) -> inf?
        
        # Actually, we want to maximize f(d).
        # If no solution, check boundaries.
        
        def h(u):
            if u <= 0: return 0
            if u > 1: u = 1.0
            exp_term = np.exp(-alpha * u)
            return alpha * u * exp_term * (dv - (1.0/k) * np.log(u)) - (1.0 - exp_term)
        
        # We know h(1) might be positive or negative.
        # h(u) for small u:
        # alpha * u * (1 - alpha u) * (dv - (1/k) ln u) - (1 - (1 - alpha u + ...))
        # ~ alpha u (dv - 1/k ln u) - alpha u
        # If dv < 0, -1/k ln u is large positive. So h(u) > 0 for small u.
        # So there is a root if h(1) < 0.
        
        u_lo = 1e-8
        u_hi = 1.0
        
        try:
            if h(u_hi) > 0:
                # Root is not in (u_lo, u_hi)? Or h is always positive?
                # If h(1) > 0, max is at boundary d=0?
                # Let's check f(d) at d=0 vs d=inf.
                # f(0) = (1-e^{-alpha}) * (0 + dv)
                # f(inf) = 1 * (inf + dv) -> inf? No, p(d) -> 1.
                # Wait, if p(d) -> 1, f(d) -> d + dv. This goes to infinity?
                # No, the probability model: p(d) = 1 - exp(-A exp(-kd) dt).
                # As d -> inf, exp(-kd) -> 0, p(d) -> 0?
                # NO. exp(-k*d). If d > 0, exp(-kd) < 1.
                # As d -> 0, exp(-kd) -> 1. p(0) = 1 - exp(-A dt).
                # As d -> inf, exp(-kd) -> 0. p(inf) = 0.
                # So p(d) is decreasing in d.
                # My previous analysis assumed p increases with d?
                # "bid hit with probability 1 - exp(-A exp(-k bid_depth) dt)"
                # If bid_depth is distance below mid.
                # Usually, tighter quotes (smaller depth) have higher fill probability.
                # So small d -> high p. Large d -> low p.
                # This matches p(d) = 1 - exp(-A e^{-kd} dt).
                # So we are maximizing p(d)(d + dv) + (1-p(d))V_q.
                # As d -> 0, p -> 1-e^{-alpha}. Reward ~ (1-e^{-alpha})dv.
                # As d -> inf, p -> 0. Reward -> 0?
                # Wait, if p -> 0, we don't get the depth d.
                # So f(d) -> 0?
                # Let's check f(d) for large d.
                # p(d) ~ A e^{-kd} dt.
                # f(d) ~ A dt e^{-kd} (d + dv).
                # This goes to 0 as d -> inf.
                # So there is a finite maximum.
                
                # If h(1) > 0, it means f'(0) > 0?
                # f'(d) at d=0.
                # If f'(0) > 0, we should increase d?
                # But f(d) starts at (1-e^{-alpha})dv and goes to 0.
                # If dv > 0, f(0) > 0. If f'(0) > 0, it goes up then down.
                # If dv < 0, f(0) < 0. f(d) must go up to positive?
                # Max of f(d) is positive?
                # If we quote at d, we earn d with prob p(d).
                # If dv is very negative, we don't want to fill.
                # But we can always choose d such that p(d) is small.
                
                # Let's just use brentq on a safe interval.
                # If h(1) > 0, the root might be outside or we are at boundary.
                # Given the shape, if h(1) > 0, the max is likely at small d?
                # Let's return a small d.
                return 0.0 
            else:
                # h(1) <= 0 and h(u_lo) > 0 (usually)
                root_u = brentq(h, u_lo, u_hi, xtol=1e-10)
                return -1.0/k * np.log(root_u)
        except Exception:
            return 0.0

    for i, dv in enumerate(dv_grid):
        d_opt_grid[i] = solve_d(dv)
        
    # Interpolation function
    def get_opt_depth(dv):
        if dv < dv_min: dv = dv_min
        if dv > dv_max: dv = dv_max
        return np.interp(dv, dv_grid, d_opt_grid)

    # DP Loop
    num_states = 2 * Q + 1
    V_next = np.zeros(num_states)
    
    bid_table = np.full((steps, num_states), np.inf)
    ask_table = np.full((steps, num_states), np.inf)
    
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    penalty = 0.5 * gamma * (sigma ** 2) * (q_vals ** 2) * dt
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_states)
        
        # Bid: q -> q+1
        # Valid for q < Q (indices 0 to num_states-2)
        idx_bid = np.arange(num_states - 1)
        dv_bid = V_next[idx_bid + 1] - V_next[idx_bid]
        d_bid = get_opt_depth(dv_bid)
        
        # Calculate value: p(d)(d + V_next[q+1]) + (1-p(d))V_next[q]
        p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt)
        val_bid = p_bid * (d_bid + V_next[idx_bid + 1]) + (1.0 - p_bid) * V_next[idx_bid]
        
        bid_table[n, :num_states-1] = d_bid
        bid_table[n, num_states-1] = np.inf
        V_curr[:num_states-1] = val_bid
        V_curr[num_states-1] = V_next[num_states-1] # No bid at Q
        
        # Ask: q -> q-1
        # Valid for q > -Q (indices 1 to num_states-1)
        idx_ask = np.arange(1, num_states)
        dv_ask = V_next[idx_ask - 1] - V_next[idx_ask]
        d_ask = get_opt_depth(dv_ask)
        
        p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt)
        val_ask = p_ask * (d_ask + V_next[idx_ask - 1]) + (1.0 - p_ask) * V_next[idx_ask]
        
        ask_table[n, 1:num_states] = d_ask
        ask_table[n, 0] = np.inf
        V_curr[1:num_states] = val_ask
        V_curr[0] = V_next[0] # No ask at -Q
        
        V_curr -= penalty
        V_next = V_curr
        
    return bid_table, ask_table

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        Q = params['Q']
        steps = params['steps']
        T = params['T']
        dt = T / steps
        remaining_time = T - t
        time_fraction = remaining_time / T
        q = inventory.astype(np.float64)
        base_depth = 1.0 / k
        beta = 0.02 * (1.0 + 0.5 * sigma) * (0.5 + 0.5 * time_fraction)
        q_scaled = q / Q
        q_adjust = np.tanh(1.5 * q_scaled)
        bid_depth = base_depth + beta * q_adjust
        ask_depth = base_depth - beta * q_adjust
        bid_depth = np.maximum(bid_depth, 0.01)
        ask_depth = np.maximum(ask_depth, 0.01)
        bid_depth[inventory >= Q] = np.inf
        ask_depth[inventory <= -Q] = np.inf
        return bid_depth, ask_depth

    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 10:
            _POLICY_CACHE[key] = _solve_dp(params)
        else:
            sigma = params['sigma']
            A = params['A']
            k = params['k']
            gamma = params['gamma']
            Q = params['Q']
            steps = params['steps']
            T = params['T']
            dt = T / steps
            remaining_time = T - t
            time_fraction = remaining_time / T
            q = inventory.astype(np.float64)
            base_depth = 1.0 / k
            beta = 0.02 * (1.0 + 0.5 * sigma) * (0.5 + 0.5 * time_fraction)
            q_scaled = q / Q
            q_adjust = np.tanh(1.5 * q_scaled)
            bid_depth = base_depth + beta * q_adjust
            ask_depth = base_depth - beta * q_adjust
            bid_depth = np.maximum(bid_depth, 0.01)
            ask_depth = np.maximum(ask_depth, 0.01)
            bid_depth[inventory >= Q] = np.inf
            ask_depth[inventory <= -Q] = np.inf
            return bid_depth, ask_depth
            
    bid_table, ask_table = _POLICY_CACHE[key]
    
    steps = params['steps']
    T = params['T']
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    Q = params['Q']
    indices = inventory + Q
    
    bid_depths = bid_table[n, indices]
    ask_depths = ask_table[n, indices]
    
    return bid_depths, ask_depths
