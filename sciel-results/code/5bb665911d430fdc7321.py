"""Optimal market making with inventory risk using continuous-depth HJB backward induction."""
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0

def _solve_1d_max(f, x_min, x_max, n_iter=60):
    """
    Find the maximum of a 1D function f on [x_min, x_max] using golden-section search.
    Assumes f is unimodal (or at least has a single dominant peak).
    """
    if x_min >= x_max:
        x_min, x_max = x_max, x_min
    
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio
    
    a = x_min
    b = x_max
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc = f(c)
    fd = f(d)
    
    for _ in range(n_iter):
        if fc < fd:
            a = c
            c = d
            d = a + gr * (b - a)
            fc = fd
            fd = f(d)
        else:
            b = d
            d = c
            c = b - gr * (b - a)
            fd = fc
            fc = f(c)
    
    x_opt = (a + b) / 2.0
    return x_opt

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time HJB equation with
    continuous depth optimization via golden-section search.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    V_next = np.zeros(n_inv)
    
    # Depth bounds for optimization
    depth_min = 1e-8
    depth_max = 20.0 / k  # generous upper bound
    
    bid_table = np.full((steps, n_inv), np.inf)
    ask_table = np.full((steps, n_inv), np.inf)
    
    V_cur = np.zeros(n_inv)
    
    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() > _DEADLINE:
            # Fill remaining with a simple heuristic
            for i in range(n_inv):
                q = inv_range[i]
                bid_table[n, i] = 1.0 / k
                ask_table[n, i] = 1.0 / k
                V_cur[i] = -0.5 * gamma * sigma**2 * q**2 * dt + V_next[i]
            V_next = V_cur.copy()
            continue
        
        for i in range(n_inv):
            q = inv_range[i]
            inv_cost = -0.5 * gamma * sigma**2 * q**2 * dt
            
            can_bid = q < Q
            can_ask = q > -Q
            
            if can_bid and can_ask:
                c1 = V_next[i+1] - V_next[i]
                c2 = V_next[i-1] - V_next[i]
                c3 = -c1 - c2
                
                # Objective as function of (bd, ad):
                # p(bd)*bd + p(ad)*ad + p(bd)*c1 + p(ad)*c2 + p(bd)*p(ad)*c3
                # where p(d) = 1 - exp(-A*exp(-k*d)*dt)
                
                # For fixed ad, optimize bd:
                # g(bd) = p(bd)*(bd + c1 + c3*p(ad)) + const
                # For fixed bd, optimize ad:
                # h(ad) = p(ad)*(ad + c2 + c3*p(bd)) + const
                
                # Alternating optimization
                bd = 1.0 / k
                ad = 1.0 / k
                
                for _ in range(10):
                    # Optimize bd given ad
                    p_ad = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
                    coeff_b = bd + c1 + c3 * p_ad  # not quite right, let me redo
                    
                    # Actually: g(bd) = p(bd)*bd + p(bd)*c1 + p(bd)*p(ad)*c3 + [terms in ad]
                    #          = p(bd) * (bd + c1 + p(ad)*c3) + [terms in ad]
                    # So maximize p(bd) * (bd + c1 + p_ad * c3)
                    
                    p_ad_val = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
                    
                    def obj_bd(b):
                        p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                        return p_b * (b + c1 + p_ad_val * c3)
                    
                    bd = _solve_1d_max(obj_bd, depth_min, depth_max, n_iter=40)
                    
                    # Optimize ad given bd
                    p_bd_val = 1.0 - np.exp(-A * np.exp(-k * bd) * dt)
                    
                    def obj_ad(a):
                        p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                        return p_a * (a + c2 + p_bd_val * c3)
                    
                    ad = _solve_1d_max(obj_ad, depth_min, depth_max, n_iter=40)
                
                bid_table[n, i] = bd
                ask_table[n, i] = ad
                
                # Compute total value
                p_b = 1.0 - np.exp(-A * np.exp(-k * bd) * dt)
                p_a = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
                max_val = p_b * bd + p_a * ad + p_b * c1 + p_a * c2 + p_b * p_a * c3
                
                V_cur[i] = inv_cost + V_next[i] + max_val
                
            elif can_bid and not can_ask:
                c1 = V_next[i+1] - V_next[i]
                
                def obj_bd(b):
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    return p_b * (b + c1)
                
                bd = _solve_1d_max(obj_bd, depth_min, depth_max, n_iter=40)
                bid_table[n, i] = bd
                ask_table[n, i] = np.inf
                
                p_b = 1.0 - np.exp(-A * np.exp(-k * bd) * dt)
                max_val = p_b * (bd + c1)
                V_cur[i] = inv_cost + V_next[i] + max_val
                
            elif can_ask and not can_bid:
                c2 = V_next[i-1] - V_next[i]
                
                def obj_ad(a):
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    return p_a * (a + c2)
                
                ad = _solve_1d_max(obj_ad, depth_min, depth_max, n_iter=40)
                bid_table[n, i] = np.inf
                ask_table[n, i] = ad
                
                p_a = 1.0 - np.exp(-A * np.exp(-k * ad) * dt)
                max_val = p_a * (ad + c2)
                V_cur[i] = inv_cost + V_next[i] + max_val
                
            else:
                bid_table[n, i] = np.inf
                ask_table[n, i] = np.inf
                V_cur[i] = inv_cost + V_next[i]
        
        V_next = V_cur.copy()
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths based on pre-computed policy table."""
    steps = params['steps']
    Q = params['Q']
    
    # Get current step
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index: q in [-Q, Q] -> i in [0, 2Q]
    inv_idx = np.asarray(inventory, dtype=int) + Q
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get depths
    bid_depths = bid_table[n, inv_idx].astype(float)
    ask_depths = ask_table[n, inv_idx].astype(float)
    
    # Ensure values are valid: >= -3/k, not -inf, not NaN
    k = params['k']
    min_depth = -3.0 / k
    
    # Replace any values below min_depth with min_depth
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Replace NaN with a default value
    bid_depths = np.nan_to_num(bid_depths, nan=1.0/k, posinf=np.inf, neginf=min_depth)
    ask_depths = np.nan_to_num(ask_depths, nan=1.0/k, posinf=np.inf, neginf=min_depth)
    
    return bid_depths, ask_depths
