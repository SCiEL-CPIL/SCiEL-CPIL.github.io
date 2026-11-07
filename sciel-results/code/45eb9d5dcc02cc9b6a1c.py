import numpy as np
import time

_T0 = time.time()
_DEADLINE = 290.0
_DP_CACHE = {}

def _solve_dp_continuous(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _DP_CACHE:
        return _DP_CACHE[key]

    dt = T / steps
    n_states = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    # V[t, q_idx]
    V = np.zeros((steps + 1, n_states))
    opt_bid = np.zeros((steps + 1, n_states))
    opt_ask = np.zeros((steps + 1, n_states))

    costs = 0.5 * gamma * sigma**2 * q_indices**2 * dt

    # Helper to compute p and r for a given depth
    def get_p_r(d):
        # d can be array or scalar
        lam = A * np.exp(-k * d) * dt
        p = 1.0 - np.exp(-lam)
        r = p * d
        return p, r

    # Golden section search for max of a univariate function
    # f(x) is concave
    def golden_max(f, a, b, tol=1e-4, max_iter=50):
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        for _ in range(max_iter):
            if time.time() > _DEADLINE:
                break
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
            if abs(b - a) < tol:
                break
        return (a + b) / 2.0

    for t in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fill remaining with default values to avoid errors
            for t_rem in range(t, -1, -1):
                opt_bid[t_rem+1] = 1.0/k
                opt_ask[t_rem+1] = 1.0/k
                V[t_rem+1] = V[t+1]
            break

        V_next = V[t + 1]

        for q_idx in range(n_states):
            q = q_indices[q_idx]
            
            # Boundaries
            if q == Q:
                # No bid
                opt_bid[t + 1, q_idx] = 0.0
                
                # Optimize Ask
                # J(da) = da * p_a + p_a * D2 - p_a * p_b * D3 ... wait, p_b=0 if d_b=0?
                # If d_b=0, p_b = 1 - exp(0) = 0.
                # So J(da) = da * p_a + p_a * D2 + V0 - cost
                # D2 = V(q-1) - V(q)
                D2 = V_next[q_idx - 1] - V_next[q_idx]
                
                def obj_ask(da):
                    p_a, r_a = get_p_r(da)
                    return r_a + p_a * D2
                
                d_a = golden_max(obj_ask, 0.0, 5.0/k)
                opt_ask[t + 1, q_idx] = d_a
                p_a, r_a = get_p_r(d_a)
                V[t + 1, q_idx] = V_next[q_idx] + r_a + p_a * D2 - costs[q_idx]
                
            elif q == -Q:
                # No ask
                opt_ask[t + 1, q_idx] = 0.0
                
                # Optimize Bid
                D1 = V_next[q_idx + 1] - V_next[q_idx]
                
                def obj_bid(db):
                    p_b, r_b = get_p_r(db)
                    return r_b + p_b * D1
                
                d_b = golden_max(obj_bid, 0.0, 5.0/k)
                opt_bid[t + 1, q_idx] = d_b
                p_b, r_b = get_p_r(d_b)
                V[t + 1, q_idx] = V_next[q_idx] + r_b + p_b * D1 - costs[q_idx]
                
            else:
                D1 = V_next[q_idx + 1] - V_next[q_idx]
                D2 = V_next[q_idx - 1] - V_next[q_idx]
                D3 = V_next[q_idx + 1] - 2 * V_next[q_idx] + V_next[q_idx - 1]
                
                # Iterate: optimize db given da, then da given db
                d_b = 1.0 / k
                d_a = 1.0 / k
                
                for _ in range(3):
                    # Optimize db
                    p_a, r_a = get_p_r(d_a)
                    def obj_bid(db):
                        p_b, r_b = get_p_r(db)
                        # Term: r_b + r_a + p_b D1 + p_a D2 - p_b p_a D3
                        # Constants: r_a + p_a D2
                        # Variable: r_b + p_b D1 - p_b p_a D3
                        return r_b + p_b * (D1 - p_a * D3)
                    
                    d_b_new = golden_max(obj_bid, 0.0, 5.0/k)
                    
                    # Optimize da
                    p_b, r_b = get_p_r(d_b_new)
                    def obj_ask(da):
                        p_a, r_a = get_p_r(da)
                        # Variable: r_a + p_a D2 - p_a p_b D3
                        return r_a + p_a * (D2 - p_b * D3)
                    
                    d_a_new = golden_max(obj_ask, 0.0, 5.0/k)
                    
                    d_b = d_b_new
                    d_a = d_a_new
                    
                opt_bid[t + 1, q_idx] = d_b
                opt_ask[t + 1, q_idx] = d_a
                
                p_b, r_b = get_p_r(d_b)
                p_a, r_a = get_p_r(d_a)
                V[t + 1, q_idx] = V_next[q_idx] + r_b + r_a + p_b * D1 + p_a * D2 - p_b * p_a * D3 - costs[q_idx]

    _DP_CACHE[key] = (opt_bid, opt_ask)
    return _DP_CACHE[key]

def quote(t, inventory, mid, params):
    if time.time() > _DEADLINE:
        k = params['k']
        d = 1.0 / k
        return np.full(len(inventory), d), np.full(len(inventory), d)

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    opt_bid, opt_ask = _solve_dp_continuous(sigma, A, k, gamma, T, steps, Q)

    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    n = int(t / (T / steps))
    n = np.clip(n, 0, steps - 1)

    bid_depths = opt_bid[n, q_idx].copy()
    ask_depths = opt_ask[n, q_idx].copy()

    # Error-repair pass
    d_min_valid = 0.0
    d_max_valid = 5.0 / k
    bid_depths = np.clip(bid_depths, d_min_valid, d_max_valid)
    ask_depths = np.clip(ask_depths, d_min_valid, d_max_valid)

    bad_bid = ~np.isfinite(bid_depths) | (bid_depths < -3.0 / k)
    if np.any(bad_bid):
        bid_depths[bad_bid] = d_min_valid
    bad_ask = ~np.isfinite(ask_depths) | (ask_depths < -3.0 / k)
    if np.any(bad_ask):
        ask_depths[bad_ask] = d_min_valid

    return bid_depths, ask_depths
