import numpy as np
from scipy.optimize import minimize_scalar

# Cache for DP tables
_dp_cache = {}

def _compute_dp_tables(sigma, A, k, gamma, T, steps, Q):
    """
    Pre-compute optimal bid and ask depths for all (t, q) states.
    Returns two arrays of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # V_next[q - q_min]
    V_next = np.zeros(n_states, dtype=np.float64)
    
    opt_bid = np.full((steps, n_states), np.inf)
    opt_ask = np.full((steps, n_states), np.inf)
    
    # Pre-compute constants
    # Maximize f(d) = d * p(d) - p(d) * delta
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # Let y = A * exp(-k*d) * dt
    # d = (1/k) * log(A*dt / y)
    # p = 1 - exp(-y)
    # f(y) = (1/k) * log(A*dt / y) * (1 - exp(-y)) - y * (delta / (A*dt))
    
    A_dt = A * dt
    
    def objective_bid(y, delta):
        if y <= 0: return -np.inf
        d = (1.0 / k) * np.log(A_dt / y)
        if d < -3.0/k: return -np.inf # Invalid depth
        p = 1.0 - np.exp(-y)
        return d * p - p * delta

    def objective_ask(y, delta):
        if y <= 0: return -np.inf
        d = (1.0 / k) * np.log(A_dt / y)
        if d < -3.0/k: return -np.inf
        p = 1.0 - np.exp(-y)
        return d * p - p * delta

    # We can reuse the same objective for both, just different delta
    # y range: y is small. A*dt is ~ 0.1 to 0.14.
    # y can be from 1e-6 to A*dt (d->inf) or larger (d-> -inf, but we clamp d).
    # Actually d > -3/k implies y < A*dt * exp(3).
    
    y_min = 1e-8
    y_max = A_dt * np.exp(3.0) + 1e-8
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states, dtype=np.float64)
        inv_penalty = (gamma / 2.0) * (sigma ** 2) * dt
        
        for i in range(n_states):
            q = i + q_min
            
            # Calculate deltas from V_next
            # delta_minus = V_next[q] - V_next[q-1]
            # delta_plus = V_next[q+1] - V_next[q]
            
            if q > q_min:
                delta_minus = V_next[i] - V_next[i-1]
            else:
                delta_minus = 0.0 # No bid possible at boundary? 
                # Actually, if q=-Q, we don't quote bid. So d_bid=inf.
                # The transition prob for bid is 0.
                # So the term -p_b * delta_minus is 0.
                
            if q < q_max:
                delta_plus = V_next[i+1] - V_next[i]
            else:
                delta_plus = 0.0

            # Optimize bid
            if q < q_max:
                # Maximize objective_bid(y, delta_minus)
                # Note: if delta_minus is very negative, we want high p (low y? no, high p means low d, high y? 
                # p increases as y increases. d decreases as y increases.
                # If delta_minus > 0 (V decreases as q decreases), we want to avoid bid fills? 
                # Wait. V_next[q-1] is value with lower inventory.
                # If V_next[q-1] < V_next[q], then delta_minus = V_next[q] - V_next[q-1] > 0.
                # Then -p * delta_minus is a cost. We want small p.
                # If V_next[q-1] > V_next[q], delta_minus < 0. -p * delta_minus is a benefit. We want large p.
                
                res = minimize_scalar(lambda y: -objective_bid(y, delta_minus), 
                                      bounds=(y_min, y_max), method='bounded')
                if res.success:
                    y_opt = res.x
                    d_opt = (1.0 / k) * np.log(A_dt / y_opt)
                    if d_opt > -3.0/k:
                        opt_bid[n, i] = d_opt
                    else:
                        opt_bid[n, i] = -3.0/k + 1e-9
                else:
                    opt_bid[n, i] = 1.0/k # Fallback
            else:
                opt_bid[n, i] = np.inf

            # Optimize ask
            if q > q_min:
                res = minimize_scalar(lambda y: -objective_ask(y, delta_plus), 
                                      bounds=(y_min, y_max), method='bounded')
                if res.success:
                    y_opt = res.x
                    d_opt = (1.0 / k) * np.log(A_dt / y_opt)
                    if d_opt > -3.0/k:
                        opt_ask[n, i] = d_opt
                    else:
                        opt_ask[n, i] = -3.0/k + 1e-9
                else:
                    opt_ask[n, i] = 1.0/k # Fallback
            else:
                opt_ask[n, i] = np.inf

            # Calculate V_curr
            p_b = 1.0 - np.exp(-A * np.exp(-k * opt_bid[n, i]) * dt) if q < q_max else 0.0
            p_a = 1.0 - np.exp(-A * np.exp(-k * opt_ask[n, i]) * dt) if q > q_min else 0.0
            
            # Expected value
            # E[V_next] = V_next[q] - p_b * delta_minus + p_a * delta_plus 
            # (Ignoring p_b p_a term for simplicity in V calculation, or including it?)
            # Let's include it for accuracy in V.
            # E[V] = p_b p_a V_next[q] + p_b(1-p_a)V_next[q-1] + (1-p_b)p_a V_next[q+1] + (1-p_b)(1-p_a)V_next[q]
            # This requires V_next[q-1] and V_next[q+1].
            
            v_q = V_next[i]
            v_qm1 = V_next[i-1] if q > q_min else v_q # If no bid, transition to q-1 is 0 prob, but for formula we can just use 0 prob
            v_qp1 = V_next[i+1] if q < q_max else v_q
            
            # If q is boundary, p_b or p_a is 0, so terms vanish.
            # But we need to be careful with indices.
            
            exp_v = 0.0
            if q > q_min and q < q_max:
                exp_v = p_b * p_a * v_q + p_b * (1 - p_a) * v_qm1 + (1 - p_b) * p_a * v_qp1 + (1 - p_b) * (1 - p_a) * v_q
            elif q == q_min:
                # No bid
                exp_v = p_a * v_qp1 + (1 - p_a) * v_q
            elif q == q_max:
                # No ask
                exp_v = p_b * v_qm1 + (1 - p_b) * v_q
            else:
                exp_v = v_q

            # Immediate reward
            # Depth earned: d_b * p_b + d_a * p_a
            # Note: if no quote, d is inf, p is 0. inf*0 is nan. Handle carefully.
            reward_b = opt_bid[n, i] * p_b if q < q_max else 0.0
            reward_a = opt_ask[n, i] * p_a if q > q_min else 0.0
            
            # Inventory penalty
            pen = inv_penalty * q * q
            
            V_curr[i] = reward_b + reward_a - pen + exp_v
        
        V_next = V_curr
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _dp_cache:
        _dp_cache[key] = _compute_dp_tables(sigma, A, k, gamma, T, steps, Q)
    
    opt_bid_table, opt_ask_table = _dp_cache[key]
    
    # Determine step n
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to [0, steps-1]
    if n >= steps: n = steps - 1
    if n < 0: n = 0
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    q_int = inventory.astype(np.int64)
    # Clip to ensure valid index
    q_int = np.clip(q_int, -Q, Q)
    indices = q_int + Q
    
    # Get depths
    d_bid = opt_bid_table[n, indices]
    d_ask = opt_ask_table[n, indices]
    
    # Ensure shape and type
    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)
    
    # Handle any NaNs or invalid values
    d_bid = np.nan_to_num(d_bid, nan=np.inf, posinf=np.inf, neginf=-3.0/k)
    d_ask = np.nan_to_num(d_ask, nan=np.inf, posinf=np.inf, neginf=-3.0/k)
    
    # Final safety clamp
    d_bid = np.maximum(d_bid, -3.0/k + 1e-9)
    d_ask = np.maximum(d_ask, -3.0/k + 1e-9)
    
    return d_bid, d_ask
