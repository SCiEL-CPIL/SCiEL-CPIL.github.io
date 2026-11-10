"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _solve_depth_foc(delta, A_dt, k, depth_min=1e-4, depth_max=20.0):
    """
    Solve the first-order condition for optimal depth d > 0:
    maximize f(d) = p(d) * (d + delta)
    where p(d) = 1 - exp(-A_dt * exp(-k*d))
    
    FOC: f'(d) = p'(d)*(d+delta) + p(d) = 0
    p'(d) = A_dt * k * exp(-k*d) * exp(-A_dt*exp(-k*d))
    
    So: A_dt*k*exp(-k*d)*exp(-A_dt*exp(-k*d))*(d+delta) + 1 - exp(-A_dt*exp(-k*d)) = 0
    
    Let u = exp(-k*d), so d = -ln(u)/k
    g(u) = A_dt*k*u*exp(-A_dt*u)*(-ln(u)/k + delta) + 1 - exp(-A_dt*u) = 0
    g(u) = A_dt*u*exp(-A_dt*u)*(-ln(u) + k*delta) + 1 - exp(-A_dt*u) = 0
    
    Newton on u in (0, 1]:
    g'(u) = A_dt * [exp(-A_dt*u)*(-ln(u)+k*delta) + u*(-A_dt*exp(-A_dt*u))*(-1/u) 
                   + u*exp(-A_dt*u)*(-1/u)]
         = A_dt * [exp(-A_dt*u)*(-ln(u)+k*delta) - A_dt*u*exp(-A_dt*u) - u*exp(-A_dt*u)]
         = A_dt * exp(-A_dt*u) * [-ln(u) + k*delta - A_dt*u - u]
    """
    # Handle edge case: if delta is very negative, optimal d may be at depth_min
    # If delta is very positive, optimal d may be at depth_max
    
    # Initial guess from small-dt approximation: d* ≈ max(-1/k - delta, depth_min)
    d0 = -1.0/k - delta
    if d0 < depth_min:
        d0 = depth_min
    if d0 > depth_max:
        d0 = depth_max
    
    # Convert to u
    u = np.exp(-k * d0)
    
    # Newton iterations
    for _ in range(30):
        # Compute exp(-A_dt * u)
        e = np.exp(-A_dt * u)
        
        # g(u) = A_dt * u * e * (-ln(u) + k*delta) + 1 - e
        ln_u = np.log(max(u, 1e-300))
        g = A_dt * u * e * (-ln_u + k * delta) + 1.0 - e
        
        # g'(u) = A_dt * e * (-ln(u) + k*delta - A_dt*u - u)
        gp = A_dt * e * (-ln_u + k * delta - A_dt * u - u)
        
        if abs(gp) < 1e-30:
            break
        
        u_new = u - g / gp
        
        # Keep u in valid range (0, 1]
        if u_new <= 0:
            u_new = u * 0.5
        if u_new > 1.0:
            u_new = 1.0
        
        if abs(u_new - u) < 1e-12:
            u = u_new
            break
        u = u_new
    
    d = -np.log(max(u, 1e-300)) / k
    d = np.clip(d, depth_min, depth_max)
    return d


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via DP over (time, inventory) states.
    Uses Newton-Raphson to solve the continuous first-order condition for depth.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    depth_min = 1e-4
    depth_max = 20.0 / k

    A_dt = A * dt

    # V_next[q_idx] where q_idx = q + Q, q in [-Q, Q]
    V_next = np.zeros(2 * Q + 1)

    # Store optimal depths: shape (steps, 2*Q+1)
    opt_bid = np.full((steps, 2 * Q + 1), np.inf)
    opt_ask = np.full((steps, 2 * Q + 1), np.inf)

    q_vals = np.arange(-Q, Q + 1)

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 3.0:
            # Fill remaining steps with current best estimate
            for nn in range(n - 1, -1, -1):
                opt_bid[nn] = opt_bid[n]
                opt_ask[nn] = opt_ask[n]
            break

        V_curr = np.zeros(2 * Q + 1)
        penalty = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            pen = penalty[q_idx]

            if q == Q:
                # Only ask quote allowed
                delta = V_next[q_idx - 1] - V_next[q_idx]
                d_opt = _solve_depth_foc(delta, A_dt, k, depth_min, depth_max)
                
                best_val = (1.0 - np.exp(-A_dt * np.exp(-k * d_opt))) * (d_opt + delta)
                V_curr[q_idx] = best_val - pen + V_next[q_idx]
                opt_ask[n, q_idx] = d_opt
                opt_bid[n, q_idx] = np.inf

            elif q == -Q:
                # Only bid quote allowed
                delta = V_next[q_idx + 1] - V_next[q_idx]
                d_opt = _solve_depth_foc(delta, A_dt, k, depth_min, depth_max)
                
                best_val = (1.0 - np.exp(-A_dt * np.exp(-k * d_opt))) * (d_opt + delta)
                V_curr[q_idx] = best_val - pen + V_next[q_idx]
                opt_bid[n, q_idx] = d_opt
                opt_ask[n, q_idx] = np.inf

            else:
                # Both sides allowed
                delta_plus = V_next[q_idx + 1] - V_next[q_idx]
                delta_minus = V_next[q_idx - 1] - V_next[q_idx]
                
                # Optimize bid: maximize p(d) * (d + delta_plus)
                d_b = _solve_depth_foc(delta_plus, A_dt, k, depth_min, depth_max)
                val_b = (1.0 - np.exp(-A_dt * np.exp(-k * d_b))) * (d_b + delta_plus)
                
                # Optimize ask: maximize p(d) * (d + delta_minus)
                d_a = _solve_depth_foc(delta_minus, A_dt, k, depth_min, depth_max)
                val_a = (1.0 - np.exp(-A_dt * np.exp(-k * d_a))) * (d_a + delta_minus)
                
                best_val = val_b + val_a - pen + V_next[q_idx]
                
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = d_b
                opt_ask[n, q_idx] = d_a

        V_next = V_curr

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _analytical_quote(t, inventory, mid, params):
    """
    Avellaneda-Stoikov style analytical quote.
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
    
    base = 1.0 / k
    skew_coeff = gamma * sigma**2 * remaining / 2.0
    c = skew_coeff / (2.0 * k)
    
    bid_depths = base + c * inventory
    ask_depths = base - c * inventory
    
    bid_depths = np.clip(bid_depths, 1e-4, 20.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 20.0 / k)
    
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
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        bid_depths = np.asarray(bid_depths, dtype=np.float64)
        ask_depths = np.asarray(ask_depths, dtype=np.float64)
        return bid_depths, ask_depths
    else:
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
