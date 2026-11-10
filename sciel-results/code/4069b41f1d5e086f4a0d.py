"""Optimal market making with inventory risk - Continuous Newton-Raphson Solver.

Solves the first-order conditions for optimal bid/ask spreads directly,
eliminating grid discretization error.
"""
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill


def _fill_prob_and_deriv(depth, A, k, dt):
    """Return (p, p_prime) for fill probability and its derivative w.r.t. depth.
    
    p(b) = 1 - exp(-A * exp(-k*b) * dt)
    p'(b) = -A * k * dt * exp(-k*b) * (1 - p(b))
    """
    if depth <= 0:
        return 0.0, 0.0
    e = np.exp(-k * depth)
    inner = A * e * dt
    one_minus_p = np.exp(-inner)  # = 1 - p
    p = 1.0 - one_minus_p
    p_prime = -A * k * dt * e * one_minus_p
    return p, p_prime


def _solve_foc_bid(K, A, k, dt):
    """Solve p(b)*(b+K) + p'(b) = 0 for b using Newton-Raphson.
    
    The objective to maximize is f(b) = p(b)*(b+K).
    f'(b) = p'(b)*(b+K) + p(b) = 0
    f''(b) = p''(b)*(b+K) + 2*p'(b)
    
    We use Newton's method on f'(b) = 0.
    """
    # Initial guess
    b = 0.5
    
    for _ in range(50):
        p, p_prime = _fill_prob_and_deriv(b, A, k, dt)
        
        # f'(b) = p_prime * (b + K) + p
        f_prime = p_prime * (b + K) + p
        
        # f''(b) = p''(b) * (b + K) + 2 * p_prime
        # p''(b) = derivative of -A*k*dt*exp(-k*b)*(1-p)
        # p''(b) = A*k^2*dt*exp(-k*b)*(1-p) + A*k*dt*exp(-k*b)*(-p')
        #        = A*k^2*dt*exp(-k*b)*(1-p) - A*k*dt*exp(-k*b)*p_prime
        # Since p_prime = -A*k*dt*exp(-k*b)*(1-p), we have:
        # p''(b) = -k * p_prime + A*k*dt*exp(-k*b)*p_prime
        #        = p_prime * (-k + A*k*dt*exp(-k*b))
        # But A*dt*exp(-k*b) = inner, so:
        # p''(b) = p_prime * (-k + inner)
        
        e = np.exp(-k * b)
        inner = A * e * dt
        p_double_prime = p_prime * (-k + inner)
        
        f_double_prime = p_double_prime * (b + K) + 2 * p_prime
        
        if abs(f_double_prime) < 1e-15:
            break
            
        delta = f_prime / f_double_prime
        b_new = b - delta
        
        # Ensure b stays positive
        if b_new <= 0:
            b_new = b * 0.5
            if b_new <= 0:
                b_new = 1e-6
        
        if abs(b_new - b) < 1e-10:
            b = b_new
            break
            
        b = b_new
    
    return max(b, 0.0)


def _solve_foc_ask(L, A, k, dt):
    """Solve p(a)*(a+L) + p'(a) = 0 for a using Newton-Raphson."""
    return _solve_foc_bid(L, A, k, dt)


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes optimal bid/ask depths via dynamic programming with
    continuous Newton-Raphson optimization.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1
    
    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # V[n, q_idx]
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    start_time = time.time()
    deadline = start_time + (_TIME_BUDGET - (time.time() - _GLOBAL_START))
    
    # Initial guesses
    b_cur = np.full(n_q, 0.5)
    a_cur = np.full(n_q, 0.5)
    
    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break
        
        V_next = V[n + 1]
        
        # Transition inventory offsets
        V_q_plus = np.zeros(n_q)
        V_q_minus = np.zeros(n_q)
        V_q = V_next.copy()
        
        V_q_plus[:n_q - 1] = V_next[1:]
        V_q_minus[1:] = V_next[:-1]
        
        # Coordinate descent iterations with Newton-Raphson
        max_iters = 10
        tol = 1e-8
        
        for _ in range(max_iters):
            b_old = b_cur.copy()
            a_old = a_cur.copy()
            
            # 1. Optimize bid (b) given ask (a)
            exp_a = np.exp(-k * a_cur)
            p_a = 1.0 - np.exp(-A * exp_a * dt)
            
            # Coefficient K_q for bid optimization
            K = (1.0 - p_a) * (V_q_plus - V_q) - p_a * V_q_minus
            
            # Solve FOC for each q
            b_new = np.zeros(n_q)
            for qi in range(n_q):
                if qi == n_q - 1:  # q = Q, no bid fill
                    b_new[qi] = 0.0
                else:
                    b_new[qi] = _solve_foc_bid(K[qi], A, k, dt)
            
            # 2. Optimize ask (a) given bid (b_new)
            exp_b = np.exp(-k * b_new)
            p_b = 1.0 - np.exp(-A * exp_b * dt)
            
            # Coefficient L_q for ask optimization
            L = (1.0 - p_b) * (V_q_minus - V_q) - p_b * V_q_plus
            
            a_new = np.zeros(n_q)
            for qi in range(n_q):
                if qi == 0:  # q = -Q, no ask fill
                    a_new[qi] = 0.0
                else:
                    a_new[qi] = _solve_foc_ask(L[qi], A, k, dt)
            
            # Check convergence
            diff_b = np.max(np.abs(b_new - b_old))
            diff_a = np.max(np.abs(a_new - a_old))
            
            b_cur = b_new
            a_cur = a_new
            
            if diff_b < tol and diff_a < tol:
                break
        
        # Compute final value for this step
        exp_b_final = np.exp(-k * b_cur)
        p_b_final = 1.0 - np.exp(-A * exp_b_final * dt)
        
        exp_a_final = np.exp(-k * a_cur)
        p_a_final = 1.0 - np.exp(-A * exp_a_final * dt)
        
        # Calculate V[n, q]
        trans_plus = p_b_final * (1.0 - p_a_final) * V_q_plus
        trans_minus = p_a_final * (1.0 - p_b_final) * V_q_minus
        trans_same = (1.0 - p_b_final - p_a_final + p_b_final * p_a_final) * V_q
        
        V_n = b_cur * p_b_final + a_cur * p_a_final - C_inv + trans_plus + trans_minus + trans_same
        
        V[n, :] = V_n
        opt_bid[n, :] = b_cur
        opt_ask[n, :] = a_cur
    
    return opt_bid, opt_ask


def _get_policy(params):
    """Retrieves or computes the optimal policy for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Returns optimal bid and ask depths based on pre-computed DP policy."""
    opt_bid, opt_ask = _get_policy(params)

    dt = params['T'] / params['steps']
    n = int(round(t / dt))

    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1

    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]

    # Ensure valid outputs: must be >= 0 and finite
    bid_depths = np.where(np.isnan(bid_depths) | ~np.isfinite(bid_depths), 0.01, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | ~np.isfinite(ask_depths), 0.01, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.01, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.01, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
