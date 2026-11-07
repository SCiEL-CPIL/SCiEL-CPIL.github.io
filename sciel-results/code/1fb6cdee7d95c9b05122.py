import numpy as np
import time
from scipy.special import lambertw

_start_time = None
_dp_cache = {}


def _solve_depth_newton(d0, A, k, dt, lam0):
    """
    Solve for d given initial guess d0, with fallback.
    """
    if not np.isfinite(d0):
        return d0
    d = max(0.0, d0)
    for _ in range(20):
        if d > 50.0 / k:
            break
        lam = A * np.exp(-k * d)
        f = lam - 1.0 + np.exp(-(lam - lam0) * dt)
        f_prime = -k * lam * (1.0 - np.exp(-(lam - lam0) * dt))
        if abs(f_prime) < 1e-12:
            break
        d_new = d - f / f_prime
        if d_new < 0.0:
            d_new = 0.0
        if abs(d_new - d) < 1e-9:
            d = d_new
            break
        d = d_new
    return d


def _get_dp_tables(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _dp_cache:
        return _dp_cache[key]

    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    lam0 = A * dt

    V_next = np.zeros(n_states, dtype=np.float64)
    bid_depths = np.zeros((steps, n_states), dtype=np.float64)
    ask_depths = np.zeros((steps, n_states), dtype=np.float64)

    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states, dtype=np.float64)
        for i, q in enumerate(q_values):
            cost = 0.5 * gamma * sigma * sigma * q * q * dt

            if q == Q:
                d_b = np.inf
                delta_minus = V_next[i - 1] - V_next[i]
                d_a = max(0.0, (1.0 / k) - delta_minus)
                if np.isinf(d_a):
                    p_a = 0.0
                else:
                    lam_a = A * np.exp(-k * d_a)
                    p_a = 1.0 - np.exp(-lam_a * dt)
                    d_a = _solve_depth_newton(d_a, A, k, dt, lam0)
                    lam_a = A * np.exp(-k * d_a)
                    p_a = 1.0 - np.exp(-lam_a * dt)
                exp_V = p_a * V_next[i - 1] + (1.0 - p_a) * V_next[i]
                reward_b = 0.0
                reward_a = 0.0 if np.isinf(d_a) else d_a * p_a

            elif q == -Q:
                d_a = np.inf
                delta_plus = V_next[i + 1] - V_next[i]
                d_b = max(0.0, (1.0 / k) - delta_plus)
                if np.isinf(d_b):
                    p_b = 0.0
                else:
                    lam_b = A * np.exp(-k * d_b)
                    p_b = 1.0 - np.exp(-lam_b * dt)
                    d_b = _solve_depth_newton(d_b, A, k, dt, lam0)
                    lam_b = A * np.exp(-k * d_b)
                    p_b = 1.0 - np.exp(-lam_b * dt)
                exp_V = p_b * V_next[i + 1] + (1.0 - p_b) * V_next[i]
                reward_b = 0.0 if np.isinf(d_b) else d_b * p_b
                reward_a = 0.0

            else:
                delta_plus = V_next[i + 1] - V_next[i]
                delta_minus = V_next[i - 1] - V_next[i]
                d_b = max(0.0, (1.0 / k) - delta_plus)
                d_a = max(0.0, (1.0 / k) - delta_minus)
                if np.isinf(d_b):
                    p_b = 0.0
                else:
                    lam_b = A * np.exp(-k * d_b)
                    p_b = 1.0 - np.exp(-lam_b * dt)
                    d_b = _solve_depth_newton(d_b, A, k, dt, lam0)
                    lam_b = A * np.exp(-k * d_b)
                    p_b = 1.0 - np.exp(-lam_b * dt)
                if np.isinf(d_a):
                    p_a = 0.0
                else:
                    lam_a = A * np.exp(-k * d_a)
                    p_a = 1.0 - np.exp(-lam_a * dt)
                    d_a = _solve_depth_newton(d_a, A, k, dt, lam0)
                    lam_a = A * np.exp(-k * d_a)
                    p_a = 1.0 - np.exp(-lam_a * dt)

                p_b_fill_a_no = p_b * (1.0 - p_a)
                p_a_fill_b_no = (1.0 - p_b) * p_a
                p_no_fill = (1.0 - p_b) * (1.0 - p_a)
                p_both_fill = p_b * p_a
                exp_V = p_b_fill_a_no * V_next[i + 1] + \
                        p_a_fill_b_no * V_next[i - 1] + \
                        (p_no_fill + p_both_fill) * V_next[i]
                reward_b = 0.0 if np.isinf(d_b) else d_b * p_b
                reward_a = 0.0 if np.isinf(d_a) else d_a * p_a

            bid_depths[n, i] = d_b
            ask_depths[n, i] = d_a
            V_curr[i] = -cost + reward_b + reward_a + exp_V

        V_next = V_curr

    _dp_cache[key] = (bid_depths, ask_depths)
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    if time.time() - _start_time > 295:
        k = params['k']
        base = 1.0 / k
        return np.full(len(inventory), base), np.full(len(inventory), base)

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    n = int(t / (T / steps))
    if n >= steps:
        n = steps - 1

    bid_depths_table, ask_depths_table = _get_dp_tables(sigma, A, k, gamma, T, steps, Q)

    q = np.asarray(inventory, dtype=np.int32)
    idx = q + Q

    bid_d = bid_depths_table[n, idx].astype(np.float64)
    ask_d = ask_depths_table[n, idx].astype(np.float64)

    min_depth = -3.0 / k + 0.1
    max_depth = 100.0 / k

    finite_mask_bid = np.isfinite(bid_d)
    bid_d[finite_mask_bid] = np.clip(bid_d[finite_mask_bid], min_depth, max_depth)

    finite_mask_ask = np.isfinite(ask_d)
    ask_d[finite_mask_ask] = np.clip(ask_d[finite_mask_ask], min_depth, max_depth)

    bid_d[np.isnan(bid_d)] = min_depth
    ask_d[np.isnan(ask_d)] = min_depth

    return bid_d, ask_d
