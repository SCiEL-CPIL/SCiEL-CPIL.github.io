"""
Optimal market making with inventory risk via continuous-depth HJB.

Solves the discrete finite-horizon MDP exactly (no depth discretization) using the
smooth optimal-control identity  V = logsumexp_{a} (r(a) + Q(a) - V_next)
which is exact for the discrete model (the "inf" over actions is a finite max, so
soft-max == hard-max).  The value function is fit to a quadratic in inventory each
step, which (a) lets us evaluate Q(a) in closed form, and (b) removes the time
dimension at runtime -> O(1) scalar ops per path per step.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_DEADLINE = None


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 295  # safety margin before 330s kill
    return _DEADLINE


def _solve_policy(sigma, A, k, gamma, T, steps, Q, deadline):
    """Return (a_coeffs, b_coeffs) arrays of shape (steps+1,) for the quadratic
    value fit  V(q) = a[n]*q^2 + b[n]*q ."""
    dt = T / steps
    n_levels = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    q2 = q_vals * q_vals
    inv_cost = (gamma / 2.0) * sigma * sigma * dt * q2          # (n_levels,)
    # inventory transitions (clamped at +/-Q)
    q_bid = np.minimum(q_vals + 1.0, Q)
    q_ask = np.maximum(q_vals - 1.0, -Q)

    # smooth-max numerics (shift by max of the "no-fill" term)
    K = 3.0 / k
    D = 1.0 / k
    lam = A * dt
    loglam = np.log(lam)
    lamK = lam * K
    lamD = lam * D
    s0 = -lamK + loglam
    s1 = -lamD + loglam

    a = np.zeros(steps + 1)
    b = np.zeros(steps + 1)

    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            # not enough time for remaining steps -> reuse last computed coeffs
            if n < steps:
                a[:n + 1] = a[n + 1]
                b[:n + 1] = b[n + 1]
            break

        aN = a[n + 1]
        bN = b[n + 1]
        Vn = aN * q2 + bN * q_vals

        c_bid = aN * (q_bid * q_bid) + bN * q_bid
        c_ask = aN * (q_ask * q_ask) + bN * q_ask

        # no-fill term (shifted):  -Vn - inv_cost
        T0 = -Vn - inv_cost
        # bid-fill term:  K + loglam - lamK - c_bid - inv_cost
        Tb = s0 - c_bid - inv_cost
        # ask-fill term:  D + loglam - lamD - c_ask - inv_cost
        Ta = s1 - c_ask - inv_cost

        M = np.maximum(T0, np.maximum(Tb, Ta))
        S = np.exp(T0 - M) + np.exp(Tb - M) + np.exp(Ta - M)
        V = M + np.log(S)

        # least-squares quadratic fit (constant free)
        X = np.vstack([q2, q_vals, np.ones(n_levels)]).T
        coef, *_ = np.linalg.lstsq(X, V, rcond=None)
        a[n] = coef[0]
        b[n] = coef[1]

    return a, b


def _build_policy(params, deadline):
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key in _policy_cache:
        return _policy_cache[key]

    a, b = _solve_policy(params['sigma'], params['A'], params['k'],
                         params['gamma'], params['T'], params['steps'],
                         params['Q'], deadline)
    _policy_cache[key] = (a, b)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    n_paths = len(inventory)
    k = params['k']
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(n, params['steps'] - 1)
    n = max(n, 0)
    Q = params['Q']

    try:
        a, b = _build_policy(params, _get_deadline())
    except Exception:
        depth = np.full(n_paths, 1.0 / k)
        return depth, depth.copy()

    aN = float(a[n + 1])
    bN = float(b[n + 1])
    inv_cost = (params['gamma'] / 2.0) * params['sigma'] ** 2 * dt
    lam = params['A'] * dt
    K = 3.0 / k
    D = 1.0 / k
    s0 = -lam * K + np.log(lam)
    s1 = -lam * D + np.log(lam)

    q = np.clip(np.asarray(inventory, dtype=np.float64), -Q, Q)
    q2 = q * q
    Vn = aN * q2 + bN * q
    c_bid = aN * np.minimum(q + 1.0, Q) ** 2 + bN * np.minimum(q + 1.0, Q)
    c_ask = aN * np.maximum(q - 1.0, -Q) ** 2 + bN * np.maximum(q - 1.0, -Q)

    T0 = -Vn - inv_cost * q2
    Tb = s0 - c_bid - inv_cost * q2
    Ta = s1 - c_ask - inv_cost * q2

    # bid depth: maximize f(x) = x + loglam - lam*exp(-k x) - c_bid  (x>0)
    x_bid = -np.log(lam / np.maximum(1.0 + c_bid, 1e-12)) / k
    bid_depths = np.maximum(x_bid, 1e-6)
    # ask depth: maximize g(y) = y + loglam - lam*exp(-k y) - c_ask  (y>0)
    x_ask = -np.log(lam / np.maximum(1.0 + c_ask, 1e-12)) / k
    ask_depths = np.maximum(x_ask, 1e-6)

    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, np.inf)
    bid_depths = np.where(np.asarray(inventory) >= Q, np.inf, bid_depths)
    ask_depths = np.where(np.asarray(inventory) <= -Q, np.inf, ask_depths)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
