"""Uncertainty bound minimization via analytic gradient + L-BFGS-B with multi-start and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pp * I[idx]

    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Analytic gradient
    da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])

    # dp_k/da_c for c=0,1,2 (a0, a4, a8)
    # p0 = a0 + 3a4 + 105a8 + 945a12
    # p1 = -6a4 - 420a8 - 7560a12
    # p2 = a4 + 210a8 + 13860a12
    # p3 = -28a8 - 6930a12
    # p4 = a8 + 1155a12
    # p5 = -66a12
    # p6 = a12

    # dp/da0: [1, 0, 0, 0, 0, 0, 0] + 945*(-1/945) for p0, etc.
    # Actually: dp_k/da_c = base_kc + coeff_k12 * da12_c
    # base for p0: [1, 3, 105]
    # base for p1: [0, -6, -420]
    # base for p2: [0, 1, 210]
    # base for p3: [0, 0, -28]
    # base for p4: [0, 0, 1]
    # base for p5: [0, 0, 0]
    # base for p6: [0, 0, 0]
    # coeff_k12: [945, -7560, 13860, -6930, 1155, -66, 1]

    base = np.array([
        [1.0, 3.0, 105.0],
        [0.0, -6.0, -420.0],
        [0.0, 1.0, 210.0],
        [0.0, 0.0, -28.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    coeff12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

    grad = np.zeros(3)
    for c in range(3):
        dp = base[:, c] + coeff12 * da12[c]

        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * I[idx]
                dm[k] += 2.0 * dpi * s

        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * I[idx]
                dn[k] += 2.0 * 2.0 * np.pi * dqi * s

        dvm2 = (dm[1] * m0 - m2 * dm[0]) / (m0 * m0)
        dvn2 = (dn[1] * n0 - n2 * dn[0]) / (n0 * n0)

        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses analytic gradient with L-BFGS-B optimization from multiple starting points,
    followed by Nelder-Mead polish.
    """
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _compute_bound(*seed)

    best_bound = seed_bound
    best_params = seed.copy()

    # Multiple starting points
    starts = [
        seed.copy(),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.06, 0.0004]),
        np.array([-1.2, 0.03, 0.0002]),
    ]

    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))

    # Phase 1: L-BFGS-B from multiple starting points
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                           jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead polish from the best point
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18})
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: A few more L-BFGS-B runs from points near the best
    if time.time() < _DEADLINE - 10:
        for i in range(5):
            if time.time() > _DEADLINE - 5:
                break
            perturbed = best_params + rng.normal(0, 0.05, 3)
            try:
                res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                               jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
                if np.isfinite(res.fun) and res.fun < best_bound:
                    best_bound = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Verify the result
    final_bound = _compute_bound(*best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()

    return best_params
# EVOLVE-BLOCK-END
