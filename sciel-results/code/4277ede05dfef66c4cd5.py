"""Uncertainty bound minimization: analytic gradient + L-BFGS-B with sphere-reparameterized local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200.0


def _moments(p):
    """
    Given polynomial coefficients p[0..6] (P(x) = sum p_k x^{2k}),
    compute m[0..2] = integral x^{2k} |P(x)|^2 exp(-x^2) dx
    and n[0..2] = integral xi^{2k} |Q(xi)|^2 exp(-xi^2) dx * 2*pi
    where Q(xi) = sum p_k (-1)^k xi^{2k}.
    """
    sp = np.sqrt(np.pi)
    # I[n] = integral x^{2n} exp(-x^2) dx
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

    return m, n


def _bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(<x^2>/<1> * <xi^2>/<1>) and its gradient
    with respect to (a0, a4, a8).
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    m, n = _moments(p)
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Gradient computation
    # da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

    # Precompute I for gradient
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for nn in range(1, 14):
        I[nn] = I[nn - 1] * (2 * nn - 1) / 2.0

    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    grad = np.zeros(3)
    for c in range(3):
        # dp_k/da_c
        dp = np.zeros(7)
        dp[0] = np.array([1.0, 3.0, 105.0])[c] + 945.0 * da12[c]
        dp[1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0 * da12[c]
        dp[2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0 * da12[c]
        dp[3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0 * da12[c]
        dp[4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0 * da12[c]
        dp[5] = np.array([0.0, 0.0, 0.0])[c] - 66.0 * da12[c]
        dp[6] = np.array([0.0, 0.0, 0.0])[c] + 1.0 * da12[c]

        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * I[idx]
                dm[k] += 2.0 * dpi * s

        # dq_k/da_c = dp_k * (-1)^k
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

        # dn_k/da_c = 2 * 2*pi * sum_i dq_i * sum_j q_j I[k+i+j]
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
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


def _bound_from_normalized(c0, c4, c8, scale):
    """Compute bound from normalized direction and scale."""
    a0 = scale * c0
    a4 = scale * c4
    a8 = scale * c8
    val, _ = _bound_and_grad(np.array([a0, a4, a8]))
    return val


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses analytic gradient + L-BFGS-B from multiple starts, then polishes
    with sphere-reparameterized local search.
    """
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound, _ = _bound_and_grad(seed)

    best_bound = seed_bound
    best_params = seed.copy()

    rng = np.random.default_rng(42)

    # Phase 1: L-BFGS-B from diverse starting points
    starts = [
        seed.copy(),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0004]),
        np.array([-2.5, 0.12, 0.0012]),
    ]

    # Add random starts
    for i in range(15):
        starts.append(rng.normal(0, 0.8, 3))

    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_bound and np.isfinite(res.fun):
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Polish with Nelder-Mead from the best
    if time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_bound and np.isfinite(res.fun):
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Sphere-reparameterized local search for fine-tuning
    r_best = np.linalg.norm(best_params)
    if r_best < 1e-15:
        r_best = 1.0

    dir_best = best_params / r_best
    log_scale_best = np.log(r_best)

    state = np.array([dir_best[0], dir_best[1], dir_best[2], log_scale_best])
    best_state = state.copy()
    best_val = best_bound

    steps = np.array([0.01, 0.01, 0.01, 0.01])
    temperature = 0.01
    cooling = 0.9999
    max_iters = 20000
    iter_count = 0

    while iter_count < max_iters:
        if time.time() > _DEADLINE - 15:
            break

        delta = rng.normal(0, 1, 4) * steps
        new_state = state + delta

        dir_part = new_state[:3]
        dir_norm = np.linalg.norm(dir_part)
        if dir_norm < 1e-15:
            dir_part = np.array([1.0, 0.0, 0.0])
        else:
            dir_part = dir_part / dir_norm

        new_state[3] = np.clip(new_state[3], -3.0, 3.0)
        new_state[:3] = dir_part

        scale = np.exp(new_state[3])
        new_val = _bound_from_normalized(new_state[0], new_state[1], new_state[2], scale)

        if not np.isfinite(new_val):
            new_val = 1e18

        if new_val < best_val:
            state = new_state
            best_val = new_val
            best_state = new_state.copy()
            steps *= 0.95
        else:
            if new_val - best_val < 1e-12 or np.exp(-(new_val - best_val) / temperature) > rng.random():
                state = new_state
                steps *= 1.01
            else:
                steps *= 1.03

        temperature *= cooling
        if temperature < 1e-10:
            temperature = 1e-10

        iter_count += 1

    # Convert back to (a0, a4, a8)
    dir_final = best_state[:3]
    scale_final = np.exp(best_state[3])
    result = np.array([
        scale_final * dir_final[0],
        scale_final * dir_final[1],
        scale_final * dir_final[2]
    ])

    # Verify
    final_bound, _ = _bound_and_grad(result)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return best_params.copy()

    return result
# EVOLVE-BLOCK-END
