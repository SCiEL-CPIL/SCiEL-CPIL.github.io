# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200.0

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n-1)!! sqrt(pi) / 2^n
_sp = np.sqrt(np.pi)
_I = np.zeros(14)
_I[0] = _sp
for _n in range(1, 14):
    _I[_n] = _I[_n - 1] * (2 * _n - 1) / 2.0

# Precompute the 3x7x7 tensor M[k,i,j] = I[k+i+j]
_M = np.zeros((3, 7, 7))
for _k in range(3):
    for _i in range(7):
        for _j in range(7):
            _idx = _k + _i + _j
            if _idx < 14:
                _M[_k, _i, _j] = _I[_idx]

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_PI2 = 2.0 * np.pi

# Precompute dp/da_c matrix: dp[coeff, param] for each p_k
# p0 = a0 + 3a4 + 105a8 + 945a12, a12 = -(a0+3a4+105a8)/945
# da12/da0 = -1/945, da12/da4 = -3/945, da12/da8 = -105/945
_da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# dp_k/da_c = direct_coeff[c] + coeff_of_a12 * da12[c]
# p0: direct = [1, 3, 105], a12_coeff = 945
# p1: direct = [0, -6, -420], a12_coeff = -7560
# p2: direct = [0, 1, 210], a12_coeff = 13860
# p3: direct = [0, 0, -28], a12_coeff = -6930
# p4: direct = [0, 0, 1], a12_coeff = 1155
# p5: direct = [0, 0, 0], a12_coeff = -66
# p6: direct = [0, 0, 0], a12_coeff = 1
_direct = np.array([
    [1.0, 3.0, 105.0],
    [0.0, -6.0, -420.0],
    [0.0, 1.0, 210.0],
    [0.0, 0.0, -28.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
])
_a12_coeff = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

# dp[k, c] = _direct[k, c] + _a12_coeff[k] * _da12[c]
_dp = _direct + np.outer(_a12_coeff, _da12)  # shape (7, 3)

# Precompute M @ I vectors for fast gradient: for each k in 0..2,
# S[k, i] = sum_j p_j I[k+i+j] -- but this depends on p, so compute on fly
# Instead precompute _M2[k] = _M[k] as (7,7) matrix


def _compute_bound_and_grad(a0, a4, a8):
    """Compute the uncertainty bound and its exact analytic gradient."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # m_k = sum_{i,j} p_i p_j I[k+i+j] = p^T M[k] p
    m = np.einsum('i,kij,j->k', p, _M, p)

    q = p * _SIGNS
    n = np.einsum('i,kij,j->k', q, _M, q) * _PI2

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.zeros(3)

    bound = np.sqrt(vf * vF)

    # Exact analytic gradient using chain rule
    # d(m_k)/da_c = 2 * sum_i dp_i * sum_j p_j I[k+i+j] = 2 * dp^T M[k] p
    # d(n_k)/da_c = 2 * PI2 * sum_i dq_i * sum_j q_j I[k+i+j] = 2 * PI2 * dq^T M[k] q
    # where dq = dp * SIGNS

    # Precompute M[k] @ p and M[k] @ q for each k
    Mp = np.array([np.dot(_M[k], p) for k in range(3)])  # (3, 7)
    Mq = np.array([np.dot(_M[k], q) for k in range(3)])  # (3, 7)

    grad_bound = np.zeros(3)
    for c in range(3):
        dpc = _dp[:, c]  # (7,)
        dqc = dpc * _SIGNS  # (7,)

        # dm_k/da_c = 2 * dpc^T M[k] p
        dm = 2.0 * np.array([np.dot(dpc, Mp[k]) for k in range(3)])
        # dn_k/da_c = 2 * PI2 * dqc^T M[k] q
        dn = 2.0 * _PI2 * np.array([np.dot(dqc, Mq[k]) for k in range(3)])

        dm0, dm2 = dm[0], dm[1]
        dn0, dn2 = dn[0], dn[1]

        dvf = (m0 * dm2 - m2 * dm0) / (m0 * m0)
        dvF = (n0 * dn2 - n2 * dn0) / (n0 * n0)

        grad_bound[c] = 0.5 / bound * (vF * dvf + vf * dvF)

    return bound, grad_bound


def _objective(x):
    b, _ = _compute_bound_and_grad(x[0], x[1], x[2])
    return b


def _objective_grad(x):
    b, g = _compute_bound_and_grad(x[0], x[1], x[2])
    return b, g


def construct():
    seed = np.array([-1.13, 0.04, 0.000308642])
    seed_bound = _objective(seed)

    rng = np.random.default_rng(42)

    best_bound = seed_bound
    best_params = seed.copy()

    # Build diverse starting points
    starts = [seed.copy()]

    # Structured grid around seed
    for a0 in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        for a4 in [-0.1, -0.05, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1]:
            for a8 in [-0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002]:
                starts.append(np.array([a0, a4, a8]))

    # Log-scale random starts
    for _ in range(20):
        a0 = rng.uniform(-5, 5)
        a4 = rng.uniform(-0.2, 0.2)
        a8 = rng.uniform(-0.01, 0.01)
        starts.append(np.array([a0, a4, a8]))

    # Gaussian random starts
    for _ in range(20):
        starts.append(rng.normal(0, 1.0, 3))

    # Run L-BFGS-B from each start
    for x0 in starts:
        if time.time() > _DEADLINE - 25:
            break
        try:
            res = minimize(_objective_grad, x0, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # High-precision polish with Nelder-Mead
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final L-BFGS-B polish from the best point with very tight tolerance
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if np.isfinite(res.fun) and res.fun < best_bound:
                best_bound = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final verification
    final_bound = _objective(best_params)
    if not np.isfinite(final_bound) or final_bound > seed_bound:
        return seed.copy()
    return best_params
# EVOLVE-BLOCK-END
