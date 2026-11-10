# EVOLVE-BLOCK-START
import time
import numpy as np

_K = 10
_NTH = 70
_NPHI = 130
_DEADLINE = 540.0  # safety margin under 630s wall-clock


def _build_grid():
    # Gauss-Legendre nodes/weights in u = cos(theta), u in [-1,1]
    u, w = np.polynomial.legendre.leggauss(_NTH)
    # theta in [0, pi]; sin(theta) = sqrt(1 - u^2)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    # weights for surface measure dOmega = sin(theta) dtheta dphi = -du dphi
    wth = w * sin_t
    wphi = np.full(_NPHI, 2.0 * np.pi / _NPHI)
    return u, wth, phi, wphi


def _assoc_legendre(k, m, u):
    # P_k^m(u) with Condon-Shortley phase, m >= 0, k >= m
    if m < 0:
        m = -m
    # P_0^0 = 1, P_1^0 = u
    if k == 0:
        return np.ones_like(u)
    if k == 1:
        return u.copy()
    # P_1^1 = -sqrt(1-u^2)
    if k == 1 and m == 1:
        return -np.sqrt(np.maximum(0.0, 1.0 - u * u))
    # upward recurrence for P_k^m
    # P_k^m = ((2k-1) u P_{k-1}^m - (k+m-1) P_{k-2}^m) / (k-m)
    P = np.empty(k + 1, dtype=object)
    P[0] = None
    # start at j = m
    if m == 0:
        P[0] = np.ones_like(u)
        P[1] = u.copy()
        jstart = 1
    elif m == 1:
        P[1] = -np.sqrt(np.maximum(0.0, 1.0 - u * u))
        jstart = 1
    else:
        # P_m^m = (-1)^m (2m-1)!! (1-u^2)^{m/2}
        coeff = (-1.0) ** m
        for i in range(1, 2 * m, 2):
            coeff *= i
        P[m] = coeff * (np.maximum(0.0, 1.0 - u * u) ** (0.5 * m))
        jstart = m
    for j in range(jstart + 1, k + 1):
        P[j] = ((2 * j - 1) * u * P[j - 1] - (j + m - 1) * P[j - 2]) / (j - m)
    return P[k]


def _build_basis(u):
    # Returns B of shape (21, NTH); column index i corresponds to m = i - 10
    k = _K
    B = np.empty((2 * k + 1, _NTH), dtype=np.float64)
    for m in range(0, k + 1):
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi) *
                    float(np.math.factorial(k - m)) / float(np.math.factorial(k + m)))
        Pm = _assoc_legendre(k, m, u)
        if m == 0:
            B[k] = N * Pm
        else:
            B[k + m] = np.sqrt(2.0) * N * Pm
            B[k - m] = np.sqrt(2.0) * N * Pm
    return B


def _precompute():
    u, wth, phi, wphi = _build_grid()
    B = _build_basis(u)  # (21, NTH)
    # trigonometric factors per (m, phi)
    # For m>0: cos(m phi); for m<0: sin(|m| phi)
    cosm = np.zeros(_NPHI)
    sinm = np.zeros(_NPHI)
    for m in range(1, _K + 1):
        cosm[m] = np.cos(m * phi)
        sinm[m] = np.sin(m * phi)
    # weight per grid point (product)
    wfull = np.outer(wth, wphi)  # (NTH, NPHI)
    return B, cosm, sinm, wfull


def _eval_f(c, B, cosm, sinm):
    # f of shape (NTH, NPHI)
    # f(theta, phi) = sum_m c_m Y_m
    # m=0: c_10 * B[10, :].broadcast
    # m>0: c_{10+m} * (sqrt2 N Pm) * cos(m phi)
    # m<0: c_{10-|m|} * (sqrt2 N Pm) * sin(|m| phi)
    # Build by summing over m
    f = np.zeros((_NTH, _NPHI), dtype=np.float64)
    # m = 0
    f += c[10] * B[10, :, None]
    for m in range(1, _K + 1):
        Pm_scaled = B[10 + m, :]  # already sqrt2*N*Pm
        f += c[10 + m] * Pm_scaled[:, None] * cosm[m][None, :]
        f += c[10 - m] * Pm_scaled[:, None] * sinm[m][None, :]
    return f


def _objective_and_grad(c, B, cosm, sinm, wfull):
    f = _eval_f(c, B, cosm, sinm)
    f2 = f * f
    s2 = np.sum(f2 * wfull)
    if s2 <= 0.0:
        return 1e12, np.zeros_like(c)
    logf2 = np.log(f2 + 1e-300)
    # logsumexp of 4*log|f| = logsumexp(2*logf2)
    a = 2.0 * logf2
    amax = np.max(a)
    lse = amax + np.log(np.sum(np.exp(a - amax)))
    val = lse - 2.0 * np.log(s2)
    # gradient
    # dval/dc = (4/s4) sum w f^3 df/dc - (4/s2) sum w f^2 df/dc
    s4 = np.exp(lse)  # sum w f^4
    df = np.zeros_like(f)
    # df/dc[10] = B[10]
    g = np.zeros_like(c)
    # m=0
    df = B[10, :, None]
    g[10] = (4.0 / s4) * np.sum(wfull * f * f2 * df) - (4.0 / s2) * np.sum(wfull * f2 * df)
    for m in range(1, _K + 1):
        Pm_scaled = B[10 + m, :]
        df = Pm_scaled[:, None] * cosm[m][None, :]
        g[10 + m] = (4.0 / s4) * np.sum(wfull * f * f2 * df) - (4.0 / s2) * np.sum(wfull * f2 * df)
        df = Pm_scaled[:, None] * sinm[m][None, :]
        g[10 - m] = (4.0 / s4) * np.sum(wfull * f * f2 * df) - (4.0 / s2) * np.sum(wfull * f2 * df)
    return val, g


def _polish(c0, B, cosm, sinm, wfull, max_iter=400, tol=1e-12):
    from scipy.optimize import minimize
    res = minimize(_objective_and_grad, c0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12,
                            'maxls': 40, 'maxfun': 4000})
    return res.x, res.fun


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    B, cosm, sinm, wfull = _precompute()

    def elapsed():
        return time.time() - t0

    from scipy.optimize import minimize

    def run_minimize(c0, max_iter, ftol):
        res = minimize(_objective_and_grad, c0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10,
                                'maxls': 30, 'maxfun': 3000})
        return res.x, res.fun

    # candidate initializations
    inits = []
    # m=0 dominant (known to reach a good basin)
    c = np.zeros(21)
    c[10] = 1.0
    inits.append(c)
    # symmetric cosine pattern (even)
    c = np.zeros(21)
    for m in range(0, 11):
        c[10 + m] = 1.0
        if m > 0:
            c[10 - m] = 1.0
    inits.append(c)
    # alternating
    c = np.zeros(21)
    for m in range(0, 11):
        c[10 + m] = 1.0 if (m % 2 == 0) else -1.0
        if m > 0:
            c[10 - m] = 1.0 if (m % 2 == 0) else -1.0
    inits.append(c)
    # random restarts
    rng = np.random.default_rng(12345)
    for _ in range(6):
        c = rng.standard_normal(21)
        inits.append(c)

    best_c = inits[0].copy()
    best_val = _objective_and_grad(best_c, B, cosm, sinm, wfull)[0]

    # First, a quick pass over all inits with moderate iterations
    for i, c0 in enumerate(inits):
        if elapsed() > _DEADLINE - 120:
            break
        c, val = run_minimize(c0, max_iter=200, ftol=1e-10)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Polish the best with high precision
    if elapsed() < _DEADLINE - 60:
        c, val = _polish(best_c, B, cosm, sinm, wfull, max_iter=400, tol=1e-12)
        if val < best_val:
            best_val = val
            best_c = c.copy()

    # Optionally polish a couple of the top inits if time remains
    if elapsed() < _DEADLINE - 90:
        for c0 in inits[:3]:
            if elapsed() > _DEADLINE - 60:
                break
            c, val = _polish(c0, B, cosm, sinm, wfull, max_iter=300, tol=1e-12)
            if val < best_val:
                best_val = val
                best_c = c.copy()

    # Final safety: ensure finite
    if not np.all(np.isfinite(best_c)):
        best_c = inits[0].copy()

    # Scale to satisfy max|c| constraints (keep in [1e-3, 1e5]); ratio is scale-invariant
    mx = np.max(np.abs(best_c))
    if mx == 0.0:
        best_c = inits[0].copy()
        mx = np.max(np.abs(best_c))
    if mx > 1e5:
        best_c = best_c * (1e4 / mx)
    elif mx < 1e-3:
        best_c = best_c * (1e-2 / mx)

    return best_c


# EVOLVE-BLOCK-END
