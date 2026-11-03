"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines spectral projection (L-BFGS-B on DFT coefficients) with a bisection on
the overlap threshold C_max using a convex feasibility objective, followed by
high-precision smooth-max SLSQP polish passes."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 138  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r_k = sum_{i=k}^{N-1} h_i * h_{i-k} for k=0..N-1 via FFT."""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]


def _overlap_values(h):
    """C_k = (2/N) * (N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))


def _smooth_max_value(h, beta):
    """Smooth maximum of overlap values via log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    m = np.max(cs)
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """Gradient of smooth max objective. Returns (value, gradient)."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    grad = np.zeros(n)
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n - k]
        if k < n:
            grad[:n - k] += factor * h[k:]
    return val, grad


def _violation_obj_grad(h, c_max):
    """
    Convex feasibility objective: sum_k max(0, C_k - c_max)^2 and its gradient.
    This is a smooth surrogate for the constraint max(C_k) <= c_max.
    """
    n = len(h)
    cs = _overlap_values(h)
    viol = np.maximum(0.0, cs - c_max)
    obj = np.sum(viol ** 2)
    grad = np.zeros(n)
    mask = cs > c_max
    if np.any(mask):
        idx = np.where(mask)[0]
        weights = 2.0 * (cs[idx] - c_max)
        for k, w in zip(idx, weights):
            factor = -(2.0 / n) * w
            if k > 0:
                grad[k:] += factor * h[:n - k]
            if k < n:
                grad[:n - k] += factor * h[k:]
    return obj, grad


def _fix_sum(h):
    """Adjust h to sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)

    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)

    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)

    return h


def _spectral_h_from_coeffs(coeffs):
    """Reconstruct h from real DFT coefficients (modes 1..K)."""
    n_modes = len(coeffs) // 2
    h = np.full(N, 0.5)
    i = np.arange(N)
    for k in range(1, n_modes + 1):
        a = coeffs[2 * (k - 1)]
        b = coeffs[2 * (k - 1) + 1]
        h += a * np.cos(2.0 * np.pi * k * i / N)
        h += b * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral_coeffs(h, K):
    """Extract first K real DFT coefficients from h."""
    H = np.fft.rfft(h)
    out = []
    for k in range(1, K + 1):
        a = 2.0 / N * H[k].real
        b = -2.0 / N * H[k].imag
        out.append(a)
        out.append(b)
    return out


def _optimize_spectral(h_init, K, beta, maxiter):
    """Optimize spectral coefficients to minimize smooth max objective."""
    coeffs0 = _h_to_spectral_coeffs(h_init, K)

    def obj(coeffs):
        h = _spectral_h_from_coeffs(coeffs)
        penalty = 0.0
        if np.any(h < 0) or np.any(h > 1):
            penalty = np.sum(np.maximum(0, -h) ** 2) + np.sum(np.maximum(0, h - 1) ** 2)
        val, _ = _smooth_max_grad(h, beta)
        return val + 1.0 * penalty

    try:
        res = minimize(obj, coeffs0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success:
            h_opt = _spectral_h_from_coeffs(res.x)
            return h_opt
    except Exception:
        pass
    return h_init


def _run_slsqp_smooth(h0, beta, maxiter, ftol, gtol):
    """Run SLSQP on the smooth max objective."""
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N

    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val

    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad

    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def _feasibility_check(h0, c_max, maxiter=150):
    """
    Minimize sum of squared violations of C_k <= c_max via SLSQP.
    Returns (achieved_max, solution_h).
    """
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N

    def obj_func(x):
        obj, _ = _violation_obj_grad(x, c_max)
        return obj

    def jac_func(x):
        _, grad = _violation_obj_grad(x, c_max)
        return grad

    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-10})
        h_sol = res.x
        h_sol = _fix_sum(h_sol)
        achieved = _score(h_sol)
        return achieved, h_sol
    except Exception:
        return _score(h0), h0


def _polish(h, beta, maxiter, ftol, gtol):
    """High-precision polish pass with tighter tolerances."""
    h0 = h.copy()
    h_new = _run_slsqp_smooth(h0, beta, maxiter, ftol, gtol)
    h_new = _fix_sum(h_new)
    if _score(h_new) < _score(h0):
        return h_new
    return h0


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # 1. Cosine start
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)

    # 2. Spectral optimization for a good starting point
    for K in [2, 4, 8, 16]:
        if time.time() - t0 > _DEADLINE - 40:
            break
        h_spec = _optimize_spectral(best_h, K, beta=50.0, maxiter=200)
        _update_best(h_spec)

    # 3. Bisection on C_max using convex feasibility objective
    lo = 0.0
    hi = best_score
    # Run bisection for a fixed number of iterations or until time runs out
    for _ in range(20):
        if time.time() - t0 > _DEADLINE - 30:
            break
        if hi - lo < 1e-9:
            break
        mid = (lo + hi) / 2.0
        achieved, h_sol = _feasibility_check(best_h, mid, maxiter=150)
        if achieved <= mid + 1e-7:
            hi = mid
            _update_best(h_sol)
        else:
            lo = mid
            # Update best if this solution is still good
            _update_best(h_sol)

    # 4. Smooth-max SLSQP polish with moderate beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_opt = _run_slsqp_smooth(best_h, beta=50.0, maxiter=300, ftol=1e-10, gtol=1e-8)
        _update_best(h_opt)

    # 5. Random restarts with spectral polish (few, to save time)
    if time.time() - t0 < _DEADLINE - 25:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            h_spec_rand = _optimize_spectral(h_rand, K=8, beta=50.0, maxiter=100)
            _update_best(h_spec_rand)

    # 6. High-precision polish passes
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_polished = _polish(best_h, beta=100.0, maxiter=500, ftol=1e-12, gtol=1e-10)
        _update_best(h_polished)

    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_final = _polish(best_h, beta=200.0, maxiter=1000, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
