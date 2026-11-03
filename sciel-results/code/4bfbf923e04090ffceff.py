"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses a direct, vectorized gradient of the smooth-max objective and a
multi-start SLSQP refinement. The key improvement is a fully vectorized
gradient computation (no Python loop over k) and a tighter deadline."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    Fully vectorized: no Python loop over k.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max

    val = m + np.log(sum_exp) / beta

    # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    # grad_j = sum_k w[k] * dC_k/dh_j
    #        = -(2/n) * sum_k w[k] * (h_{j-k} [j>=k] + h_{j+k} [j+k<n])
    #
    # For each j, the first term sums over k=0..j of w[k]*h[j-k]
    #            the second term sums over k=0..n-1-j of w[k]*h[j+k]
    #
    # This is a convolution-like operation. We can compute it as:
    # term1_j = sum_{k=0}^{j} w[k] * h[j-k]  = (w * h)_j  (full convolution, first n entries)
    # term2_j = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    #          = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # Let's reverse h: h_rev[i] = h[n-1-i]
    # term2_j = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # Let i = n-1-j-k, then k = n-1-j-i
    # term2_j = sum_{i=0}^{n-1-j} w[n-1-j-i] * h[n-1-i]
    #         = (w_rev * h_rev)_{n-1-j} where w_rev[i] = w[n-1-i]
    # Actually, let's just use np.convolve or FFT.

    # Use FFT for both convolutions
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)

    # term1: (w * h) circular, take first n
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]

    # term2: sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # This is the same as the cross-correlation of w and h at lag j, but truncated.
    # Actually, term2_j = sum_{k} w[k] * h[j+k] for k such that j+k < n
    # = sum_{k} w[k] * h[j+k] * 1_{j+k < n}
    # This is the same as (w * h_flipped) at position n-1-j, where h_flipped[i] = h[n-1-i]
    h_flip = h[::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv2_full = np.fft.irfft(W * HF, n=n_fft)
    # conv2_full[n-1-j] = sum_k w[k] * h_flip[n-1-j-k] = sum_k w[k] * h[j+k]
    # for k such that n-1-j-k in [0, n-1], i.e., j+k in [0, n-1], i.e., k in [0, n-1-j]
    # So conv2_full[n-1-j] = term2_j
    term2 = conv2_full[n - 1 - np.arange(n)]

    grad = -(2.0 / n) * (conv1 + term2)

    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
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

    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)

    # 2. Constant start
    h_const = np.full(N, 0.5)
    _update_best(h_const)

    # 3. Optimization with Analytical Gradient
    def _run_optimization(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score

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
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Refine Cosine Start
    if time.time() - t0 < _DEADLINE - 40:
        _run_optimization(h_cos, beta=50.0, maxiter=200, ftol=1e-9)

    # Phase 2: Try random starts
    for trial in range(5):
        if time.time() - t0 > _DEADLINE - 20:
            break
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _fix_sum(h_rand)
        _update_best(h_rand)
        if time.time() - t0 < _DEADLINE - 15:
            _run_optimization(h_rand, beta=50.0, maxiter=100, ftol=1e-8)

    # Phase 3: Final Polish with higher beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=100.0, maxiter=300, ftol=1e-12)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
