"""Search for nonnegative step heights maximizing (L2 norm)^2 / (L1 norm * sup autoconvolution)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# Equal intervals over [-1/4, 1/4]; step width
_dx = (0.25 - (-0.25)) / N  # 0.5/50 = 0.01

def _autoconv_sup(h):
    """Supremum norm of the autoconvolution of the step function.
    h is an array of N nonnegative heights on equal intervals of width _dx.
    The step function is f(x) = h[i] on interval i. Its autocorrelation
    (autoconvolution) g(t) = integral f(x) f(x+t) dx.
    We approximate via discrete convolution scaled by _dx.
    """
    h = np.asarray(h, dtype=float)
    # Full discrete convolution of heights
    conv = np.convolve(h, h, mode='full')  # length 2N-1
    # Each product h[i]h[j] corresponds to overlap area ~ _dx * _dx
    # The continuous autoconvolution value at lag k is sum_i h[i] h[i+k] * _dx
    # sup over continuous t is approximated by max of these discrete values
    sup = np.max(conv) * _dx
    return sup

def _ratio(h):
    h = np.asarray(h, dtype=float)
    l2sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 <= 0:
        return 0.0
    sup = _autoconv_sup(h)
    if sup <= 0:
        return 0.0
    return l2sq / (l1 * sup)

def _neg_ratio(h):
    return -_ratio(h)

def construct():
    start = time.time()
    DEADLINE = 150.0
    SAFETY = 5.0

    best_h = None
    best_val = -np.inf

    def consider(h):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        v = _ratio(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
        return v

    # Initial candidates
    candidates = []
    # uniform
    candidates.append(np.ones(N))
    # single spike at center
    for pos in [0, N//2, N-1]:
        s = np.zeros(N); s[pos] = 1.0
        candidates.append(s)
    # a few narrow blocks
    for w in [1,2,3,4,5]:
        for pos in [0, N//2 - w//2, N - w]:
            b = np.zeros(N)
            b[pos:pos+w] = 1.0
            candidates.append(b)

    for c in candidates:
        consider(c)

    # Local optimization from each promising start using Nelder-Mead,
    # with a per-start time budget.
    elapsed = time.time() - start
    per_start = max(0.5, (DEADLINE - SAFETY - elapsed) / max(1, len(candidates)))

    for c in candidates:
        if time.time() > DEADLINE - SAFETY:
            break
        # scale to a moderate magnitude; ratio improves as scale decreases,
        # but we keep scale=1 baseline and also try smaller scales.
        for scale in [1.0, 0.1, 0.01]:
            if time.time() > DEADLINE - SAFETY:
                break
            x0 = c * scale
            try:
                res = minimize(_neg_ratio, x0, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4,
                                        'adaptive': True})
                consider(res.x)
            except Exception:
                pass

    # A final coordinate-descent refinement on the best found, if time remains.
    if best_h is not None and time.time() < DEADLINE - SAFETY:
        h = best_h.copy()
        step = 1e-3
        for it in range(2000):
            if time.time() > DEADLINE - SAFETY:
                break
            improved = False
            for i in range(N):
                old = h[i]
                for delta in (step, -step, 2*step, 0.5*step):
                    cand = h.copy()
                    cand[i] = max(0.0, old + delta)
                    if _ratio(cand) > best_val:
                        h = cand
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-7:
                    break
        consider(h)

    if best_h is None:
        return [1.0] * N
    # Return as a plain list of floats
    return [float(x) for x in best_h]
