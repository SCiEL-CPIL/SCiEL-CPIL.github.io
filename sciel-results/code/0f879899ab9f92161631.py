"""Second autocorrelation inequality - skew-optimized amplitude search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 135.0  # safety margin before 150s kill


def _autoconv_peak(h):
    """Compute max of linear autoconvolution h*h[0..n-1]."""
    n = len(h)
    fft_h = np.fft.rfft(h, n=2 * n)
    autoconv = np.fft.irfft(fft_h * np.conj(fft_h), n=2 * n)[:n]
    return autoconv.max()


def _score(h):
    """Compute (L2^2) / (L1 * ||autoconv||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 < 1e-15:
        return 0.0
    l2_sq = float(np.dot(h, h))
    peak = _autoconv_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)


def _skew_profile(exp_p):
    """Monotonic skew profile h_i = (i+1)^p for i in 0..N-1, normalized to L1=1."""
    idx = np.arange(1, N + 1, dtype=float)
    h = np.power(idx, exp_p)
    h = h / h.sum()
    return h


def _bump_profile(c, w, side=1):
    """Gaussian-like bump centered at c with width w, truncated to [0, N)."""
    idx = np.arange(N, dtype=float)
    h = np.exp(-((idx - c) ** 2) / (2 * w * w))
    # Optionally zero out one side to make it asymmetric
    if side == 1:
        h[idx < c] *= 0.3
    elif side == -1:
        h[idx > c] *= 0.3
    h = np.maximum(0, h)
    s = h.sum()
    if s < 1e-15:
        return np.zeros(N)
    return h / s


def construct():
    t_start = time.time()
    time_budget = _DEADLINE - t_start - 5.0
    if time_budget < 1.0:
        return [1.0] * N

    rng = np.random.RandomState(42)

    best_h = np.ones(N, dtype=float) / N
    best_score = _score(best_h)

    def _try(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        s = h.sum()
        if s < 1e-15:
            return
        h = h / s
        sc = _score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # Phase 1: skew profiles (monotonic) with varying steepness
    exponents = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0, 50.0, 80.0, 150.0]
    for p in exponents:
        if time.time() - t_start > _DEADLINE - 10.0:
            break
        _try(_skew_profile(p))
        _try(_skew_profile(p)[::-1])  # decreasing

    # Phase 2: asymmetric bumps
    for c in [10, 15, 20, 25, 30, 35, 40]:
        for w in [2, 3, 4, 6, 8, 12]:
            for side in [0, 1, -1]:
                if time.time() - t_start > _DEADLINE - 10.0:
                    break
                _try(_bump_profile(c, w, side))

    # Phase 3: local optimization from best skew profile
    if time.time() - t_start < _DEADLINE - 15.0:
        # Take best skew exponent found (we'll just optimize from a strong candidate)
        start_h = _skew_profile(10.0)
        # Parameterize via log-softmax to keep positive
        x0 = np.log(start_h + 1e-12)

        def neg_score(x):
            a = np.exp(x)
            a = a / a.sum()
            sc = _score(a)
            return -sc

        def grad_neg_score(x):
            # Numerical gradient
            eps = 1e-7
            g = np.zeros_like(x)
            base = neg_score(x)
            for i in range(N):
                xp = x.copy()
                xp[i] += eps
                g[i] = (neg_score(xp) - base) / eps
            return g

        for restart in range(4):
            if time.time() - t_start > _DEADLINE - 15.0:
                break
            if restart == 0:
                x0 = np.log(_skew_profile(10.0) + 1e-12)
            elif restart == 1:
                x0 = np.log(_skew_profile(30.0) + 1e-12)
            elif restart == 2:
                x0 = np.log(_bump_profile(30, 5, 1) + 1e-12)
            else:
                x0 = np.log(_skew_profile(10.0)[::-1] + 1e-12)

            res = minimize(neg_score, x0, method='BFGS', jac=grad_neg_score,
                           options={'maxiter': 80, 'gtol': 1e-8})
            a = np.exp(res.x)
            a = a / a.sum()
            _try(a)

    # Phase 4: Nelder-Mead on log-params from top candidates
    if time.time() - t_start < _DEADLINE - 12.0:
        for p in [5.0, 10.0, 20.0]:
            if time.time() - t_start > _DEADLINE - 12.0:
                break
            x0 = np.log(_skew_profile(p) + 1e-12)

            def neg_log_ratio(x):
                a = np.exp(x)
                a = a / a.sum()
                sc = _score(a)
                if sc < 1e-15:
                    return 0.0
                return -np.log(sc)

            res = minimize(neg_log_ratio, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            a = np.exp(res.x)
            a = a / a.sum()
            _try(a)

    # Phase 5: random skew-like perturbations
    if time.time() - t_start < _DEADLINE - 10.0:
        for _ in range(30):
            if time.time() - t_start > _DEADLINE - 10.0:
                break
            base = _skew_profile(rng.uniform(3, 40))
            noise = rng.randn(N) * 0.3
            a = np.exp(np.log(base + 1e-12) + noise)
            _try(a)

    # Final cleanup
    best_h = np.maximum(0, best_h)
    if best_h.sum() < 1e-15:
        best_h = np.ones(N)
    else:
        best_h = best_h / best_h.sum()

    return best_h.tolist()
