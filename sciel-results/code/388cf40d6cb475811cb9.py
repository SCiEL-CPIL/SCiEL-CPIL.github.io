"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0  # safety margin before kill


def _elapsed():
    return time.time() - _START


def _time_left():
    return _DEADLINE - _MARGIN - _elapsed()


def _score(f):
    """Return max|A| / (sum f)^2 where A = f * f (full autoconvolution)."""
    f = np.asarray(f, dtype=float)
    s = f.sum()
    if abs(s) < 1e-12:
        return np.inf
    # autoconvolution via FFT
    n = len(f)
    m = 2 * n - 1
    F = np.fft.rfft(f, n=m)
    A = np.fft.irfft(F * np.conj(F), n=m)
    return np.max(np.abs(A)) / (s * s)


def _softmax_score(f, lam=100.0):
    """Smooth surrogate: soft-max of |A| / (sum f)^2 using log-sum-exp."""
    f = np.asarray(f, dtype=float)
    s = f.sum()
    if abs(s) < 1e-12:
        return 1e12
    n = len(f)
    m = 2 * n - 1
    F = np.fft.rfft(f, n=m)
    A = np.fft.irfft(F * np.conj(F), n=m)
    absA = np.abs(A)
    # log-sum-exp of absA
    amax = absA.max()
    if amax < 1e-300:
        logsum = amax + np.log(absA.size)
    else:
        logsum = amax + np.log(np.sum(np.exp(absA - amax)))
    # soft-max approx: exp(logsum - log(N)) but we want a smooth max; use exp(logsum)/size as mean,
    # better: use amax + (1/lam)*log(sum(exp(lam*(x-amax))))
    soft = amax + (1.0 / lam) * np.log(np.sum(np.exp(lam * (absA - amax))))
    return soft / (s * s)


def _softmax_grad(f, lam=100.0):
    """Gradient of soft-max surrogate w.r.t. f."""
    f = np.asarray(f, dtype=float)
    s = f.sum()
    if abs(s) < 1e-12:
        return np.zeros_like(f)
    n = len(f)
    m = 2 * n - 1
    F = np.fft.rfft(f, n=m)
    A = np.fft.irfft(F * np.conj(F), n=m)
    absA = np.abs(A)
    amax = absA.max()
    # weights for soft-max
    w = np.exp(lam * (absA - amax))
    w = w / w.sum()
    # d|A(k)|/df via chain rule: A = ifft(|F|^2), F = fft(f)
    # dA/dF = 2*conj(F) (elementwise) for the real part; use complex gradient
    # d|A(k)|/dA(k) = A(k)/|A(k)| (where |A|>0)
    signA = np.where(absA > 1e-12, A / absA, 0.0)
    dA_dF = 2.0 * np.conj(F)
    dsoft_dA = w * signA
    dsoft_dF = dA_dF * dsoft_dA
    dsoft_df = np.fft.irfft(dsoft_dF, n=m)[:n]
    # divide by (s^2), and account for derivative of 1/s^2
    val = (amax + (1.0 / lam) * np.log(np.sum(np.exp(lam * (absA - amax))))) / (s * s)
    # d(val)/df = (dsoft_dA/dF chain) / s^2 - 2*val * (1/s) * d(s)/df
    grad = dsoft_df / (s * s) - 2.0 * val * (1.0 / s) * np.ones_like(f)
    return grad


def _flat_spectrum_init(n, seed=0):
    """Build a real signal whose squared spectrum is approximately flat."""
    rng = np.random.RandomState(seed)
    m = 2 * n - 1
    # random phases with conjugate symmetry
    # rfft length = m//2 + 1
    nr = m // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, size=nr)
    F = np.exp(1j * phases)
    # enforce conjugate symmetry for real signal: F[k] = conj(F[m-k])
    # rfft already gives positive freqs; build full then ifft
    Ffull = np.zeros(m, dtype=complex)
    Ffull[:nr] = F
    for k in range(1, nr):
        Ffull[m - k] = np.conj(F[k])
    f = np.fft.ifft(Ffull).real[:n]
    # normalize
    s = f.sum()
    if abs(s) < 1e-12:
        f = rng.randn(n)
        s = f.sum()
    f = f / s
    return f


def _lbfgs_polish(f0, max_iter=200, tol=1e-8):
    """Simple L-BFGS-like polish using scipy if available, else gradient descent."""
    from scipy.optimize import minimize
    f0 = np.asarray(f0, dtype=float)
    res = minimize(
        _softmax_score,
        f0,
        jac=_softmax_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6},
    )
    return res.x


def _gradient_descent_polish(f0, lr=0.01, steps=200):
    """Fallback plain gradient descent."""
    f = f0.copy()
    for i in range(steps):
        if _time_left() < 2.0:
            break
        g = _softmax_grad(f, lam=50.0)
        f = f - lr * g
        # renormalize to keep sum ~ constant (helps conditioning)
        s = f.sum()
        if abs(s) > 1e-9:
            f = f / s
    return f


def construct():
    best = None
    best_val = np.inf

    # Candidate 1: flat-spectrum init
    for seed in range(8):
        if _time_left() < 10.0:
            break
        f = _flat_spectrum_init(N, seed=seed)
        val = _score(f)
        if val < best_val:
            best_val = val
            best = f.copy()

    # Polish the best flat-spectrum init
    if best is not None and _time_left() > 15.0:
        try:
            f_pol = _lbfgs_polish(best, max_iter=150)
            val = _score(f_pol)
            if val < best_val:
                best_val = val
                best = f_pol
        except Exception:
            pass

    # Additional random inits to diversify
    if _time_left() > 15.0:
        rng = np.random.RandomState(123)
        for seed in range(6):
            if _time_left() < 10.0:
                break
            f = rng.randn(N)
            s = f.sum()
            if abs(s) < 1e-9:
                continue
            f = f / s
            val = _score(f)
            if val < best_val:
                best_val = val
                best = f.copy()

    # Final polish on best
    if best is not None and _time_left() > 10.0:
        try:
            f_pol = _lbfgs_polish(best, max_iter=100)
            val = _score(f_pol)
            if val < best_val:
                best_val = val
                best = f_pol
        except Exception:
            pass

    # Safety: if we somehow have no result, return ones
    if best is None:
        best = np.ones(N)

    return best
# EVOLVE-BLOCK-END
