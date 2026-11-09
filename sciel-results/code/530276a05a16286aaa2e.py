"""Rewritten program: phase-retrieval / sinusoidal sign family + local bit polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 150.0
START = time.time()

# Precompute the autocorrelation kernel: for a signed step function on N equal
# intervals, the autoconvolution upper bound (C3) is proportional to
# sum_{k} (N - |k|) * r(k) where r(k) is the (unnormalized) autocorrelation.
# We work in the discrete signed array x (length N, values +/-1).
# Define the objective as the sum over all lags of (N-|k|)*autocorr(k),
# which is a standard proxy for the autoconvolution upper bound.
# This is equivalent (up to a constant) to ||x * x||_1 type energy.

def _autoconv_objective(x):
    """Return the autoconvolution upper bound proxy for signed array x (+/-1).

    Uses FFT for speed: autocorr(k) = sum_i x[i] x[i+k] (circular) but we want
    linear autocorrelation. We compute linear autocorr via zero-padding FFT.
    """
    n = len(x)
    # Zero-pad to next power of 2 >= 2n-1 for linear convolution
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    X = np.fft.rfft(x, m)
    # Linear autocorrelation: ifft( |X|^2 ) gives circular autocorr of zero-padded,
    # which equals linear autocorr in the first n positions.
    ac = np.fft.irfft(X * np.conj(X), m)
    ac = ac[:n]  # lags 0..n-1
    # ac[k] = sum_{i=0}^{n-1-k} x[i] x[i+k]
    weights = np.arange(n, 0, -1)  # (N - k)
    return float(np.sum(weights * ac))


def _sign_pattern(f, phi):
    """Build signed +/-1 array from frequency f (in cycles over N) and phase phi."""
    k = np.arange(N)
    s = np.sin(2.0 * np.pi * f * k / N + phi)
    return np.sign(s) * 1.0


def _surrogate(fphi):
    """Differentiable surrogate: use soft sign (tanh) so L-BFGS can optimize."""
    f = fphi[0]
    phi = fphi[1]
    k = np.arange(N)
    z = np.tanh(8.0 * np.sin(2.0 * np.pi * f * k / N + phi))
    # z in (-1,1); compute autoconv objective with z (continuous) as proxy
    n = N
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    X = np.fft.rfft(z, m)
    ac = np.fft.irfft(X * np.conj(X), m)[:n]
    weights = np.arange(n, 0, -1)
    return float(np.sum(weights * ac))


def _bit_polish(x, budget=200000):
    """Greedy + simulated-annealing style bit flips to reduce objective."""
    x = x.copy()
    cur = _autoconv_objective(x)
    best = cur
    best_x = x.copy()
    t = 1.0
    rng = np.random.default_rng(12345)
    steps = 0
    while steps < budget:
        if time.time() - START > DEADLINE - 5:
            break
        i = rng.integers(0, N)
        x[i] *= -1.0
        new = _autoconv_objective(x)
        delta = new - cur
        if delta < 0 or (t > 0.01 and rng.random() < np.exp(-delta / max(t, 1e-9))):
            cur = new
            if new < best:
                best = new
                best_x = x.copy()
        else:
            x[i] *= -1.0
        t *= 0.999
        steps += 1
    return best_x, best


def construct():
    """Return a sequence of signed step heights (length N, values +/-1).

    Score is the autoconvolution upper bound C3 (lower is better).
    """
    if time.time() - START > DEADLINE - 3:
        return np.ones(N)

    # Phase 1: optimize the sinusoidal sign family with L-BFGS on the surrogate.
    best_x = np.ones(N)
    best_val = _autoconv_objective(best_x)

    # Try a set of initial (f, phi) seeds; keep the best.
    inits = []
    for f in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 0.333, 0.667, 1.333]:
        for phi in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, np.pi / 4, 3 * np.pi / 4]:
            inits.append((f, phi))

    for f0, p0 in inits:
        if time.time() - START > DEADLINE - 10:
            break
        try:
            res = minimize(_surrogate, [f0, p0], method='L-BFGS-B',
                           options={'maxiter': 60, 'ftol': 1e-9, 'gtol': 1e-7})
            f_opt, p_opt = res.x
            x = _sign_pattern(f_opt, p_opt)
            v = _autoconv_objective(x)
            if v < best_val:
                best_val = v
                best_x = x
        except Exception:
            continue

    # Phase 2: bit-flip polish on the best pattern found.
    if time.time() - START < DEADLINE - 15:
        polished, pv = _bit_polish(best_x, budget=300000)
        if pv < best_val:
            best_x = polished
            best_val = pv

    # Ensure the output is a clean +/-1 sequence.
    best_x = np.where(best_x >= 0, 1.0, -1.0)
    return best_x
