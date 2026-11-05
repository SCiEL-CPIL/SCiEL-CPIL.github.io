"""Rewritten: low-dimensional spectral/FFT optimization of the peak autocorrelation ratio."""
N = 400

import time
import numpy as np
from scipy.optimize import minimize

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = np.convolve(h, h)
    return abs(2.0 * N * np.max(ac) / (s * s))

def _obj_soft(c, tau=8.0):
    # c: first 21 FFT coefficients (amplitudes) -> h (real)
    h = _fft_to_h(c)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = np.convolve(h, h)
    mx = np.max(ac)
    # soft max via logsumexp
    tau_l = tau
    shifted = ac - mx
    # avoid underflow
    shifted = np.clip(shifted, -1e4, 0.0)
    lse = mx + np.log(np.sum(np.exp(tau_l * shifted))) / tau_l
    return 2.0 * N * lse / (s * s)

def _obj_exact(c):
    h = _fft_to_h(c)
    return _value(h)

def _fft_to_h(c):
    # c length 21: c[0]=real DC, c[1..10] are complex amplitudes for k=1..10
    # Reconstruct h real length N
    h = np.zeros(N)
    # c[0] is real part of DC
    dc = c[0]
    for k in range(1, 11):
        ck = c[k] + 0j
        h += 2 * np.real(ck * np.exp(2j * np.pi * k * np.arange(N) / N))
    h += dc
    return h

def construct():
    start = time.time()
    deadline = start + 120.0

    # Initial h: ones
    best_h = np.ones(N)
    best_val = _value(best_h)

    # Build initial c from best_h
    H = np.fft.fft(best_h)
    c0 = np.zeros(21)
    c0[0] = np.real(H[0])
    for k in range(1, 11):
        c0[k] = np.real(H[k])
        # we'll treat complex as two? Actually we can allow c to be complex? No, minimize expects real.
        # Use c[k] = real part, and we can also store imag in c[k+10]? Better: use 1 + 2*10 = 21 real variables.
        pass

    # Instead, define variables: c[0] real, c[1..10] real and c[11..20] imag?
    # Let's restructure: variable vector x of length 21:
    # x[0] = real DC
    # x[1] = imag DC (should be 0, skip)
    # Use x[0] = real(H0), x[1..10] = real(Hk), x[11..20] = imag(Hk)
    # So length 21.

    def _x_to_h(x):
        h = np.zeros(N)
        h += x[0]
        for k in range(1, 11):
            re = x[k]
            im = x[10 + k]
            h += 2 * (re * np.cos(2*np.pi*k*np.arange(N)/N) - im * np.sin(2*np.pi*k*np.arange(N)/N))
        return h

    def _obj_soft_x(x, tau=10.0):
        h = _x_to_h(x)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        ac = np.convolve(h, h)
        mx = np.max(ac)
        tau_l = tau
        shifted = ac - mx
        shifted = np.clip(shifted, -1e4, 0.0)
        lse = mx + np.log(np.sum(np.exp(tau_l * shifted))) / tau_l
        return 2.0 * N * lse / (s * s)

    def _obj_exact_x(x):
        h = _x_to_h(x)
        return _value(h)

    # initial x
    H = np.fft.fft(best_h)
    x0 = np.zeros(21)
    x0[0] = np.real(H[0])
    for k in range(1, 11):
        x0[k] = np.real(H[k])
        x0[10 + k] = np.imag(H[k])

    # Try a few initializations
    rng = np.random.RandomState(123)
    inits = [x0.copy()]

    # random small perturbations
    for _ in range(3):
        inits.append(x0 + rng.randn(21) * 0.1)

    # sine-based: choose a few k to emphasize
    for k_emph in [3, 5, 7]:
        x = x0.copy()
        x[k_emph] += 5.0
        inits.append(x)

    best_x = x0.copy()

    for xi in inits:
        v = _obj_exact_x(xi)
        if v < best_val:
            best_val = v
            best_x = xi.copy()
            best_h = _x_to_h(xi)

    # Optimize with L-BFGS-B on the low-dim surrogate
    elapsed = time.time() - start
    if elapsed < deadline - 10:
        try:
            res = minimize(_obj_soft_x, best_x, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10},
                           args=(10.0,))
            if res.success:
                v = _obj_exact_x(res.x)
                if v < best_val:
                    best_val = v
                    best_x = res.x.copy()
                    best_h = _x_to_h(res.x)
        except Exception:
            pass

    # Polish with exact objective
    elapsed = time.time() - start
    if elapsed < deadline - 5:
        try:
            res2 = minimize(_obj_exact_x, best_x, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res2.success:
                v = _obj_exact_x(res2.x)
                if v < best_val:
                    best_val = v
                    best_x = res2.x.copy()
                    best_h = _x_to_h(res2.x)
        except Exception:
            pass

    # Final safety
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
