"""Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/(sum h)^2.

Fresh design: parameterize h as a smooth, windowed, sign-symmetric (odd)
signal whose autocorrelation approximates a smoothed boxcar (flat inside,
~0 outside). We optimize a smooth surrogate (sum of squared interior
autocorrelation entries, which equals sum(h^2) via Parseval) plus a
peak-penalty, then polish the exact signed-max objective with a few
gradient-descent / coordinate-descent steps. The sign of h is fixed so
that sum(h) > 0, which makes the signed maximum the positive peak.
"""
import numpy as np
import time

N = 400
DEADLINE = 120.0          # hard wall-clock limit (evaluator kills at 150s)
SAFETY = 10.0             # keep a safety margin


def _value(h):
    s = float(np.sum(h))
    if s == 0.0:
        return float("inf")
    c = np.convolve(h, h)          # full autoconvolution, signed
    return abs(2.0 * N * float(np.max(c)) / (s * s))


def _smooth_boxcar_target(n_half, width):
    """Target autocorrelation: flat 1 for |k|<=n_half, smooth falloff outside."""
    k = np.arange(-N + 1, N)
    t = 1.0 - np.tanh((np.abs(k) - n_half) / width)
    return t


def _initial_h(n_half, width, sigma):
    """Build a smooth, sign-symmetric (odd) h from a smoothed boxcar autocorr.

    We want autocorr(h) ~ boxcar. The square root of a boxcar in the
    frequency domain gives a smooth h. We construct h in frequency domain
    so that |H|^2 approximates the DTFT of the boxcar, then take the
    magnitude spectrum and set a linear phase (odd signal) via a Hilbert-like
    construction. Simpler: take h = real part of sqrt of the spectrum of the
    target autocorrelation, then enforce odd symmetry.
    """
    # Target autocorrelation in time domain
    target = _smooth_boxcar_target(n_half, width)
    # Its DTFT (periodic) via FFT
    spec = np.fft.rfft(target, n=2 * N - 1)
    # We want |H|^2 ~ spec. Use the magnitude of the FFT of target as a
    # proxy for the desired power spectrum, then take sqrt.
    mag = np.sqrt(np.maximum(spec.real, 0.0))
    # Reconstruct h from the magnitude spectrum with zero phase, then make it
    # odd (sign-symmetric) by anti-symmetrizing, and smooth.
    h = np.fft.irfft(mag, n=2 * N - 1)
    h = h[:N]
    # Make h odd-symmetric about the center: h[i] = -h[N-1-i]
    h = h - h[::-1]
    # Smooth
    k = np.arange(N)
    c = (N - 1) / 2.0
    h = h * np.exp(-0.5 * ((k - c) / sigma) ** 2)
    return h


def _refine(h, iters, lr):
    """Polish the exact objective with gradient descent on a smoothed max."""
    h = h.copy()
    for _ in range(iters):
        s = float(np.sum(h))
        if s == 0.0:
            break
        # value uses signed max of conv(h,h)
        c = np.convolve(h, h)
        m = float(np.max(c))
        # Smoothed surrogate: soft-max of the convolution entries weighted
        # so that the peak dominates. We use a squared-max penalty on the
        # positive peak plus keep sum fixed.
        # Gradient of max(c) w.r.t. h: c is quadratic, dc/dh = 2*conv(h, e_k)
        # where e_k is the unit vector at the argmax lag.
        kmax = int(np.argmax(c))
        # d conv(h,h)[kmax] / d h[j] = h[kmax - j] + h[j - kmax] (valid indices)
        grad = np.zeros(N)
        # term1: h[kmax - j] for j in [0, N-1], kmax-j in [0,N-1]
        a = kmax
        b = kmax - (N - 1)
        lo = max(0, b)
        hi = min(N - 1, a)
        if hi >= lo:
            j = np.arange(lo, hi + 1)
            grad[j] += h[kmax - j]
        # term2: h[j - kmax] for j in [0, N-1], j-kmax in [0,N-1]
        a = (N - 1) + kmax
        b = kmax
        lo = max(0, b)
        hi = min(N - 1, a)
        if hi >= lo:
            j = np.arange(lo, hi + 1)
            grad[j] += h[j - kmax]
        # Objective ~ m / s^2 (with s>0). d/dh (m/s^2) = (grad*m_den - 2*m*ds)/s^2
        ds = np.ones(N)
        val = m / (s * s)
        g = (val * s * s * grad - 2.0 * m * s * ds) / (s * s * s)
        # Simplify: g = grad/s^2 - 2*m*ds/s^3
        g = grad / (s * s) - 2.0 * m * ds / (s ** 3)
        h -= lr * g
        # Re-center sign so sum>0
        if float(np.sum(h)) < 0:
            h = -h
    return h


def _coordinate_descent(h, iters):
    """Cheap coordinate descent on the exact objective."""
    h = h.copy()
    best = _value(h)
    for _ in range(iters):
        improved = False
        for j in range(0, N, 1):
            old = h[j]
            # try a few perturbations
            for cand in (old + 0.05, old - 0.05, old + 0.01, old - 0.01):
                h[j] = cand
                v = _value(h)
                if v < best:
                    best = v
                    improved = True
                else:
                    h[j] = old
        if not improved:
            break
    return h


def construct():
    t0 = time.time()
    # Start from a good smooth, odd-symmetric seed
    n_half = N // 2 - 1
    width = 8.0
    sigma = N / 6.0
    h = _initial_h(n_half, width, sigma)

    # Ensure sum > 0
    if float(np.sum(h)) < 0:
        h = -h
    # Normalize (value is scale-invariant, but keep magnitudes sane)
    h = h / (np.max(np.abs(h)) + 1e-12)

    best_h = h.copy()
    best_v = _value(h)

    # Gradient refinement on the exact signed-max objective
    while time.time() - t0 < DEADLINE - SAFETY:
        h = _refine(best_h, iters=200, lr=1e-3)
        if float(np.sum(h)) < 0:
            h = -h
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        else:
            # reduce learning rate
            break

    # Coordinate-descent polish with remaining time
    while time.time() - t0 < DEADLINE - SAFETY:
        h = _coordinate_descent(best_h, iters=30)
        if float(np.sum(h)) < 0:
            h = -h
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        else:
            break

    # Final sign fix
    if float(np.sum(best_h)) < 0:
        best_h = -best_h

    return best_h


if __name__ == "__main__":
    h = construct()
    print("value =", _value(h))
