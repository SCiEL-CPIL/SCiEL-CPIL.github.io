"""Minimize the uncertainty upper bound C4 for a Hermite-Gaussian test function."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite

# Deadline guard: kill before 210s wall-clock, keep safety margin.
_START = time.time()
_DEADLINE = 205.0  # 5s margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _g_and_ft(a0, a4, a8):
    """Return g(x) and its Fourier transform values on a grid, plus norms."""
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 10395.0
    return a0, a4, a8, a12


def _compute_bound(a0, a4, a8):
    """Compute C4 = ||g||_1 * ||g_hat||_1 with ||g||_2 = 1 normalization."""
    a0, a4, a8, a12 = _g_and_ft(a0, a4, a8)

    # Grid for numerical integration
    N = 4096
    x = np.linspace(-12.0, 12.0, N)
    dx = x[1] - x[0]
    gauss = np.exp(-0.5 * x * x)
    g = (a0 * eval_hermite(0, x) + a4 * eval_hermite(4, x)
         + a8 * eval_hermite(8, x) + a12 * eval_hermite(12, x)) * gauss

    # Normalize so ||g||_2 = 1
    norm2 = np.sqrt(np.sum(g * g) * dx)
    if norm2 < 1e-15:
        return 1e10
    g = g / norm2

    # L1 norm of g
    l1_g = np.sum(np.abs(g)) * dx

    # Fourier transform via FFT (unitary, matches eigenfunction convention up to phase)
    # Use a proper DFT: g_hat(k) = (1/sqrt(2*pi)) * integral g(x) e^{-ikx} dx
    # Use FFT with appropriate scaling.
    gft = np.fft.fftshift(np.fft.fft(g))
    # Frequency spacing
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2.0 * np.pi
    # Unitary FFT scaling: multiply by dx/sqrt(2*pi)
    gft = gft * (dx / np.sqrt(2.0 * np.pi))
    l1_gft = np.sum(np.abs(gft)) * (2.0 * np.pi / N / dx)  # dk spacing

    return l1_g * l1_gft


def _objective(params):
    a0, a4, a8 = params
    return _compute_bound(a0, a4, a8)


def construct():
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound(*best_params)

    # Multi-start optimization
    rng = np.random.default_rng(42)
    n_starts = 12
    for i in range(n_starts):
        if _time_left() < 15.0:
            break
        if i == 0:
            start = best_params.copy()
        else:
            # Random starts in a reasonable box
            start = np.array([
                rng.uniform(-2.0, 2.0),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.05, 0.05),
            ])
        try:
            res = minimize(_objective, start, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
        except Exception:
            continue

    return np.array(best_params)
# EVOLVE-BLOCK-END
