"""Optimized program: fast multi-start optimization of the analytic C4 bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import warnings

warnings.filterwarnings("ignore")

# Probabilists' Hermite polynomials (He_n)
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 54x^10 + 1260x^8 - 15120x^6 + 75600x^4 - 127008x^2 + 10395
def _h0(x):
    return np.ones_like(x)
def _h4(x):
    x2 = x * x
    return x2 * x2 - 6.0 * x2 + 3.0
def _h8(x):
    x2 = x * x
    x4 = x2 * x2
    return x4 * x4 - 28.0 * x2 * x4 + 210.0 * x4 - 420.0 * x2 + 105.0
def _h12(x):
    x2 = x * x
    x4 = x2 * x2
    x6 = x2 * x4
    x8 = x4 * x4
    x10 = x2 * x8
    x12 = x4 * x8
    return x12 - 54.0 * x10 + 1260.0 * x8 - 15120.0 * x6 + 75600.0 * x4 - 127008.0 * x2 + 10395.0

def _c12(c0, c4, c8):
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    return -(c0 + 3.0 * c4 + 105.0 * c8) / 10395.0

def _g(x, c):
    c0, c4, c8 = c
    c12 = _c12(c0, c4, c8)
    return c0 * _h0(x) + c4 * _h4(x) + c8 * _h8(x) + c12 * _h12(x)

def _bound(c, grid):
    """Upper bound on A(f)*A(fft(f)) = (max|g|/L2) * (max|g_hat|/L2).
    g_hat is the Fourier transform; for this normalization the bound is
    computed as the product of the two L-infinity over L2 ratios."""
    c0, c4, c8 = c
    c12 = _c12(c0, c4, c8)
    g = c0 * _h0(grid) + c4 * _h4(grid) + c8 * _h8(grid) + c12 * _h12(grid)
    # L2 norm of g (continuous, approximate via dense grid)
    l2 = np.sqrt(np.trapz(g * g, grid))
    if l2 < 1e-12:
        return 1e12
    max_g = np.max(np.abs(g))
    # Fourier transform of g: use FFT on the grid (approximation of the
    # continuous FT). The bound uses the L-infinity norm of the FT.
    n = len(grid)
    dx = grid[1] - grid[0]
    # Shift to center for a cleaner FT magnitude
    g_centered = np.fft.fftshift(g)
    G = np.fft.fft(g_centered) * dx
    G = np.fft.fftshift(G)
    max_g_hat = np.max(np.abs(G))
    return (max_g / l2) * (max_g_hat / l2)

def construct():
    start = time.time()
    deadline = start + 180.0
    safety = 20.0  # return at least this many seconds before the 210s kill

    # Seed / prior best (Goncalves et al. 2017)
    prior = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Build a dense, symmetric grid for the bound evaluation.
    # The Hermite-Gaussian function is even, so a symmetric grid is natural.
    half = 12.0
    npts = 4096
    grid = np.linspace(-half, half, npts)

    def objective(c):
        return _bound(c, grid)

    # Evaluate the prior
    best_c = prior.copy()
    best_val = objective(best_c)

    # Multi-start Nelder-Mead with a strict time budget.
    rng = np.random.default_rng(12345)
    # Start points: the prior plus perturbations of increasing scale.
    starts = [prior.copy()]
    scales = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
    for s in scales:
        for _ in range(6):
            pert = rng.standard_normal(3) * s
            starts.append(prior + pert)
    # A few fully random starts in a reasonable box
    for _ in range(20):
        starts.append(rng.uniform(-5.0, 5.0, size=3))

    max_iter = 400
    for i, x0 in enumerate(starts):
        if time.time() + 8.0 > deadline - safety:
            break
        try:
            res = minimize(
                objective, x0, method="Nelder-Mead",
                options={"maxiter": max_iter, "xatol": 1e-9, "fatol": 1e-11}
            )
            if res.success or res.fun < best_val:
                val = res.fun
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_c = res.x.copy()
        except Exception:
            continue

    # Final local refinement around the best point found.
    if time.time() + 10.0 < deadline - safety:
        try:
            res = minimize(
                objective, best_c, method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-11, "fatol": 1e-13}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Guarantee we never return a worse value than the prior (the seed).
    prior_val = objective(prior)
    if not np.isfinite(best_val) or best_val > prior_val:
        best_c = prior.copy()
        best_val = prior_val

    # Hard safety: if we are too close to the deadline, return immediately.
    if time.time() > deadline - safety:
        return np.array(best_c, dtype=float)

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
