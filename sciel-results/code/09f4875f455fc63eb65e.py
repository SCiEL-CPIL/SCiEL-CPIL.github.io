"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_start_time = time.time()
_DEADLINE = 210.0  # wall-clock limit
_SAFETY_MARGIN = 10.0  # seconds before deadline


def _hermite_h(n, x):
    """Compute physicists' Hermite polynomial H_n(x) via recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    h0 = np.ones_like(x)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _build_g(x, a0, a4, a8):
    """Build g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2).
    a12 is chosen so that P(0) = 0."""
    h0 = _hermite_h(0, x)
    h4 = _hermite_h(4, x)
    h8 = _hermite_h(8, x)
    h12 = _hermite_h(12, x)
    # P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-103950
    h0_0 = 1.0
    h4_0 = -12.0
    h8_0 = 1680.0
    h12_0 = -103950.0
    a12 = -(a0 * h0_0 + a4 * h4_0 + a8 * h8_0) / h12_0
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-x**2 / 2.0), a12


def _fourier_transform_g(a0, a4, a8, k):
    """Compute Fourier transform of g at frequency k.
    Using the property that FT of H_n(x)*exp(-x^2/2) relates to H_n evaluated at -k."""
    # For the convention F(k) = integral g(x) exp(-i*k*x) dx
    # FT of H_n(x) exp(-x^2/2) = sqrt(2*pi) * i^n * H_n(-k) * exp(-k^2/2)
    # But we need to be careful with the exact convention used in the problem.
    # Let's compute numerically via FFT for accuracy.
    x = np.linspace(-20, 20, 2048)
    g, _ = _build_g(x, a0, a4, a8)
    dx = x[1] - x[0]
    # FFT
    G = np.fft.fftshift(np.fft.fft(g * dx))
    k_freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx))
    # Interpolate to desired k values
    from scipy.interpolate import interp1d
    # Make sure k_freqs is sorted
    idx = np.argsort(k_freqs)
    k_sorted = k_freqs[idx]
    G_sorted = G[idx]
    f_interp = interp1d(k_sorted, G_sorted, kind='cubic', fill_value='extrapolate')
    return f_interp(k)


def _compute_bound(a0, a4, a8):
    """Compute the upper bound C4 = ||g||_1^2 / min|g_hat|^2 or similar.
    The uncertainty product A(f)*A(F(f)) involves ||g||_1 and min|g_hat|."""
    # Compute ||g||_1
    x = np.linspace(-15, 15, 4096)
    g, _ = _build_g(x, a0, a4, a8)
    dx = x[1] - x[0]
    norm1_sq = np.sum(np.abs(g))**2 * dx**2  # |integral g|^2? No, ||g||_1 = integral |g|
    # Actually ||g||_1 = integral |g(x)| dx
    norm1 = np.sum(np.abs(g)) * dx
    
    # Compute min |g_hat(k)|
    k = np.linspace(-10, 10, 4096)
    g_hat = _fourier_transform_g(a0, a4, a8, k)
    min_abs_g_hat = np.min(np.abs(g_hat))
    
    if min_abs_g_hat < 1e-12:
        return np.inf
    
    # The bound structure: C4 ~ ||g||_1 / min|g_hat| or (||g||_1 / min|g_hat|)^2
    # From the knowledge: "ratio ||g||_1 / min|ĝ|"
    bound = norm1 / min_abs_g_hat
    return bound


def _compute_bound_fast(coeffs):
    """Fast computation of the bound for optimization."""
    a0, a4, a8 = coeffs
    try:
        return _compute_bound(a0, a4, a8)
    except:
        return 1e10


def construct():
    """Find optimal coefficients (a0, a4, a8) minimizing the uncertainty bound."""
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return np.array([-113/100, 1/25, 1/3240])
    
    # Start with the known good point
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_bound = _compute_bound_fast(best_coeffs)
    
    # Grid search over a reasonable range
    # Based on the known solution, coefficients are O(1) to O(10)
    # Let's search in a box around the known solution
    centers = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    step = 0.5
    
    # First do a coarse grid
    for a0 in centers:
        if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
            break
        for a4 in centers:
            if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
                break
            for a8 in centers:
                if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
                    break
                coeffs = np.array([a0, a4, a8])
                b = _compute_bound_fast(coeffs)
                if b < best_bound:
                    best_bound = b
                    best_coeffs = coeffs.copy()
    
    # Local refinement using scipy
    if time.time() - _start_time < _DEADLINE - _SAFETY_MARGIN:
        try:
            result = minimize(
                _compute_bound_fast,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except:
            pass
    
    # Additional refinement with tighter settings
    if time.time() - _start_time < _DEADLINE - _SAFETY_MARGIN:
        try:
            result2 = minimize(
                _compute_bound_fast,
                best_coeffs,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            if result2.fun < best_bound:
                best_bound = result2.fun
                best_coeffs = result2.x
        except:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
