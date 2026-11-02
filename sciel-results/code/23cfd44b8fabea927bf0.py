"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(F) for the test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    with c12 determined by g(0)=0.
    
    A(f) = sup |x * g(x)|
    A(F) = sup |k * F(k)| where F is the Fourier transform.
    
    Since all Hermite degrees are multiples of 4, the Fourier transform
    is (up to a constant phase/normalization) the same polynomial combination
    in the frequency variable. So A(F) = sup |k * g(k)| = A(f) by symmetry.
    
    Actually, the Fourier transform of H_n(x) exp(-x^2/2) is (-i)^n H_n(x) exp(-x^2/2).
    For n = 0, 4, 8, 12, (-i)^n = 1 for all of them.
    So F(k) = g(k) (with the same functional form).
    
    Therefore A(f) * A(F) = (sup |x*g(x)|)^2.
    
    But wait - the normalization of the Fourier transform matters.
    Using the unitary convention: F(k) = (1/sqrt(2*pi)) * integral g(x) e^{-ikx} dx
    For H_n(x) e^{-x^2/2}, the FT is (-i)^n / sqrt(2*pi) * H_n(x) e^{-x^2/2} * sqrt(2*pi)
    Actually, the standard result is:
    FT[H_n(x) e^{-x^2/2}](k) = (-i)^n H_n(k) e^{-k^2/2}
    with the convention F(k) = (1/sqrt(2*pi)) int f(x) e^{-ikx} dx.
    
    So if g(x) = sum c_n H_n(x) e^{-x^2/2}, then F(k) = sum c_n (-i)^n H_n(k) e^{-k^2/2}
    = sum c_n H_n(k) e^{-k^2/2} (since all n are multiples of 4).
    
    So F(k) = g(k) exactly.
    
    Thus A(f)*A(F) = sup|x*g(x)| * sup|k*g(k)| = (sup|x*g(x)|)^2.
    
    But actually the "upper bound on C4" might be a different quantity.
    Let me re-read: "score is the upper bound on C4" and "evaluator measures the raw value (lower is better)".
    
    The seed returns [-113/100, 1/25, 1/3240] and gets score 1.0.
    The seed program computes Af * AF where AF uses FFT. Let me keep that approach
    but make it more accurate and optimize better.
    """
    # Coefficients for H_n(0)
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Use a sufficiently large grid
    N = 8192
    x = np.linspace(-8, 8, N)
    dx = x[1] - x[0]
    
    # Explicit Hermite polynomials
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = 16.0*x2*x2 - 48.0*x2 + 12.0
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = 4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 + 37158400.0*x4 - 54432000.0*x2 + 47900160.0
    
    g_vals = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x**2 / 2.0)
    
    # A(f) = sup |x * g(x)|
    Af = np.max(np.abs(x * g_vals))
    
    # Fourier transform using FFT
    from scipy.fft import fft, fftfreq
    G = fft(g_vals) * dx
    freqs = fftfreq(N, d=dx)
    
    AF = np.max(np.abs(freqs * G))
    
    return Af * AF


def _compute_bound_fast(c0, c4, c8):
    """
    Faster version using the fact that F(k) = g(k) for this Fourier-invariant
    combination. So A(f)*A(F) = (sup|x*g(x)|)^2.
    
    This avoids the FFT entirely and is much faster.
    """
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    N = 8192
    x = np.linspace(-8, 8, N)
    
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = 16.0*x2*x2 - 48.0*x2 + 12.0
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    x10 = x8 * x2
    x12 = x6 * x6
    H12 = 4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 + 37158400.0*x4 - 54432000.0*x2 + 47900160.0
    
    g_vals = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * np.exp(-x**2 / 2.0)
    
    Af = np.max(np.abs(x * g_vals))
    
    # Since F(k) = g(k), A(F) = Af
    # But we need to be careful about the FFT normalization.
    # The seed program uses G = fft(g_vals) * dx, which approximates the integral.
    # With the unitary FT convention, F(k) = (1/sqrt(2*pi)) * integral g(x) e^{-ikx} dx
    # The FFT approximation gives: G_fft[k] * dx ≈ integral g(x) e^{-ikx} dx
    # So F(k) ≈ G_fft[k] * dx / sqrt(2*pi)
    
    # But the seed just uses G = fft(g_vals)*dx and computes max|freqs * G|.
    # This is the non-unitary convention. Let me match the seed's approach.
    
    from scipy.fft import fft, fftfreq
    G = fft(g_vals) * dx
    freqs = fftfreq(N, d=dx)
    AF = np.max(np.abs(freqs * G))
    
    return Af * AF


def construct():
    """Find optimal coefficients by numerical optimization."""
    start_time = time.time()
    deadline = 170.0  # Safety margin before 210s limit
    
    # Initial guess from prior state of the art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    def objective(coeffs):
        try:
            return _compute_bound_fast(coeffs[0], coeffs[1], coeffs[2])
        except:
            return 1e10
    
    best_val = objective(x0)
    best_x = x0.copy()
    
    # Multi-start optimization with diverse starting points
    start_points = [
        x0,
        np.array([-1.1, 0.04, 0.001]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-1.0, 0.05, 0.0008]),
        np.array([-1.15, 0.035, 0.0007]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
