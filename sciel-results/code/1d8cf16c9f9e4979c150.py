"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """Find optimal Hermite coefficients minimizing the C4 upper bound."""
    t0 = time.time()
    deadline = t0 + 180  # safety margin before 210s kill
    
    # Physicists' Hermite polynomials
    def H(n, x):
        if n == 0: return np.ones_like(x)
        if n == 1: return 2*x
        if n == 2: return 4*x**2 - 2
        if n == 3: return 8*x**3 - 12*x
        if n == 4: return 16*x**4 - 48*x**2 + 12
        if n == 8: return 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
        if n == 12: return 4096*x**12 - 73728*x**10 + 483840*x**8 - 1445760*x**6 + 1814400*x**4 - 806400*x**2 + 40840
        raise ValueError(f"H({n}) not implemented")
    
    # Compute the upper bound on C4 = A(f) * A(F)
    # For f(x) = exp(-x^2/4) * P(x) where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # a12 is determined by P(0) = 0
    
    def compute_bound(coeffs):
        a0, a4, a8 = coeffs
        # P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
        # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=40840
        h0_0 = 1.0
        h4_0 = 12.0
        h8_0 = 1680.0
        h12_0 = 40840.0
        a12 = -(a0 * h0_0 + a4 * h4_0 + a8 * h8_0) / h12_0
        
        # Numerical integration using Gauss-Hermite quadrature
        # ∫ e^{-x^2} f(x) dx ≈ sum w_i f(x_i)
        # We need integrals of form ∫ x^k * e^{-x^2/2} * P(x)^2 dx for A(f)
        # and ∫ ξ^k * e^{-ξ^2/2} * |F(ξ)|^2 dξ for A(F)
        
        # Actually, let's use direct numerical integration on a grid
        x = np.linspace(-20, 20, 10000)
        dx = x[1] - x[0]
        
        # Build P(x)
        P = a0 * H(0, x) + a4 * H(4, x) + a8 * H(8, x) + a12 * H(12, x)
        
        # f(x) = exp(-x^2/4) * P(x)
        f = np.exp(-x**2 / 4) * P
        
        # |f(x)|^2 = exp(-x^2/2) * P(x)^2
        f2 = np.exp(-x**2 / 2) * P**2
        
        # A(f) = ∫ x^2 |f|^2 dx / ∫ |f|^2 dx
        norm_f = np.trapz(f2, x)
        if norm_f <= 0 or not np.isfinite(norm_f):
            return 1e10
        var_f = np.trapz(x**2 * f2, x) / norm_f
        
        # Fourier transform: F(ξ) = ∫ f(x) e^{-2πiξx} dx
        # For f(x) = e^{-x^2/4} P(x), we can compute numerically
        # Or use the fact that FT of e^{-x^2/4} H_n(x) is related to H_n
        # FT[e^{-x^2/4} H_n(x)] = (-i)^n (2π)^{-1/2} * 2^{n/2} e^{-π^2 ξ^2} H_n(2πξ) * normalization
        
        # Let's compute numerically using FFT or direct integration
        # Actually, for the variance in frequency domain:
        # A(F) = ∫ ξ^2 |F(ξ)|^2 dξ / ∫ |F(ξ)|^2 dξ
        
        # By Parseval: ∫ |F|^2 dξ = ∫ |f|^2 dx = norm_f
        # And ∫ ξ^2 |F(ξ)|^2 dξ can be computed from the second moment
        
        # For a function f, the Fourier variance is related to:
        # ∫ ξ^2 |F(ξ)|^2 dξ = (1/(4π^2)) ∫ |f''(x)|^2 dx (for suitable f)
        # But this requires f to be in H^2
        
        # Alternative: compute F numerically
        # Use a sufficiently large grid for FT
        N = 2**14
        xft = np.linspace(-50, 50, N)
        dxft = xft[1] - xft[0]
        
        # Compute F(ξ) = ∫ f(x) e^{-2πiξx} dx numerically
        # This is expensive; let's use a smarter approach
        
        # Actually, for Hermite-Gaussian functions, we can use orthogonality.
        # Let f = e^{-x^2/4} Σ c_n H_n(x)
        # The FT of e^{-x^2/4} H_n(x) is: (-i)^n 2^{n/2} (2π)^{-1/2} e^{-π^2 ξ^2} H_n(2πξ)
        # Wait, need to be careful with conventions.
        
        # Let's just do direct numerical FT on a reasonable grid
        # f(x) = e^{-x^2/4} P(x)
        # F(ξ) = ∫_{-∞}^{∞} e^{-x^2/4} P(x) e^{-2πiξx} dx
        
        # Use trapezoidal rule
        # To make this efficient, precompute the exponential
        exp_term = np.exp(-x**2 / 4)
        
        # For each ξ in xft, compute F(ξ)
        # This is O(N * M) which is too slow for N=M=16384
        # Let's use FFT instead
        
        # Use FFT: sample f at equally spaced points
        nfft = 2**15
        xfft = np.linspace(-30, 30, nfft, endpoint=False)
        dxfft = xfft[1] - xfft[0]
        
        f_fft = np.exp(-xfft**2 / 4) * (a0 * H(0, xfft) + a4 * H(4, xfft) + a8 * H(8, xfft) + a12 * H(12, xfft))
        
        F_fft = np.fft.fftshift(np.fft.fft(f_fft * dxfft))
        freq = np.fft.fftshift(np.fft.fftfreq(nfft, dxfft))
        
        F2 = np.abs(F_fft)**2
        norm_F = np.trapz(F2, freq)
        if norm_F <= 0 or not np.isfinite(norm_F):
            return 1e10
        var_F = np.trapz(freq**2 * F2, freq) / norm_F
        
        bound = var_f * var_F
        if not np.isfinite(bound) or bound < 0:
            return 1e10
        return bound
    
    # Initial guess: the seed values
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try to optimize
    best_x = x0.copy()
    best_bound = compute_bound(x0)
    
    # Use Nelder-Mead or L-BFGS-B
    try:
        result = minimize(
            lambda c: compute_bound(c),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if result.success or result.fun < best_bound:
            if np.isfinite(result.fun) and result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30:
        return best_x
    
    # Try L-BFGS-B with gradient approximation
    try:
        result2 = minimize(
            lambda c: compute_bound(c),
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if np.isfinite(result2.fun) and result2.fun < best_bound:
            best_bound = result2.fun
            best_x = result2.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 20:
        return best_x
    
    # Try a few random restarts if time permits
    rng = np.random.RandomState(42)
    for i in range(10):
        if time.time() > deadline - 15:
            break
        x_rand = x0 + rng.randn(3) * 0.5
        try:
            result3 = minimize(
                lambda c: compute_bound(c),
                x_rand,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if np.isfinite(result3.fun) and result3.fun < best_bound:
                best_bound = result3.fun
                best_x = result3.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END

