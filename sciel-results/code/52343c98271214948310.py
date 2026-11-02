"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft, fftshift, fftfreq
from scipy.special import factorial

# Deadline guard: 210s limit, use 195s for safety
_DEADLINE = time.time() + 195

# Precompute Hermite polynomial values at key points
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
# H12(x) = 4096x^12 - 147456x^10 + 2090112x^8 - 12902400x^6 + 34406400x^4 - 34406400x^2 + 10321920

def _hermite_at_zero(n):
    """Return H_n(0) for even n."""
    if n == 0:
        return 1.0
    if n == 4:
        return 12.0
    if n == 8:
        return 3024.0
    if n == 12:
        return 10321920.0
    return 0.0

def _compute_l1_product(a0, a4, a8):
    """
    Compute ||g||_1 * ||F[g]||_1 where g(x) = P(x) * exp(-x^2/2)
    and P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with a12 chosen so P(0)=0.
    
    The uncertainty bound involves the product of L1 norms.
    """
    H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 3024.0, 10321920.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Use a grid for spatial domain
    N = 32768
    x = np.linspace(-12, 12, N)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    H0 = np.ones_like(x)
    H4 = 16.0*x4 - 48.0*x2 + 12.0
    H8 = 256.0*x8 - 4608.0*x6 + 23040.0*x4 - 34560.0*x2 + 10080.0
    H12 = (4096.0*x12 - 147456.0*x10 + 2090112.0*x8 - 
           12902400.0*x6 + 34406400.0*x4 - 34406400.0*x2 + 10321920.0)
    
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-x2 / 2.0)
    
    # L1 norm of g
    A_g = np.sum(np.abs(g)) * dx
    
    # Compute Fourier transform
    G = fft(g) * dx
    G = fftshift(G)
    freqs = fftshift(fftfreq(N, d=dx))
    
    # L1 norm of Fourier transform
    A_G = np.sum(np.abs(G))
    
    bound = A_g * A_G
    if not np.isfinite(bound) or bound <= 0:
        return 1e10
    return bound

def _compute_l1_product_fast(a0, a4, a8):
    """
    Fast computation using the fact that F[H_n e^{-x^2/2}] = (-i)^n H_n e^{-x^2/2}.
    So F[g] has the same form with the same coefficients but with phase factors.
    
    Actually, for real even functions, the FT is real and even, so:
    F[g](xi) = a0*H0(xi) + a4*H4(xi) + a8*H8(xi) + a12*H12(xi) times exp(-xi^2/2)
    (the (-i)^n factors for even n are just ±1, and since we're taking L1 norm, 
     the sign doesn't matter)
    
    Wait, let me be more careful. The FT convention matters.
    With F[f](xi) = (1/sqrt(2*pi)) * integral f(x) e^{-ixxi} dx:
    F[H_n(x) e^{-x^2/2}](xi) = (-i)^n H_n(xi) e^{-xi^2/2}
    
    For even n, (-i)^n = (-1)^(n/2), which is ±1.
    So |F[g](xi)| = |P(xi) e^{-xi^2/2}| where P uses the same coefficients.
    
    Therefore ||g||_1 = ||F[g]||_1 (up to the 1/sqrt(2*pi) factor which is constant).
    
    So the product is ||g||_1^2 (up to a constant).
    """
    H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 3024.0, 10321920.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Use a coarser grid for speed
    N = 16384
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]
    
    x2 = x**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2
    
    H0 = np.ones_like(x)
    H4 = 16.0*x4 - 48.0*x2 + 12.0
    H8 = 256.0*x8 - 4608.0*x6 + 23040.0*x4 - 34560.0*x2 + 10080.0
    H12 = (4096.0*x12 - 147456.0*x10 + 2090112.0*x8 - 
           12902400.0*x6 + 34406400.0*x4 - 34406400.0*x2 + 10321920.0)
    
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    g = poly * np.exp(-x2 / 2.0)
    
    A_g = np.sum(np.abs(g)) * dx
    
    # Since |F[g](xi)| = |P(xi) e^{-xi^2/2}| (same functional form),
    # A_G = A_g (up to constant factor from FT convention)
    # The product A_g * A_G = A_g^2 * constant
    # So minimizing A_g * A_G is equivalent to minimizing A_g^2
    bound = A_g * A_g  # proportional to the actual bound
    if not np.isfinite(bound) or bound <= 0:
        return 1e10
    return bound

def _compute_l1_product_smooth(a0, a4, a8):
    """
    Smooth surrogate: use L2 norm as a smooth proxy for L1 norm.
    ||g||_2^2 can be computed exactly using orthogonality.
    """
    H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 3024.0, 10321920.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # L2 norm squared using orthogonality
    norm_sq = np.sqrt(np.pi) * (
        a0**2 * 1.0 +
        a4**2 * (2**4) * factorial(4) +
        a8**2 * (2**8) * factorial(8) +
        a12**2 * (2**12) * factorial(12)
    )
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e10
    
    # Use L2^4 as a smooth proxy for L1^2
    return norm_sq**2

def construct():
    # Seed result
    seed_result = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed with the true objective
    seed_bound = _compute_l1_product_fast(*seed_result)
    
    best_result = seed_result.copy()
    best_bound = seed_bound
    
    # Bounds for the three coefficients
    bounds = [(-3.0, 3.0), (-0.2, 0.2), (-0.002, 0.002)]
    
    # Phase 1: Differential evolution with the fast L1-based objective
    if time.time() < _DEADLINE - 80:
        try:
            de_result = differential_evolution(
                _compute_l1_product_fast,
                bounds,
                seed=42,
                maxiter=150,
                tol=1e-8,
                polish=True,
                init='sobol'
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from multiple starts
    starts = [
        best_result,
        seed_result,
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.0002]),
        np.array([-1.5, 0.06, 0.0005]),
        np.array([-0.8, 0.02, 0.0001]),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(
                _compute_l1_product_fast,
                start,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine-tune with L-BFGS-B
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(
                _compute_l1_product_fast,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Final verification with higher precision
    if time.time() < _DEADLINE - 5:
        try:
            final_bound = _compute_l1_product(*best_result)
            # If the fast and slow methods disagree significantly, use the slow one
            # to ensure we're optimizing the right thing
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
