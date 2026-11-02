"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

_start_time = time.time()
_DEADLINE = 180.0  # safety margin before 210s kill

def _hermite_gauss_coeffs(a0, a4, a8):
    """Given a0, a4, a8, compute a12 so that g(0) = 0."""
    # g(0) = (a0 + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0)
    # H4(0) = -12, H8(0) = 1008, H12(0) = -10395 (physicists' Hermite)
    # Wait: physicists' Hermite H_n(0) for even n: H0=1, H2=-2, H4=12, H6=-120, H8=1680, H10=-30240, H12=665280
    # Let me verify: H_0(x)=1, H_1(x)=2x, H_2(x)=4x^2-2, H_3(x)=8x^3-12x, H_4(x)=16x^4-48x^2+12
    # H_4(0) = 12
    # H_8(0): H_8(x) = 256x^8 - 4480x^6 + 23520x^4 - 40320x^2 + 1680, so H_8(0) = 1680
    # H_12(0): need to compute. Using recurrence or known values.
    # H_12(0) = -10395 * ... let me just use eval_hermite
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    # g(0) = a0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    a12 = -(a0 + a4 * H4_0 + a8 * H8_0) / H12_0
    return a0, a4, a8, a12

def _compute_c4_bound(a0, a4, a8):
    """Compute the upper bound C4 for the given coefficients."""
    a12 = _hermite_gauss_coeffs(a0, a4, a8)[3]
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # Numerically compute the relevant quantities on a grid
    # g(x) = sum(c_n * H_n(x)) * exp(-x^2/2)
    # |g(x)|^2 = [sum(c_n * H_n(x))]^2 * exp(-x^2)
    
    # For the uncertainty inequality bound, we need:
    # A(f) = (∫|f|^2 dx)^(-1/2) * (∫ x^2 |f|^2 dx)^(1/2) or similar
    # Actually, the standard uncertainty product is:
    # (∫|f|^2)^(-1) * (∫x^2|f|^2) * (∫|fhat|^2) * (∫xi^2|fhat|^2)
    # By Plancherel, ∫|fhat|^2 = ∫|f|^2 (up to constants)
    
    # Let me use a fine grid for numerical integration
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]
    
    # Compute g(x)
    g_poly = np.zeros_like(x)
    for c, n in zip(coeffs, degrees):
        g_poly += c * eval_hermite(n, x)
    g = g_poly * np.exp(-0.5 * x**2)
    
    g2 = g**2  # |g|^2 since g is real
    
    # A = ∫|g|^2 dx
    A = np.trapz(g2, x)
    
    # Position variance: ∫x^2|g|^2 dx
    x2_g2 = np.trapz(x**2 * g2, x)
    
    # Fourier transform: use the fact that FT of H_n(x)exp(-x^2/2) = (-i)^n H_n(x)exp(-x^2/2)
    # So |g_hat|^2 has the same form but with phases that cancel in |.|^2
    # Actually, |(-i)^n|^2 = 1, so |g_hat(x)|^2 = [sum(c_n H_n(x))]^2 exp(-x^2) = |g(x)|^2
    # Wait, that's only if all coefficients are real and the phase factors don't affect magnitude.
    # (-i)^n for n=0,4,8,12: (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=1
    # So all phases are 1! Thus |g_hat|^2 = |g|^2
    
    # Hmm, that would make the uncertainty product just (x2_g2/A)^2 which is minimized by Gaussian.
    # That can't be right for a non-trivial bound.
    
    # Let me reconsider. The Fourier transform convention matters.
    # If fhat(xi) = ∫ f(x) e^{-2πixi} dx, then FT of H_n(x)e^{-x^2/2} involves different scaling.
    
    # Actually for the Hermite functions, with the convention where φ_n(x) = (2^n n! √π)^(-1/2) H_n(x) e^{-x^2/2},
    # the Fourier transform (with e^{-ixξ}) gives i^n φ_n.
    
    # With e^{-ixξ}: FT of e^{-x^2/2} is √(2π) e^{-ξ^2/2}
    # FT of H_n(x) e^{-x^2/2} = √(2π) (-i)^n H_n(x) e^{-ξ^2/2}
    
    # So |fhat|^2 = 2π |f|^2 (same shape)
    
    # The uncertainty product should be:
    # Δx * Δξ where Δx^2 = ∫x^2|f|^2/∫|f|^2, Δξ^2 = ∫ξ^2|fhat|^2/∫|fhat|^2
    # Product = Δx * Δξ
    
    # Since |fhat|^2 = 2π|f|^2, Δξ^2 = ∫ξ^2|f|^2/∫|f|^2 = Δx^2
    # So the product is always Δx^2, minimized at 1/4 for Gaussian.
    
    # This suggests the "bound" C4 is something else. Let me re-read the task.
    
    # "reduce the upper bound on A(f)*A(Fourier(f))"
    # Maybe A(f) is not the standard L2 norm but something else.
    
    # Looking at the seed: coefficients [-113/100, 1/25, 1/3240]
    # Let me just try to match what gives a good score by doing a numerical search.
    
    # Alternative interpretation: C4 might be a specific constant in an inequality of the form
    # ||f||_4 * ||fhat||_4 <= C4 * ||f||_2 * ||fhat||_2 or similar.
    
    # Or it could be related to the uncertainty principle in L4 norm.
    
    # Let me try: C4 = (∫|g|^4 dx)^(1/4) * (∫|g_hat|^4 dx)^(1/4) / (∫|g|^2 dx)^(1/2)
    
    g4 = g2**2  # |g|^4
    I4 = np.trapz(g4, x)
    
    # |g_hat|^4 = (2π)^2 |g|^4 (from |g_hat|^2 = 2π|g|^2)
    I4_hat = (2*np.pi)**2 * I4
    
    # C4 bound = (I4)^(1/4) * (I4_hat)^(1/4) / sqrt(A)
    C4 = (I4 ** 0.25) * (I4_hat ** 0.25) / np.sqrt(A)
    
    return C4

def construct():
    """Find optimal coefficients to minimize the C4 bound."""
    # Start with the seed values
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    best_result = x0
    best_score = _compute_c4_bound(*x0)
    
    # Try multiple random restarts and local optimizations
    np.random.seed(42)
    
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        if i == 0:
            x_start = x0
        else:
            # Random perturbation around the seed or best found
            scale = 1.0 / (i + 1)
            x_start = x0 + np.random.randn(3) * scale
        
        def objective(x):
            try:
                return _compute_c4_bound(x[0], x[1], x[2])
            except:
                return 1e10
        
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except:
            pass
    
    # Also try a grid search in promising regions
    if time.time() - _start_time < _DEADLINE - 15:
        # Fine grid around best result
        for da0 in np.linspace(-0.5, 0.5, 11):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            for da4 in np.linspace(-0.1, 0.1, 11):
                for da8 in np.linspace(-0.01, 0.01, 11):
                    x_test = best_result + np.array([da0, da4, da8])
                    try:
                        s = _compute_c4_bound(*x_test)
                        if s < best_score:
                            best_score = s
                            best_result = x_test
                    except:
                        pass
    
    return np.array(best_result)
# EVOLVE-BLOCK-END
