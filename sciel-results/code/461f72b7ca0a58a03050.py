"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomials (physicists' convention)
def H0(x): return np.ones_like(x)
def H4(x): return x**4 - 6*x**2 + 3
def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
def H12(x): return x**12 - 60*x**10 + 1260*x**8 - 7560*x**6 + 15120*x**4 - 10080*x**2 + 945

def compute_C4(a0, a4, a8):
    """Compute the upper bound C4 = ||g||_1 * ||g_hat||_1 for the given coefficients."""
    # a12 is determined by g(0) = 0: a0 + 3*a4 + 105*a8 + 945*a12 = 0
    a12 = -(a0 + 3*a4 + 105*a8) / 945.0
    
    # Numerical integration for ||g||_1 and ||g_hat||_1
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    # g_hat(xi) = (a0*H0 + a4*(-1)^4*H4 + a8*(-1)^8*H8 + a12*(-1)^12*H12) * exp(-xi^2/2)
    #           = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-xi^2/2)  [since (-1)^even = 1]
    # Wait, need to be careful with Fourier transform convention.
    # For the standard L2 Fourier transform: g_hat(xi) = (1/sqrt(2*pi)) * int g(x) exp(-i x xi) dx
    # But for the uncertainty product, we typically use the unitary convention.
    # Actually, for Hermite functions, the Fourier transform of H_k(x) e^{-x^2/2} is (-1)^k H_k(x) e^{-x^2/2}
    # under the convention where F[f](xi) = (1/sqrt(2*pi)) int f(x) e^{-ixxi} dx
    # So g_hat(xi) = (a0*H0(xi) + a4*H4(xi) + a8*H8(xi) + a12*H12(xi)) * exp(-xi^2/2) / sqrt(2*pi)
    # Hmm, but the constant factor matters. Let me think about what C4 actually is.
    
    # For the uncertainty inequality, C = ||f||_1 * ||f_hat||_1 >= some constant
    # The best constant for dimension 1 is known. We want to minimize this product.
    
    # Let's use a reasonable grid for numerical integration
    x = np.linspace(-12, 12, 20000)
    dx = x[1] - x[0]
    
    gauss = np.exp(-x**2 / 2)
    g = (a0 * H0(x) + a4 * H4(x) + a8 * H8(x) + a12 * H12(x)) * gauss
    g_hat = (a0 * H0(x) + a4 * H4(x) + a8 * H8(x) + a12 * H12(x)) * gauss  # same due to even powers
    
    # Actually, for even k, (-1)^k = 1, so g_hat has the same polynomial part
    # But we need the 1/sqrt(2*pi) factor for the unitary Fourier transform
    g_hat = g_hat / np.sqrt(2 * np.pi)
    
    norm_g = np.sum(np.abs(g)) * dx
    norm_g_hat = np.sum(np.abs(g_hat)) * dx
    
    return norm_g * norm_g_hat

def construct():
    start_time = time.time()
    deadline = 210.0  # evaluator kills at 210s, use 200s as safety margin
    
    # Seed values
    a0_seed, a4_seed, a8_seed = -113/100, 1/25, 1/3240
    best_coeffs = np.array([a0_seed, a4_seed, a8_seed])
    best_C4 = compute_C4(a0_seed, a4_seed, a8_seed)
    
    elapsed = time.time() - start_time
    if elapsed > 200.0:
        return best_coeffs
    
    # Multi-start optimization
    np.random.seed(42)
    
    # Define objective function for scipy
    def objective(params):
        a0, a4, a8 = params
        try:
            return compute_C4(a0, a4, a8)
        except:
            return 1e10
    
    # Try multiple starting points
    starts = [
        [a0_seed, a4_seed, a8_seed],
        [-1.0, 0.05, 0.001],
        [-1.2, 0.03, 0.0005],
        [-0.8, 0.06, 0.002],
        [-1.5, 0.02, 0.0003],
        [-1.13, 0.04, 0.0008],
    ]
    
    for i, x0 in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > 200.0:
            break
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x.copy()
        except:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > 200.0:
            break
    
    # Final polish with tighter tolerances
    elapsed = time.time() - start_time
    if elapsed < 195.0:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_C4:
                best_C4 = result.fun
                best_coeffs = result.x.copy()
        except:
            pass
    
    elapsed = time.time() - start_time
    if elapsed > 200.0:
        return best_coeffs
    
    return best_coeffs
# EVOLVE-BLOCK-END
