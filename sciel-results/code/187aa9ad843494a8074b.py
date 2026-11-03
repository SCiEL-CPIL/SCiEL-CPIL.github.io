"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hermite_powers(x, n):
    """
    Compute powers of Hermite polynomials H_n(x) for n in {0, 4, 8, 12}.
    These are used in the test function g(x) = a0*H0^2 + a4*H4^2 + a8*H8^2 + a12*H12^2.
    
    Note: The task says "Hermite-polynomial times Gaussian test function".
    The standard uncertainty principle for Fourier transforms involves 
    A(f) = sup |f| and A(F) = sup |F|.
    The bound is A(f) * A(F) >= C.
    
    The test function is g(x) = sum a_i H_{2i}(x) e^{-x^2/2} or similar.
    However, the coefficients multiply H0, H4, H8.
    
    Let's look at the seed:
    return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    This suggests g(x) might be related to a polynomial times a Gaussian.
    The condition P(0)=0 fixes the H12 coefficient.
    
    Let's assume the test function is of the form:
    g(x) = (a0 + a4 H_4(x) + a8 H_8(x) + a12 H_12(x)) * exp(-x^2/2)
    
    But wait, the task says "Hermite-polynomial times Gaussian test function".
    And "Three free coefficients multiply H0,H4,H8".
    So g(x) = (a0 H_0(x) + a4 H_4(x) + a8 H_8(x) + a12 H_12(x)) * exp(-x^2/2)
    
    The Fourier transform of H_n(x) exp(-x^2/2) is (-i)^n H_n(x) exp(-x^2/2).
    So F(g) = (a0 H_0(x) + a4 (-i)^4 H_4(x) + a8 (-i)^8 H_8(x) + a12 (-i)^12 H_12(x)) * exp(-x^2/2)
    = (a0 H_0(x) + a4 H_4(x) + a8 H_8(x) + a12 H_12(x)) * exp(-x^2/2)
    So g is an eigenfunction of the Fourier transform (up to a phase).
    
    The upper bound on A(f)*A(F) is sup|g| * sup|F(g)| = (sup|g|)^2.
    
    We want to minimize (sup|g|)^2 subject to P(0)=0.
    P(0)=0 means g(0)=0? Or the polynomial part is 0 at x=0?
    "impose P(0)=0" likely means the polynomial part P(x) = a0 H_0(x) + a4 H_4(x) + a8 H_8(x) + a12 H_12(x) satisfies P(0)=0.
    
    Let's define H_n(x) as physicists' Hermite polynomials:
    H_0(x) = 1
    H_4(x) = 16x^4 - 48x^2 + 12
    H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    H_12(x) = 4096x^12 - 98304x^10 + 944640x^8 - 4644480x^6 + 9676800x^4 - 7654080x^2 + 1034880
    
    P(0) = a0 * 1 + a4 * 12 + a8 * 1680 + a12 * 1034880 = 0
    So a12 = -(a0 + 12 a4 + 1680 a8) / 1034880
    
    We want to minimize (sup |P(x) exp(-x^2/2)|)^2.
    Let M(a0, a4, a8) = sup_x |P(x) exp(-x^2/2)|.
    We want to minimize M(a0, a4, a8)^2.
    
    The seed returns [-113/100, 1/25, 1/3240].
    Let's check if this is optimal or just a starting point.
    The task says "Tune ... to reduce the upper bound".
    The score is the upper bound on C4.
    Lower is better.
    """
    x = np.linspace(-10, 10, 20000)
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    H12 = 4096*x**12 - 98304*x**10 + 944640*x**8 - 4644480*x**6 + 9676800*x**4 - 7654080*x**2 + 1034880
    return H0, H4, H8, H12, x

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound (sup|g|)^2 for given coefficients.
    """
    H0, H4, H8, H12, x = _hermite_powers(x, 12)
    a12 = -(a0 + 12*a4 + 1680*a8) / 1034880
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = P * np.exp(-x**2/2)
    return np.max(np.abs(g))**2

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Keep a safety margin
    
    # Initial best
    a0, a4, a8 = -113/100, 1/25, 1/3240
    best_bound = _compute_bound(a0, a4, a8)
    best_coeffs = np.array([a0, a4, a8])
    
    # Objective function for optimization
    def objective(coeffs):
        a0, a4, a8 = coeffs
        return _compute_bound(a0, a4, a8)
    
    # Use differential evolution for global optimization
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    try:
        result = differential_evolution(objective, bounds, maxiter=100, tol=1e-8, seed=42)
        if result.fun < best_bound:
            best_bound = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    # Refine with local optimization
    try:
        result_local = minimize(objective, best_coeffs, method='Nelder-Mead', 
                                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        if result_local.fun < best_bound:
            best_bound = result_local.fun
            best_coeffs = result_local.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_coeffs
    
    # Try a few more random starts
    rng = np.random.default_rng(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        random_coeffs = rng.uniform(-10, 10, 3)
        try:
            result_local = minimize(objective, random_coeffs, method='Nelder-Mead', 
                                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result_local.fun < best_bound:
                best_bound = result_local.fun
                best_coeffs = result_local.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
