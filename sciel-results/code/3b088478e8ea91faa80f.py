"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Hermite polynomials (physicists' convention) H_0, H_4, H_8, H_12
def _H0(x):
    return np.ones_like(x, dtype=float)

def _H4(x):
    return 16*x**4 - 48*x**2 + 12

def _H8(x):
    return 256*x**8 - 1920*x**6 + 4320*x**4 - 3360*x**2 + 420

def _H12(x):
    return 4096*x**12 - 46080*x**10 + 230400*x**8 - 504000*x**6 + 504000*x**4 - 168000*x**2 + 10395

def _gaussian(x):
    return np.exp(-0.5 * x**2)

def _g(x, c0, c4, c8):
    """
    Construct g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    g(0) = (c0 + 12*c4 + 420*c8 + 10395*c12) * 1 = 0
    => c12 = -(c0 + 12*c4 + 420*c8) / 10395
    """
    c12 = -(c0 + 12*c4 + 420*c8) / 10395.0
    poly = c0*_H0(x) + c4*_H4(x) + c8*_H8(x) + c12*_H12(x)
    return poly * _gaussian(x)

def _compute_upper_bound_C4(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(F) for the test function g.
    
    A(f) = sup |f(x)|, and A(F) = sup |F(f)(k)|.
    The uncertainty inequality bound is A(f)*A(F) <= C4 * ||f||_2 * ||F(f)||_2.
    Since ||F(f)||_2 = ||f||_2 (Plancherel), we have A(f)*A(F) <= C4 * ||f||_2^2.
    
    For our specific g, we need to compute:
    A(g) = sup |g(x)|
    A(F(g)) = sup |F(g)(k)|
    
    The bound C4 is such that A(g)*A(F(g)) <= C4 * ||g||_2^2.
    We want to minimize this upper bound.
    
    Actually, re-reading the task: "score is the upper bound on C4".
    The Fourier transform of a Hermite-Gaussian is another Hermite-Gaussian.
    For g(x) = sum c_n H_n(x) exp(-x^2/2), 
    F(g)(k) = sum (-i)^n c_n H_n(k) exp(-k^2/2) (up to normalization).
    
    Let's use the convention where F(f)(k) = 1/sqrt(2*pi) * integral f(x) exp(-ikx) dx.
    For f(x) = H_n(x) exp(-x^2/2), F(f)(k) = (-i)^n H_n(k) exp(-k^2/2).
    
    So F(g)(k) = sum c_n (-i)^n H_n(k) exp(-k^2/2).
    
    We need to minimize max_x |g(x)| * max_k |F(g)(k)| / ||g||_2^2.
    But the task says "upper bound on C4" where the inequality is A(f)*A(F) <= C4 * ||f||_2^2.
    So C4 = A(g)*A(F(g)) / ||g||_2^2.
    We want to minimize this.
    """
    c12 = -(c0 + 12*c4 + 420*c8) / 10395.0
    
    # Sample on a grid
    x = np.linspace(-10, 10, 20000)
    g_vals = _g(x, c0, c4, c8)
    A_g = np.max(np.abs(g_vals))
    
    # Fourier transform: F(g)(k) = sum c_n (-i)^n H_n(k) exp(-k^2/2)
    # For even n, (-i)^n is real: n=0 -> 1, n=4 -> 1, n=8 -> 1, n=12 -> 1
    # Since all our n are even and divisible by 4, (-i)^n = 1 for all.
    # Wait: (-i)^0 = 1, (-i)^4 = (-i)^2 * (-i)^2 = (-1)*(-1) = 1, etc.
    # So F(g)(k) = (c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)) * exp(-k^2/2)
    # This is the same form as g but with the same coefficients!
    # Actually, let me double-check. The Fourier transform of H_n(x) e^{-x^2/2} is (-i)^n H_n(x) e^{-x^2/2}.
    # For n even: (-i)^n = (-1)^{n/2}.
    # n=0: (-1)^0 = 1
    # n=4: (-1)^2 = 1
    # n=8: (-1)^4 = 1
    # n=12: (-1)^6 = 1
    # So yes, F(g)(k) has the same form with the same coefficients.
    
    Fg_vals = _g(x, c0, c4, c8)  # Same function due to symmetry
    A_Fg = np.max(np.abs(Fg_vals))
    
    # ||g||_2^2 = integral |g(x)|^2 dx
    norm_sq = np.trapz(np.abs(g_vals)**2, x)
    
    if norm_sq < 1e-12:
        return np.inf
    
    C4 = A_g * A_Fg / norm_sq
    return C4

def _objective(coeffs):
    """Objective function to minimize: the upper bound C4."""
    c0, c4, c8 = coeffs
    try:
        val = _compute_upper_bound_C4(c0, c4, c8)
        return val
    except:
        return 1e10

def construct():
    # Start with the prior state of the art
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_upper_bound_C4(*best_coeffs)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 180  # 180 seconds
    
    # Try local optimization from the starting point
    try:
        res = minimize(_objective, best_coeffs, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_val:
            best_coeffs = res.x
            best_val = res.fun
    except:
        pass
    
    if time.time() > deadline - 30:
        return best_coeffs
    
    # Try differential evolution for global search
    try:
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        de_res = differential_evolution(_objective, bounds, maxiter=100, popsize=15, 
                                        tol=1e-8, seed=42, polish=False)
        if de_res.fun < best_val:
            best_coeffs = de_res.x
            best_val = de_res.fun
    except:
        pass
    
    if time.time() > deadline - 15:
        return best_coeffs
    
    # Final polish with tighter tolerance
    try:
        res = minimize(_objective, best_coeffs, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < best_val:
            best_coeffs = res.x
            best_val = res.fun
    except:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
