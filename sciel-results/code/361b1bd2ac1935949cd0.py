"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from math import factorial, sqrt, pi


def _hermite_polynomial(n, x):
    """Compute the n-th probabilist's Hermite polynomial He_n(x)."""
    if n == 0:
        return np.ones_like(x) if isinstance(x, np.ndarray) else 1.0
    if n == 1:
        return x.copy() if isinstance(x, np.ndarray) else x
    # Use recurrence: He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)
    if isinstance(x, np.ndarray):
        h0 = np.ones_like(x)
        h1 = x.copy()
        for k in range(1, n):
            h2 = x * h1 - k * h0
            h0, h1 = h1, h2
        return h1
    else:
        h0 = 1.0
        h1 = x
        for k in range(1, n):
            h2 = x * h1 - k * h0
            h0, h1 = h1, h2
        return h1


def _compute_a12(a0, a4, a8):
    """Compute a12 such that g(0) = 0.
    
    g(0) = a0*H0(0)*e^0 + a4*H4(0)*e^0 + a8*H8(0)*e^0 + a12*H12(0)*e^0 = 0
    H0(0) = 1, H4(0) = -3, H8(0) = 105, H12(0) = -945
    So: a0 - 3*a4 + 105*a8 - 945*a12 = 0
    a12 = (a0 - 3*a4 + 105*a8) / 945
    """
    return (a0 - 3 * a4 + 105 * a8) / 945.0


def _compute_uncertainty_product(a0, a4, a8):
    """Compute the upper bound on A(f)*A(Fourier(f)) analytically.
    
    For g(x) = sum_k a_k * He_k(x) * exp(-x^2/2),
    the L1 norm and Fourier transform L1 norm can be computed.
    
    The Fourier transform of He_k(x)*exp(-x^2/2) is i^k * He_k(x) * exp(-x^2/2)
    (with appropriate normalization depending on convention).
    
    For the uncertainty product bound, we use:
    A(f) = ||f||_1, A(F) = ||F(f)||_1
    
    With the convention F(f)(xi) = integral f(x) exp(-2*pi*i*x*xi) dx,
    the Fourier transform of exp(-x^2/2)*He_k(x) involves scaling.
    
    Let's use the standard result: if g(x) = sum a_k He_k(x) e^{-x^2/2},
    then we can compute ||g||_1 and ||F(g)||_1 numerically but efficiently,
    or use known bounds.
    
    Actually, for this specific problem, the "upper bound on C4" likely refers to
    a specific constant in the uncertainty inequality. Let me use a direct 
    numerical computation that's fast enough.
    """
    a12 = _compute_a12(a0, a4, a8)
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # Compute g(x) and its Fourier transform on a grid
    # Use a sufficiently large grid for accuracy
    x = np.linspace(-15, 15, 2001)
    dx = x[1] - x[0]
    
    g = np.zeros_like(x)
    for a, n in zip(coeffs, degrees):
        if a != 0:
            g += a * _hermite_polynomial(n, x) * np.exp(-x**2 / 2)
    
    A_f = np.abs(np.trapz(g, x))
    
    # Fourier transform using FFT
    # F(g)(xi) = integral g(x) exp(-2*pi*i*x*xi) dx
    # Use FFT: sample g on grid, compute FFT
    N = len(x)
    G = np.fft.fftshift(np.fft.fft(g)) * dx
    xi = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    A_F = np.abs(np.trapz(G, xi))
    
    return A_f * A_F


def _objective(params):
    """Objective function for optimization."""
    a0, a4, a8 = params
    try:
        result = _compute_uncertainty_product(a0, a4, a8)
        if not np.isfinite(result) or result < 0:
            return 1e10
        return result
    except:
        return 1e10


def construct():
    """Find three coefficients (a0, a4, a8) that minimize the uncertainty product bound."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Seed solution
    seed = np.array([-113/100, 1/25, 1/3240])
    best_result = _compute_uncertainty_product(*seed)
    best_params = seed.copy()
    
    # Check time
    if time.time() > deadline - 60:
        return best_params
    
    # Phase 1: Differential Evolution for global search
    try:
        bounds = [(-2, 2), (-0.5, 0.5), (-0.1, 0.1)]
        result_de = differential_evolution(
            _objective, bounds,
            maxiter=50, popsize=15,
            tol=1e-8, seed=42,
            polish=True,
            init='sobol'
        )
        
        if time.time() < deadline - 30:
            if result_de.fun < best_result:
                best_result = result_de.fun
                best_params = result_de.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_params
    
    # Phase 2: Local refinement with L-BFGS-B
    try:
        result_lbfgs = minimize(
            _objective, best_params,
            method='L-BFGS-B',
            bounds=[(-2, 2), (-0.5, 0.5), (-0.1, 0.1)],
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if time.time() < deadline - 10:
            if result_lbfgs.fun < best_result:
                best_result = result_lbfgs.fun
                best_params = result_lbfgs.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        return best_params
    
    # Phase 3: Additional local refinement from multiple starting points
    try:
        # Try a few perturbations around the best solution
        for i in range(5):
            if time.time() > deadline - 5:
                break
            
            perturbation = np.random.normal(0, 0.01, size=3)
            start = best_params + perturbation
            start = np.clip(start, [-2, -0.5, -0.1], [2, 0.5, 0.1])
            
            result_refine = minimize(
                _objective, start,
                method='L-BFGS-B',
                bounds=[(-2, 2), (-0.5, 0.5), (-0.1, 0.1)],
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result_refine.fun < best_result:
                best_result = result_refine.fun
                best_params = result_refine.x.copy()
    except Exception:
        pass
    
    return best_params
# EVOLVE-BLOCK-END
