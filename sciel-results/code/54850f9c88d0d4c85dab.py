"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite_coeffs(n):
    """Return coefficients of H_n(x) as a list from highest degree to constant."""
    # Use scipy to generate Hermite polynomial coefficients
    from scipy.special import eval_hermite
    # We need the actual polynomial coefficients, not just evaluation
    # Let's build them recursively or use a known formula
    # For now, use a lookup for small n
    hermite_coeffs = {
        0: [1],
        4: [1, 0, -6, 0, 3],  # x^4 - 6x^2 + 3
        8: [1, 0, -28, 0, 210, 0, -420, 0, 105],  # x^8 - 28x^6 + 210x^4 - 420x^2 + 105
        12: [1, 0, -60, 0, 1260, 0, -12600, 0, 56700, 0, -113400, 0, 10395]
    }
    return np.array(hermite_coeffs[n], dtype=float)

def _eval_hermite_poly(coeffs, x):
    """Evaluate polynomial with given coefficients at x."""
    return np.polyval(coeffs, x)

def _compute_g(x, a0, a1, a2):
    """Compute g(x) = (a0*H0 + a1*H4 + a2*H8 + a3*H12) * exp(-x^2/2)"""
    h0 = np.ones_like(x)
    h4 = _eval_hermite_poly(_hermite_coeffs(4), x)
    h8 = _eval_hermite_poly(_hermite_coeffs(8), x)
    h12 = _eval_hermite_poly(_hermite_coeffs(12), x)
    
    # Compute a3 from g(0) = 0
    # g(0) = (a0*1 + a1*3 + a2*105 + a3*10395) * 1 = 0
    a3 = -(a0 + 3*a1 + 105*a2) / 10395.0
    
    poly_part = a0 * h0 + a1 * h4 + a2 * h8 + a3 * h12
    gaussian_part = np.exp(-x**2 / 2)
    return poly_part * gaussian_part

def _compute_fft_upper_bound(a0, a1, a2):
    """Compute the upper bound on A(f)*A(F(f)) for the given coefficients."""
    # We need to compute the L∞ norm of g and its Fourier transform
    # For the purpose of this optimization, we'll approximate
    # by sampling on a grid
    
    # Use a fine grid to find the maximum of |g(x)|
    x = np.linspace(-10, 10, 10001)
    g_vals = _compute_g(x, a0, a1, a2)
    max_g = np.max(np.abs(g_vals))
    
    # The Fourier transform of a Hermite-Gaussian function can be computed
    # For now, approximate by FFT (this is a simplification)
    # In a proper implementation, we'd use the analytical Fourier transform
    
    # For a more accurate approach, note that the Fourier transform of
    # H_n(x) * exp(-x^2/2) is (-i)^n * H_n(x) * exp(-x^2/2)
    # So |F(g)| has the same form as |g| up to phase factors
    
    # The key insight: for these functions, the L∞ norm of g and F(g) 
    # are related. Let's compute both on the same grid.
    
    # Actually, let's be more careful. The Fourier transform of 
    # exp(-x^2/2) * p(x) where p is a polynomial can be computed.
    # For now, use the fact that the problem is symmetric and the 
    # upper bound is likely proportional to max_g * max_F_g
    
    # Simplified: assume the upper bound is max_g^2 (this may not be exact)
    # But we need a better model
    
    # Let's compute the actual Fourier transform numerically
    dx = x[1] - x[0]
    G = np.fft.fftshift(np.fft.fft(g_vals)) * dx
    max_F_g = np.max(np.abs(G))
    
    return max_g * max_F_g

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    # Initial guess from prior work
    x0 = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    best_x = x0.copy()
    best_score = _compute_fft_upper_bound(*x0)
    
    # Try multiple optimization methods
    methods = [
        ('Nelder-Mead', {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}),
        ('BFGS', {'maxiter': 2000}),
        ('L-BFGS-B', {'maxiter': 2000}),
    ]
    
    for method, options in methods:
        if time.time() > deadline - 30:  # Stop if running out of time
            break
            
        try:
            result = minimize(
                _compute_fft_upper_bound,
                x0,
                method=method,
                options=options
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_x = result.x
        except Exception:
            continue
    
    # Try differential evolution for global optimization
    if time.time() < deadline - 60:
        try:
            bounds = [(-10, 10), (-10, 10), (-10, 10)]
            de_result = differential_evolution(
                _compute_fft_upper_bound,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                seed=42
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_x = de_result.x
        except Exception:
            pass
    
    # Final check and return
    if time.time() > deadline - 5:
        # Return the best we have
        return best_x
    
    # One more local refinement
    try:
        final_result = minimize(
            _compute_fft_upper_bound,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        if final_result.success and final_result.fun < best_score:
            best_x = final_result.x
            best_score = final_result.fun
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
