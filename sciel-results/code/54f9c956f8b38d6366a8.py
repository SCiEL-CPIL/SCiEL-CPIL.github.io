"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_start_time = time.time()
_DEADLINE = 180.0  # safety margin before 210s kill

def _hermite_probabilists(n, x):
    """Compute probabilists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x.copy()
    else:
        h0 = np.ones_like(x)
        h1 = x.copy()
        for k in range(1, n):
            h2 = x * h1 - k * h0
            h0, h1 = h1, h2
        return h1

def _compute_coefficients(a0, a4, a8):
    """Return the full coefficient array [a0, a4, a8, a12] with g(0)=0."""
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 10395.0
    return np.array([a0, a4, a8, a12])

def _evaluate_g(coeffs, x):
    """Evaluate g(x) = sum of a_i * H_{2i}(x) * exp(-x^2/4) for i=0,2,4,6 (degrees 0,4,8,12)."""
    a0, a4, a8, a12 = coeffs
    gauss = np.exp(-x**2 / 4.0)
    H0 = _hermite_probabilists(0, x)
    H4 = _hermite_probabilists(4, x)
    H8 = _hermite_probabilists(8, x)
    H12 = _hermite_probabilists(12, x)
    return (a0*H0 + a4*H4 + a8*H8 + a12*H12) * gauss

def _objective(params):
    """Compute the upper bound C4 for given (a0, a4, a8)."""
    if time.time() - _start_time > _DEADLINE:
        return 1e10  # penalty for timeout
    
    a0, a4, a8 = params
    coeffs = _compute_coefficients(a0, a4, a8)
    
    # Use numerical integration on a fine grid
    x = np.linspace(-15, 15, 20000)
    g = _evaluate_g(coeffs, x)
    
    # Compute norms and moments
    # A(f) = ||f||_2^2, A(F(f)) = ||F(f)||_2^2
    # For the uncertainty bound, we need to compute the relevant quantity
    
    # The Fourier transform of H_n(x)*e^{-x^2/4} is related to H_n(ξ)*e^{-ξ^2/4}
    # Specifically, F[H_n(x)e^{-x^2/4}](ξ) = i^n * H_n(ξ) * e^{-ξ^2/4} (up to normalization)
    
    # Let's compute the actual L2 norm and the weighted norm
    norm_sq = np.trapz(g**2, x)
    
    if norm_sq < 1e-15:
        return 1e10
    
    # For the uncertainty principle with fourth moments:
    # The bound involves ||x^2 g||_2^2 / ||g||_2^2 type quantities
    # Or it could be the product of variances
    
    # Let me compute multiple candidate objectives and see which makes sense
    # The "upper bound on A(f)*A(Fourier(f))" with test function approach
    
    # Actually, for the specific uncertainty inequality being studied, 
    # C4 is likely computed as a ratio involving the L2 norm and 
    # the norm of the function multiplied by a weight.
    
    # Let's compute: (∫|g|²) * (∫|x²g|²) / (∫|xg|²)² or similar
    
    x2_g = x**2 * g
    x_g = x * g
    
    norm_x2 = np.trapz(x2_g**2, x)
    norm_x = np.trapz(x_g**2, x)
    
    # Candidate 1: product of second moments
    if norm_x < 1e-15 or norm_x2 < 1e-15:
        return 1e10
    
    # The uncertainty product for the test function
    # A(f) * A(F(f)) where A involves moments
    
    # Let's try: the bound is proportional to (∫x⁴|g|²)(∫ξ⁴|ĝ|²) / (∫|g|²)²
    # But by Plancherel and symmetry, for these Hermite-Gaussian functions,
    # the Fourier transform has similar structure.
    
    # Compute Fourier transform numerically
    from scipy.fft import fft, fftshift, fftfreq
    N = len(x)
    dx = x[1] - x[0]
    x_center = x - x[N//2]
    g_shifted = fftshift(g)
    G = fft(g_shifted) * dx
    freqs = fftfreq(N, d=dx)
    freqs = fftshift(freqs)
    
    norm_G = np.trapz(np.abs(G)**2, freqs)
    
    # The upper bound C4 for the uncertainty inequality
    # In many formulations, it's the ratio:
    # C4 = (∫x⁴|g|² dx) * (∫ξ⁴|ĝ|² dξ) / (∫|g|² dx)²
    
    x4_g = x**4 * g
    norm_x4 = np.trapz(np.abs(x4_g)**2, x)
    
    xi4_G = freqs**4 * G
    norm_xi4 = np.trapz(np.abs(xi4_G)**2, freqs)
    
    if norm_sq < 1e-15:
        return 1e10
    
    C4_candidate = (norm_x4 * norm_xi4) / (norm_sq**2)
    
    return C4_candidate

def construct():
    """Return three coefficients of the Hermite combination."""
    # Seed solution
    best_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _objective(best_params)
    
    if time.time() - _start_time > _DEADLINE - 10:
        return best_params
    
    # Try local optimization from seed
    try:
        res = minimize(_objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass
    
    if time.time() - _start_time > _DEADLINE - 5:
        return best_params
    
    # Try differential evolution for global search
    try:
        bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
        de_res = differential_evolution(_objective, bounds, maxiter=100, 
                                         popsize=10, tol=1e-8, 
                                         seed=42, polish=False)
        if de_res.fun < best_val:
            best_params = de_res.x
            best_val = de_res.fun
    except Exception:
        pass
    
    if time.time() - _start_time > _DEADLINE - 2:
        return best_params
    
    # Polish with high-precision local optimization
    try:
        res2 = minimize(_objective, best_params, method='BFGS',
                       options={'maxiter': 1000, 'gtol': 1e-12})
        if res2.fun < best_val:
            best_params = res2.x
            best_val = res2.fun
    except Exception:
        pass
    
    return best_params
# EVOLVE-BLOCK-END
