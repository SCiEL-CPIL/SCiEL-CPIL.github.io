"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use finer grid for accuracy
    x = np.linspace(-15, 15, 8000)
    dx = x[1] - x[0]
    
    # Hermite polynomials at x
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
            
            # Compute spatial norms using trapezoidal rule
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-12:
                return 1e10
                
            norm4 = np.trapz(np.abs(g)**4, x)
            norm4_val = norm4 ** 0.25
            
            # Compute Fourier transform analytically
            # F[H_n(x) e^{-x^2/2}] = i^n sqrt(2*pi) H_n(xi) e^{-xi^2/2}
            # We assume the standard non-unitary FT: F(f)(xi) = integral f(x) exp(-i x xi) dx
            # The grid x is symmetric, so we can evaluate H_n at x and use it for xi as well
            # since the integration domain is the same and the function form is identical (just multiplied by constants)
            
            # Coefficients for the Fourier side
            # c0 -> c0 * i^0 * sqrt(2*pi) * H0(xi) * exp(-xi^2/2)
            # c4 -> c4 * i^4 * sqrt(2*pi) * H4(xi) * exp(-xi^2/2) = c4 * 1 * sqrt(2*pi) * H4(xi) * exp(-xi^2/2)
            # c8 -> c8 * i^8 * sqrt(2*pi) * H8(xi) * exp(-xi^2/2) = c8 * 1 * sqrt(2*pi) * H8(xi) * exp(-xi^2/2)
            # c12 -> c12 * i^12 * sqrt(2*pi) * H12(xi) * exp(-xi^2/2) = c12 * 1 * sqrt(2*pi) * H12(xi) * exp(-xi^2/2)
            
            sqrt_2pi = np.sqrt(2 * np.pi)
            g_hat = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * sqrt_2pi * np.exp(-0.5 * x**2)
            
            # Note: The grid x is used for both spatial and frequency domain.
            # This is valid because the basis functions are the same form, and the grid is symmetric enough.
            
            norm2_hat_sq = np.trapz(np.abs(g_hat)**2, x)
            norm4_hat = np.trapz(np.abs(g_hat)**4, x)
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Use smooth sign: sign = tanh(s) scaled to avoid zero
            # Actually, for DE we can use direct sign via threshold
            sign0 = 1.0 if s0 > 0 else -1.0
            sign4 = 1.0 if s4 > 0 else -1.0
            sign8 = 1.0 if s8 > 0 else -1.0
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -10 to 3 (coefficients from ~0.000045 to ~20)
    # This range is necessary to capture the small magnitude of c8 in the seed (~1e-3, log ~ -6.9)
    # s_i: sign parameter, range from -1 to 1
    bounds = [(-10, 3), (-10, 3), (-10, 3), (-1, 1), (-1, 1), (-1, 1)]
    
    # Run differential evolution with multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    
    for seed in [42, 123, 456]:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=seed, maxiter=200, popsize=25,
            tol=1e-8, polish=False,
            init='sobol',
            mutation=(0.5, 1.0)
        )
        
        if result.fun < best_ratio:
            best_ratio = result.fun
            best_params = result.x
    
    if best_params is not None:
        m0, m4, m8, s0, s4, s8 = best_params
        sign0 = 1.0 if s0 > 0 else -1.0
        sign4 = 1.0 if s4 > 0 else -1.0
        sign8 = 1.0 if s8 > 0 else -1.0
        c0 = sign0 * np.exp(m0)
        c4 = sign4 * np.exp(m4)
        c8 = sign8 * np.exp(m8)
        
        # Polish with local optimization using direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
        
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
