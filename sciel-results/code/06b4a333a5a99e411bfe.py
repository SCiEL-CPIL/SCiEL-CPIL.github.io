"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
from scipy.integrate import simpson

def construct():
    # Grid for numerical integration
    # H12 * exp(-x^2/2) decays fast. 15 is enough.
    x = np.linspace(-15, 15, 20000)
    
    # Precompute Hermite polynomials on the grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0 for constraint g(0) = 0
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # Gaussian factor
    gauss = np.exp(-0.5 * x**2)
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients using analytic FT properties."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            # g(0) = (c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0) * 1 = 0
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
            
            # Compute L2 and L4 norms of g
            # Using Simpson's rule for better accuracy
            g_sq = g**2
            g_f4 = g**4
            
            norm2_sq = simpson(g_sq, x=x)
            if norm2_sq < 1e-12:
                return 1e10
                
            norm4_4 = simpson(g_f4, x=x)
            if norm4_4 < 1e-12:
                return 1e10
                
            norm2_val = np.sqrt(norm2_sq)
            norm4_val = norm4_4 ** 0.25
            
            # Analytic Fourier Transform
            # For even n, F[H_n(x) e^{-x^2/2}] = (1/sqrt(2*pi)) H_n(k) e^{-k^2/2}
            # So hat_g(k) = (1/sqrt(2*pi)) g(k)
            # This holds because the basis functions are eigenfunctions of the FT with eigenvalue 1 for even n.
            
            norm2_hat_sq = norm2_sq / (2 * np.pi)
            norm4_hat_4 = norm4_4 / (2 * np.pi)
            
            norm2_hat_val = np.sqrt(norm2_hat_sq)
            norm4_hat_val = norm4_hat_4 ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (norm2_val * norm2_hat_val)
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Signs
            sign0 = 1.0 if s0 > 0 else -1.0
            sign4 = 1.0 if s4 > 0 else -1.0
            sign8 = 1.0 if s8 > 0 else -1.0
            
            # Coefficients
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Bounds for differential evolution
    # Log-magnitudes: -4 to 4 covers ~0.018 to 54
    # Signs: -1 to 1
    bounds = [(-4, 4), (-4, 4), (-4, 4), (-1, 1), (-1, 1), (-1, 1)]
    
    # Run differential evolution
    # Using multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    
    for seed in [42, 123, 456, 789]:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=seed, maxiter=100, popsize=25,
            tol=1e-8, polish=False,
            init='sobol',
            mutation=(0.5, 1.0),
            recombination=0.8
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
        
        # Polish with local optimization on direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
            best_ratio = local_result.fun
            
        return np.array([c0, c4, c8])
    else:
        # Fallback
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
