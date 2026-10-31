"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

def construct():
    # Precompute Hermite polynomials and their values at 0
    # We need H0, H4, H8, H12
    # H_n(0) is 0 for odd n.
    # H_0(0) = 1
    # H_4(0) = -12
    # H_8(0) = 1344
    # H_12(0) = -1032192
    
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1344.0
    H12_0 = -1032192.0

    def compute_ratio(c0, c4, c8):
        """
        Compute the uncertainty ratio (A(f)A(F)) for the given coefficients.
        g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) exp(-x^2/2)
        c12 is determined by g(0)=0.
        """
        try:
            # 1. Determine c12
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Coefficients array
            coeffs = [c0, c4, c8, c12]
            degrees = [0, 4, 8, 12]
            
            # Define the polynomial Q(x) = sum c_k H_k(x)
            # We need to integrate |Q(x)|^2 exp(-x^2) and |Q(x)|^4 exp(-2x^2)
            # And for the Fourier side:
            # Q_hat(xi) = sum c_k (-i)^k H_k(xi)
            # Integrate |Q_hat(xi)|^2 exp(-xi^2) and |Q_hat(xi)|^4 exp(-2xi^2)
            
            def integrand_g2(x):
                val = 0.0
                for c, d in zip(coeffs, degrees):
                    val += c * eval_hermite(d, x)
                return val**2 * np.exp(-x**2)
            
            def integrand_g4(x):
                val = 0.0
                for c, d in zip(coeffs, degrees):
                    val += c * eval_hermite(d, x)
                return val**4 * np.exp(-2*x**2)
            
            def integrand_hat2(xi):
                val = 0.0 + 0.0j
                for c, d in zip(coeffs, degrees):
                    # (-i)^d
                    phase = (-1j)**d
                    val += c * phase * eval_hermite(d, xi)
                return (val * np.conjugate(val)) * np.exp(-xi**2)
            
            def integrand_hat4(xi):
                val = 0.0 + 0.0j
                for c, d in zip(coeffs, degrees):
                    phase = (-1j)**d
                    val += c * phase * eval_hermite(d, xi)
                abs_sq = val * np.conjugate(val)
                return (abs_sq**2) * np.exp(-2*xi**2)

            # Integrate using quad for accuracy
            # Limits: Hermite-Gaussians decay fast. -10 to 10 is safe.
            norm2_sq, _ = quad(integrand_g2, -10, 10, limit=100)
            norm4_sq, _ = quad(integrand_g4, -10, 10, limit=100)
            norm2_hat_sq, _ = quad(integrand_hat2, -10, 10, limit=100)
            norm4_hat_sq, _ = quad(integrand_hat4, -10, 10, limit=100)
            
            # Check for convergence or zero norms
            if norm2_sq < 1e-12 or norm2_hat_sq < 1e-12:
                return 1e10
                
            # A(f) = ||f||_4 / ||f||_2
            # A(F) = ||F||_4 / ||F||_2
            # Product = (||f||_4 ||F||_4) / (||f||_2 ||F||_2)
            
            A_f = (norm4_sq**0.25) / (norm2_sq**0.5)
            A_F = (norm4_hat_sq**0.25) / (norm2_hat_sq**0.5)
            
            return A_f * A_F

        except Exception:
            return 1e10

    def objective_from_params(params):
        """
        Convert log-magnitude and sign parameters to coefficients.
        params: [m0, m4, m8, s0, s4, s8]
        c_i = sign(s_i) * exp(m_i)
        """
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Use smooth sign function to allow gradient-based methods later if needed,
            # but for DE, a hard sign is fine. However, using tanh helps with continuity
            # if we were to use a gradient method on the parameters directly.
            # Here we stick to the DE requirement but ensure robustness.
            # If s is very close to 0, sign is ambiguous. 
            # Let's use: sign = 1 if s > 0 else -1.
            # To make it slightly smoother for potential future polish on params, 
            # we could use tanh, but for DE it doesn't matter.
            
            sign0 = 1.0 if s0 > 0 else -1.0
            sign4 = 1.0 if s4 > 0 else -1.0
            sign8 = 1.0 if s8 > 0 else -1.0
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10

    # Bounds for DE
    # m_i in [-5, 5] covers coefficients from ~0.006 to ~148
    # s_i in [-1, 1]
    bounds = [(-5, 5), (-5, 5), (-5, 5), (-1, 1), (-1, 1), (-1, 1)]
    
    best_ratio = 1e10
    best_params = None
    
    # Run DE with multiple seeds
    for seed in [42, 123, 456, 789]:
        result = differential_evolution(
            objective_from_params, bounds, 
            seed=seed, maxiter=200, popsize=25,
            tol=1e-9, polish=False,
            init='sobol',
            mutation=(0.5, 1.0),
            recombination=0.9
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
        
        # Polish with local optimization on raw coefficients
        # This allows fine-tuning without the log/sign discretization issues
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        # Use Nelder-Mead or Powell
        # Powell is robust for non-smooth
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Powell',
            options={'maxiter': 1000, 'ftol': 1e-12, 'xatol': 1e-12}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
            best_ratio = local_result.fun
            
        return np.array([c0, c4, c8])
    else:
        # Fallback
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
