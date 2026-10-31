"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use finer grid for accuracy
    x = np.linspace(-20, 20, 16000)
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
    
    # Precompute Gaussian weight
    gauss = np.exp(-0.5 * x**2)
    
    # Precompute powers of Hermite polynomials times Gaussian
    # For norm integrals we need:
    # ||g||_2^2 = integral of (sum c_i H_i)^2 * exp(-x^2) dx
    # ||g||_4 = (integral of |sum c_i H_i|^4 * exp(-2x^2) dx)^(1/4)
    # 
    # For Fourier: use Parseval and the fact that FT of H_n(x)exp(-x^2/2) = (-i)^n H_n(x) exp(-x^2/2)
    # So ||g_hat||_2^2 = 2*pi * ||g||_2^2 (by Parseval with proper scaling)
    # Actually let's be careful with conventions.
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct the polynomial part
            poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
            
            # g(x) = poly(x) * exp(-x^2/2)
            g = poly * gauss
            
            # Compute ||g||_2^2 = integral g^2 dx
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-15:
                return 1e10
            
            # Compute ||g||_4^4 = integral |g|^4 dx
            norm4_pow4 = np.trapz(np.abs(g)**4, x)
            if norm4_pow4 < 1e-15:
                return 1e10
            norm4_val = norm4_pow4 ** 0.25
            
            # Fourier transform: use FFT with proper scaling
            # g(x) = poly(x) * exp(-x^2/2)
            # FT convention: G(k) = integral g(x) exp(-ikx) dx
            # Using FFT: G(k_j) ≈ dx * sum g(x_i) exp(-i k_j x_i)
            # where k_j = 2*pi*j/(N*dx) for j in range
            n = len(g)
            nfft = 1
            while nfft < 4 * n:
                nfft *= 2
            
            G = np.fft.fft(g, n=nfft) * dx
            # Frequency spacing
            dk = 2 * np.pi / (nfft * dx)
            
            # ||g_hat||_2^2 = integral |G(k)|^2 dk ≈ sum |G(k_j)|^2 * dk
            norm2_hat_sq = np.sum(np.abs(G)**2) * dk
            
            # ||g_hat||_4^4 = integral |G(k)|^4 dk ≈ sum |G(k_j)|^4 * dk
            norm4_hat_pow4 = np.sum(np.abs(G)**4) * dk
            if norm4_hat_pow4 < 1e-15:
                return 1e10
            norm4_hat_val = norm4_hat_pow4 ** 0.25
            
            # The ratio A(f)*A(Fourier(f)) where A(f) = ||f||_4 / ||f||_2
            # So A(f) = norm4_val / sqrt(norm2_sq)
            # A(F(f)) = norm4_hat_val / sqrt(norm2_hat_sq)
            
            ratio = (norm4_val / np.sqrt(norm2_sq)) * (norm4_hat_val / np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Sign via smooth function to avoid discontinuities
            # Use tanh for smooth sign transition
            sign0 = np.tanh(s0)
            sign4 = np.tanh(s4)
            sign8 = np.tanh(s8)
            
            # Avoid zero coefficients
            min_mag = 1e-6
            c0 = sign0 * max(np.exp(m0), min_mag)
            c4 = sign4 * max(np.exp(m4), min_mag)
            c8 = sign8 * max(np.exp(m8), min_mag)
            
            return compute_ratio(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -3 to 2 (coefficients from ~0.05 to ~7.4)
    # s_i: sign parameter for tanh, range from -2 to 2
    bounds = [(-3, 2), (-3, 2), (-3, 2), (-2, 2), (-2, 2), (-2, 2)]
    
    # Run differential evolution with multiple seeds for robustness
    best_ratio = 1e10
    best_params = None
    
    for seed in [42, 123, 456, 789, 1024]:
        try:
            result = differential_evolution(
                objective_from_params, bounds, 
                seed=seed, maxiter=200, popsize=25,
                tol=1e-10, polish=False,
                init='sobol',
                mutation=(0.5, 1.0),
                recombination=0.8,
                update_interval=10,
                workers=-1
            )
            
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_params = result.x
        except Exception:
            continue
    
    if best_params is not None:
        m0, m4, m8, s0, s4, s8 = best_params
        sign0 = np.tanh(s0)
        sign4 = np.tanh(s4)
        sign8 = np.tanh(s8)
        min_mag = 1e-6
        c0 = sign0 * max(np.exp(m0), min_mag)
        c4 = sign4 * max(np.exp(m4), min_mag)
        c8 = sign8 * max(np.exp(m8), min_mag)
        
        # Polish with local optimization using direct coefficients
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
            best_ratio = local_result.fun
        
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
