"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration - use finer grid for accuracy
    # Increase resolution and range to minimize boundary errors for Gaussian
    N = 16384
    x_min, x_max = -20, 20
    x = np.linspace(x_min, x_max, N)
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
    
    # Pre-compute Gaussian envelope
    gauss = np.exp(-0.5 * x**2)
    
    # FFT parameters
    # Pad to next power of 2 larger than 2*N for fine frequency resolution
    nfft = 32768
    # Frequency scaling factor for the integral approximation
    # \int |G|^4 df approximated by sum |G_k|^4 * df
    # G_k = sum g_n e^{-2pi i k n / N} * dx
    # df = 1 / (N * dx) * 2pi ? No, standard FFT freq is 1/(N*dx).
    # For angular frequency omega, domega = 2pi / (N*dx).
    # Let's stick to the previous scaling logic but verify.
    # The previous code used (2 * np.pi / nfft). Let's re-derive.
    # If G = fft(g) * dx, then G_k approximates \int g(x) e^{-i 2pi k x / (N dx)} dx.
    # The frequency step in cycles/unit is df_c = 1 / (N dx).
    # The frequency step in rad/unit is domega = 2 * pi * df_c = 2 * pi / (N dx).
    # The integral \int |G|^4 domega approx sum |G_k|^4 * domega.
    # So scaling factor is 2 * pi / (N dx).
    
    def compute_ratio(c0, c4, c8):
        """Compute the uncertainty ratio for given coefficients."""
        try:
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
            
            # Compute norms using trapezoidal rule
            g2 = g**2
            norm2_sq = np.trapz(g2, x)
            if norm2_sq < 1e-12:
                return 1e10
                
            norm4 = np.trapz(g2**2, x) # |g|^4
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform using FFT
            G = np.fft.fft(g, n=nfft) * dx
            
            # Calculate norms in frequency domain
            # Scaling factor for domega
            domega = 2 * np.pi / (nfft * dx)
            
            G2 = np.abs(G)**2
            norm2_hat_sq = np.sum(G2) * domega
            norm4_hat = np.sum(G2**2) * domega
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10
    
    def objective_from_params(params):
        """Convert log-magnitude and sign parameters to coefficients and compute objective."""
        try:
            m0, m4, m8, s0, s4, s8 = params
            
            # Smooth sign transition to help DE? Or keep hard sign?
            # Hard sign is fine for DE as it jumps.
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
    # m_i: log-magnitude, range from -5 to 4
    # s_i: sign parameter, range from -1 to 1
    bounds = [(-5, 4), (-5, 4), (-5, 4), (-1, 1), (-1, 1), (-1, 1)]
    
    # Run differential evolution
    # Use a single robust run
    result = differential_evolution(
        objective_from_params, bounds, 
        seed=42, maxiter=200, popsize=30,
        tol=1e-10, polish=False,
        init='sobol',
        mutation=(0.5, 1.0),
        recombination=0.7
    )
    
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
        # Powell is robust for this kind of smooth but non-convex function
        def local_obj(coeffs):
            c0_l, c4_l, c8_l = coeffs
            return compute_ratio(c0_l, c4_l, c8_l)
        
        local_result = minimize(
            local_obj, 
            [c0, c4, c8],
            method='Powell',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if local_result.fun < best_ratio:
            c0, c4, c8 = local_result.x
            best_ratio = local_result.fun
            
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([-1.13, 0.04, 1.0 / 3240.0])

# EVOLVE-BLOCK-END
