"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration
    # Use a finer grid for better accuracy of L4 and FT
    x = np.linspace(-12, 12, 4000)
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
    
    def objective(params):
        # params: [m0, m4, m8, s0, s4, s8] where m_i are log-magnitudes, s_i are signs in {-1, 1}
        # We'll use continuous parameters and round signs, or better, use tanh for signs
        # Actually, let's use a different parameterization: 
        # c_i = exp(m_i) * tanh(s_i) but tanh(0)=0 which is bad.
        # Better: c_i = sign_i * exp(m_i) where sign_i = 1 if s_i > 0 else -1
        # But for continuous optimization, let's use:
        # c_i = exp(m_i) * (2 * (s_i > 0) - 1) -- this is discontinuous.
        
        # Alternative: use c_i = exp(m_i) * tanh(s_i) is problematic near 0.
        # Let's stick to the original approach but use differential_evolution which can handle
        # multi-modal landscapes better.
        
        try:
            m0, m4, m8, s0, s4, s8 = params
            # Convert sign parameters to actual signs (-1 or 1)
            # We'll use: sign = 1 if param > 0 else -1
            # This is discontinuous but DE can handle it
            sign0 = 1.0 if s0 > 0 else -1.0
            sign4 = 1.0 if s4 > 0 else -1.0
            sign8 = 1.0 if s8 > 0 else -1.0
            
            c0 = sign0 * np.exp(m0)
            c4 = sign4 * np.exp(m4)
            c8 = sign8 * np.exp(m8)
            
            # Enforce g(0) = 0 => c12 is determined
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            # Construct g(x)
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
            
            # Compute norms
            norm2_sq = np.sum(g**2) * dx
            if norm2_sq < 1e-10:
                return 1e10
                
            norm4 = np.sum(np.abs(g)**4) * dx
            norm4_val = norm4 ** 0.25
            
            # Fourier Transform using FFT
            G = np.fft.fft(g) * dx
            dxi = 1 / (len(g) * dx)
            
            norm2_hat_sq = np.sum(np.abs(G)**2) * dxi
            norm4_hat = np.sum(np.abs(G)**4) * dxi
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            
            return ratio
        except Exception:
            return 1e10

    # Define bounds for differential evolution
    # m_i: log-magnitude, range from -5 to 2 (coefficients from ~0.007 to ~7.4)
    # s_i: sign parameter, range from -1 to 1 (we'll use sign of value)
    bounds = [(-5, 2), (-5, 2), (-5, 2), (-1, 1), (-1, 1), (-1, 1)]
    
    # Initial guess from seed
    c0_seed = -1.13
    c4_seed = 0.04
    c8_seed = 1.0 / 3240.0
    
    m0_seed = np.log(np.abs(c0_seed))
    m4_seed = np.log(np.abs(c4_seed))
    m8_seed = np.log(np.abs(c8_seed))
    
    s0_seed = 1.0 if c0_seed > 0 else -1.0
    s4_seed = 1.0 if c4_seed > 0 else -1.0
    s8_seed = 1.0 if c8_seed > 0 else -1.0
    
    # Run differential evolution
    # Use moderate population size and max iterations to stay within time limit
    result = differential_evolution(
        objective, bounds, 
        seed=42, maxiter=200, popsize=15,
        tol=1e-6, polish=False,
        init='sobol'
    )
    
    if result.success:
        m0, m4, m8, s0, s4, s8 = result.x
        sign0 = 1.0 if s0 > 0 else -1.0
        sign4 = 1.0 if s4 > 0 else -1.0
        sign8 = 1.0 if s8 > 0 else -1.0
        c0 = sign0 * np.exp(m0)
        c4 = sign4 * np.exp(m4)
        c8 = sign8 * np.exp(m8)
        return np.array([c0, c4, c8])
    else:
        # Fallback to seed
        return np.array([c0_seed, c4_seed, c8_seed])

# EVOLVE-BLOCK-END
