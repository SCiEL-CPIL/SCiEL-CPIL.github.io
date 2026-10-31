"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

def construct():
    # Grid for numerical integration
    x = np.linspace(-15, 15, 8000)
    dx = x[1] - x[0]
    
    # Precompute Hermite polynomials
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    # Values at 0
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # FFT setup
    n = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
        
    def compute_ratio(c4, c8):
        """
        Compute the uncertainty ratio for given c4, c8.
        c0 is fixed to 1.
        c12 is derived from g(0)=0.
        """
        try:
            c0 = 1.0
            # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
            c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
            
            g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * np.exp(-0.5 * x**2)
            
            norm2_sq = np.trapz(g**2, x)
            if norm2_sq < 1e-12:
                return 1e10
                
            norm4 = np.trapz(np.abs(g)**4, x)
            norm4_val = norm4 ** 0.25
            
            G = np.fft.fft(g, n=nfft) * dx
            
            norm2_hat_sq = np.sum(np.abs(G)**2) * (2 * np.pi / nfft)
            norm4_hat = np.sum(np.abs(G)**4) * (2 * np.pi / nfft)
            norm4_hat_val = norm4_hat ** 0.25
            
            ratio = (norm4_val * norm4_hat_val) / (np.sqrt(norm2_sq) * np.sqrt(norm2_hat_sq))
            return ratio
        except Exception:
            return 1e10

    def objective(params):
        c4, c8 = params
        return compute_ratio(c4, c8)

    # Optimization
    # The problem is 2D. Use DE for global search, then NM for local polish.
    bounds = [(-10, 10), (-10, 10)]
    
    # Run DE
    result = differential_evolution(
        objective, bounds, 
        seed=42, maxiter=500, popsize=25,
        tol=1e-10, polish=False,
        init='sobol',
        mutation=(0.5, 1.0)
    )
    
    c4_best, c8_best = result.x
    
    # Polish with Nelder-Mead
    local_result = minimize(
        objective, 
        [c4_best, c8_best],
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
    )
    
    c4_final, c8_final = local_result.x
    
    return np.array([1.0, c4_final, c8_final])

# EVOLVE-BLOCK-END
