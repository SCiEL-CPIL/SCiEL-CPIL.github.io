# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Basis: Low-degree Polynomial + 2 Gaussian RBFs.
    
    Strategy:
    The target function is decomposed into a global trend (polynomial) and local features (RBFs).
    This structure is more robust than pure RBFs (which struggle to fit global trends) 
    and more flexible than pure polynomials (which struggle with local bumps).
    
    Parameters (10 total):
    p[0], p[1], p[2], p[3] : Polynomial coefficients for 1, x, x^2, x^3
    p[4], p[5], p[6]       : Center (mu), Log-width (log_sig), Amplitude (amp) for RBF 1
    p[7], p[8], p[9]       : Center (mu), Log-width (log_sig), Amplitude (amp) for RBF 2
    
    Note: Using log-width ensures sigma > 0.
    """
    if time.time() > _DEADLINE:
        # Fallback: simple linear
        return params[0] * x + params[1]
    
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    if p.shape[0] < 10:
        p = np.concatenate([p, np.zeros(10 - p.shape[0])])
        
    # 1. Global Polynomial Component (Degree 3)
    # Using raw monomials is fine for degree 3 on [-3, 3] as they don't blow up.
    # c0 + c1*x + c2*x^2 + c3*x^3
    poly_term = p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3
    
    # 2. Local RBF Component (2 Gaussians)
    # RBF 1
    mu1 = p[4]
    log_sig1 = np.clip(p[5], -10, 10)
    sig1 = np.exp(log_sig1)
    amp1 = p[6]
    
    # RBF 2
    mu2 = p[7]
    log_sig2 = np.clip(p[8], -10, 10)
    sig2 = np.exp(log_sig2)
    amp2 = p[9]
    
    # Calculate RBFs safely
    # z = (x - mu) / sig
    # exp(-0.5 * z^2)
    
    z1 = (x - mu1) / sig1
    rbf1 = amp1 * np.exp(-0.5 * z1 * z1)
    
    z2 = (x - mu2) / sig2
    rbf2 = amp2 * np.exp(-0.5 * z2 * z2)
    
    result = poly_term + rbf1 + rbf2
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback: just the polynomial part
        return poly_term
        
    return result
# EVOLVE-BLOCK-END
