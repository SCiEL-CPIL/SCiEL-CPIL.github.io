# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model for nonlinear oscillator dynamics.
    
    Uses 10 fitted parameters arranged as 3 Gaussian RBFs + 1 global linear bias:
    - p[0]: global constant offset
    - For k in 0..2:
      - p[1+3k]: center x-coordinate (cx_k)
      - p[2+3k]: center v-coordinate (cv_k)  
      - p[3+3k]: combined amplitude/width parameter (a_k)
    
    Model:
    a = p0 + sum_k a_k * exp(-((x - cx_k)^2 + (v - cv_k)^2) / (2 * sigma^2))
    
    where sigma is a fixed width chosen to cover the input domain.
    This captures localized nonlinear dynamics that global polynomials miss.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure 1D
    x_flat = x.ravel()
    v_flat = v.ravel()
    n = x_flat.shape[0]
    
    # Global offset
    result = np.full(n, p[0], dtype=np.float64)
    
    # Fixed width to cover the input domain
    # x range ~ [-0.57, 0.53], v range ~ [-0.18, 0.28]
    # Use a moderate sigma so Gaussians have meaningful support
    sigma2 = 0.25  # sigma = 0.5, so 2*sigma^2 = 0.5
    
    # Three Gaussian RBFs
    for k in range(3):
        cx = p[1 + 3*k]
        cv = p[2 + 3*k]
        amp = p[3 + 3*k]
        
        dx = x_flat - cx
        dv = v_flat - cv
        dist2 = dx * dx + dv * dv
        gaussian = np.exp(-dist2 / (2.0 * sigma2))
        result += amp * gaussian
    
    return result
# EVOLVE-BLOCK-END
