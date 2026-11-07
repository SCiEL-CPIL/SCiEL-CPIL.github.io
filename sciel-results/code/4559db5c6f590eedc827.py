"""Symmetric regression with RBF + polynomial hybrid basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 RBF terms + 3 polynomial terms = 8 params, plus 2 more
    for flexibility (total 10 params).
    
    y = sum_{i=1}^{5} a_i * exp(-beta_i * (x - c_i)^2) + p0 + p1*x + p2*x^2
    
    Centers c_i and widths beta_i are derived from params 5..9:
    c_i = tanh(params[5+i])  (in [-1, 1] for i=0..3, and c_4 = 0 fixed)
    beta_i = softplus(params[5+i]) for i=0..4
    
    Actually let's use a cleaner parameterization:
    params[0:5] = amplitudes a_i
    params[5:8] = centers c_i for i=0,1,2 (c_3, c_4 fixed at -1.5, 1.5)
    params[8:10] = polynomial coefficients p1, p2 (p0 = 0 or absorbed)
    
    Wait, that's 5 + 3 + 2 = 10. Let me restructure:
    
    params[0:5] = amplitudes a_0..a_4
    params[5:7] = centers c_0, c_1 (c_2, c_3, c_4 fixed at 0, -1.5, 1.5)
    params[7:10] = widths log-scale w_0, w_1, w_2 (w_3, w_4 fixed)
    
    Hmm, this is getting complex. Let me simplify:
    
    Use 4 RBF terms with 2 free centers, 2 free widths, and 4 amplitudes = 10 params
    Plus a linear term.
    
    Actually, simplest clean design:
    - 5 RBF terms with centers at fixed locations: -1.5, -0.75, 0, 0.75, 1.5
    - 5 amplitudes (params[0:5])
    - 5 widths via softplus(params[5:10])
    
    That's exactly 10 params, all used. Centers are fixed to evenly spaced points.
    This is a standard RBF network with fixed centers.
    """
    x = np.asarray(x, dtype=float)
    
    # Fixed centers
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    
    # Amplitudes: params[0:5]
    amplitudes = params[0:5]
    
    # Widths: softplus(params[5:10]) to ensure positive
    raw_widths = params[5:10]
    # softplus: log(1 + exp(z)), stable version
    widths = np.log1p(np.exp(np.clip(raw_widths, -50, 50)))
    # Ensure minimum width to avoid division by zero or extremely narrow peaks
    widths = np.maximum(widths, 0.01)
    
    # Compute RBF terms
    result = np.zeros_like(x, dtype=float)
    for i in range(5):
        diff = x - centers[i]
        result += amplitudes[i] * np.exp(-widths[i] * diff * diff)
    
    return result
# EVOLVE-BLOCK-END
