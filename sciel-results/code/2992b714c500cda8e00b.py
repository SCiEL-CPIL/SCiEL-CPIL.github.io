import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a soft-thresholded hybrid basis.
    
    The basis combines:
    - Chebyshev-like polynomial terms (T0..T3)
    - Fourier terms (sin/cos at frequencies 1, 2, 3 * pi)
    - Each term is modulated by a smooth soft-thresholding factor:
        w_k(t) = 1 + alpha_k * tanh(beta_k * (|t| - c_k))
    This introduces piecewise-linear-like flexibility (kinks/saturation)
    while keeping the model linear-in-params for BFGS convergence.
    
    10 parameters:
    p[0]: DC offset
    p[1]: coefficient for sin(pi*t) * w_1
    p[2]: coefficient for cos(pi*t) * w_2
    p[3]: coefficient for sin(2*pi*t) * w_3
    p[4]: coefficient for cos(2*pi*t) * w_4
    p[5]: coefficient for sin(3*pi*t) * w_5
    p[6]: coefficient for cos(3*pi*t) * w_6
    p[7]: coefficient for t * w_7
    p[8]: coefficient for t^2 * w_8
    p[9]: coefficient for t^3 * w_9
    
    The soft-thresholding parameters (alpha_k, beta_k, c_k) are fixed
    hyperparameters chosen to provide mild non-smoothness without
    destabilizing BFGS.
    """
    t = x / 3.0
    abs_t = np.abs(t)
    
    # Fixed hyperparameters for soft-thresholding
    # alpha: max modulation amplitude (small to keep near-linear)
    # beta: sharpness of the tanh transition
    # c: center of the transition (where |t| = c)
    alphas = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    betas = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    centers = np.array([0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.5, 0.5, 0.5])
    
    # Pre-compute basis functions
    sin_1 = np.sin(np.pi * t)
    cos_1 = np.cos(np.pi * t)
    sin_2 = np.sin(2.0 * np.pi * t)
    cos_2 = np.cos(2.0 * np.pi * t)
    sin_3 = np.sin(3.0 * np.pi * t)
    cos_3 = np.cos(3.0 * np.pi * t)
    t_1 = t
    t_2 = t * t
    t_3 = t_2 * t
    
    # Basis terms
    bases = np.stack([
        np.ones_like(t),  # p[0]: DC (no soft-threshold)
        sin_1,            # p[1]
        cos_1,            # p[2]
        sin_2,            # p[3]
        cos_2,            # p[4]
        sin_3,            # p[5]
        cos_3,            # p[6]
        t_1,              # p[7]
        t_2,              # p[8]
        t_3,              # p[9]
    ], axis=-1)  # shape: (N, 10)
    
    # Soft-thresholding weights for each basis term (except DC)
    # w_k(t) = 1 + alpha_k * tanh(beta_k * (|t| - c_k))
    weights = 1.0 + alphas * np.tanh(betas * (abs_t - centers))
    weights[0] = 1.0  # DC term is not modulated
    
    # Apply weights to bases
    bases = bases * weights[np.newaxis, :]
    
    # Linear combination
    y = bases @ params
    
    return y

