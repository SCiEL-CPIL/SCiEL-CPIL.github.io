"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize x to [-1, 1] for better conditioning
    z = x / 3.0
    
    # --- Chebyshev polynomials T0..T4 (better conditioned than raw powers) ---
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * z - 1.0
    T3 = 4.0 * z * z * z - 3.0 * z
    T4 = 8.0 * z * z * z * z - 8.0 * z * z + 1.0
    
    # --- Trigonometric basis with learnable frequency ---
    # params[5] is used as frequency w for sine, params[6] as amplitude
    w = params[5]
    A = params[6]
    sin_term = A * np.sin(w * z)
    
    # --- Cosine with fixed frequency 2.0 (captures higher freq oscillations) ---
    cos_term = params[7] * np.cos(2.0 * z)
    
    # --- Gaussian bump with learnable width ---
    # params[8] controls width (positive via softplus to keep exp stable)
    width = 1.0 + 0.5 * np.log1p(np.exp(np.abs(params[8])))  # softplus-like, ensures width > 0
    gauss = params[9] * np.exp(-0.5 * (z / width) ** 2)
    
    # --- Additional smooth nonlinear term (smooth step-like) ---
    # Use a smooth approximation to capture saturation without tanh's sharp kink
    smooth_sat = np.arctan(z)  # smoother than tanh, bounded in (-pi/2, pi/2)
    
    # Construct prediction:
    # params[0:4] -> Chebyshev T0-T3
    # params[4]   -> Chebyshev T4
    # params[5]   -> frequency w
    # params[6]   -> amplitude A for sin(w*z)
    # params[7]   -> amplitude for cos(2*z)
    # params[8]   -> width parameter for Gaussian
    # params[9]   -> amplitude for Gaussian bump
    
    pred = (params[0] * T0 +
            params[1] * T1 +
            params[2] * T2 +
            params[3] * T3 +
            params[4] * T4 +
            sin_term +
            cos_term +
            gauss +
            0.1 * smooth_sat)  # small fixed contribution from smooth saturation
    
    return pred
# EVOLVE-BLOCK-END
