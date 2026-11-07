"""Symbolic regression: nonlinear oscillator acceleration from position and velocity."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Gaussian RBF network with
    tanh-mapped centers for BFGS stability.
    
    params[0..9]: 3 RBF amplitudes, 3 x-centers (tanh-mapped), 3 v-centers (tanh-mapped), 1 width
    """
    # Training ranges for center mapping
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Tanh-mapped centers ensure BFGS starts within data range
    # Map from [-1, 1] (tanh output) to training range
    cx1 = x_min + 0.5 * (x_max - x_min) * (1 + np.tanh(params[3]))
    cx2 = x_min + 0.5 * (x_max - x_min) * (1 + np.tanh(params[4]))
    cx3 = x_min + 0.5 * (x_max - x_min) * (1 + np.tanh(params[5]))
    
    cv1 = v_min + 0.5 * (v_max - v_min) * (1 + np.tanh(params[6]))
    cv2 = v_min + 0.5 * (v_max - v_min) * (1 + np.tanh(params[7]))
    cv3 = v_min + 0.5 * (v_max - v_min) * (1 + np.tanh(params[8]))
    
    # Width parameter: use softplus to ensure positive width
    # Scale width to be reasonable for the data range
    base_width = 0.5 * max(x_max - x_min, v_max - v_min)
    width = base_width * np.exp(params[9])
    
    # Compute Gaussian RBFs
    # RBF_i(x, v) = exp(-((x - cx_i)^2 + (v - cv_i)^2) / (2 * width^2))
    
    dx1 = x - cx1
    dv1 = v - cv1
    rbf1 = np.exp(-(dx1**2 + dv1**2) / (2 * width**2))
    
    dx2 = x - cx2
    dv2 = v - cv2
    rbf2 = np.exp(-(dx2**2 + dv2**2) / (2 * width**2))
    
    dx3 = x - cx3
    dv3 = v - cv3
    rbf3 = np.exp(-(dx3**2 + dv3**2) / (2 * width**2))
    
    # Linear combination of RBFs with amplitudes
    result = params[0] * rbf1 + params[1] * rbf2 + params[2] * rbf3
    
    return result
# EVOLVE-BLOCK-END
