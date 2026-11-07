"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10 localized Gaussian basis functions evenly spaced across [-3, 3]
    # Fixed centers and widths for stability; params are the 10 amplitudes
    # plus potentially unused params for shape compatibility.
    
    # Evenly spaced centers in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed widths: choose width so that Gaussians have reasonable overlap
    # spacing is 6/9 = 2/3, so width ~ 0.5 gives good local flexibility
    widths = np.full(10, 0.5)
    
    # Compute Gaussian basis: shape (n_points, 10)
    # x has shape (n_points,)
    x_col = x[:, np.newaxis]  # (n, 1)
    c_row = centers[np.newaxis, :]  # (1, 10)
    w_row = widths[np.newaxis, :]  # (1, 10)
    
    diff = (x_col - c_row) / w_row  # (n, 10)
    basis = np.exp(-0.5 * diff ** 2)  # (n, 10)
    
    # Use first 10 params as amplitudes for each Gaussian
    # If params has more than 10, ignore extras; if fewer, pad with zeros
    if len(params) >= 10:
        amps = params[:10]
    else:
        amps = np.zeros(10)
        amps[:len(params)] = params
    
    # Prediction: (n, 10) @ (10,) -> (n,)
    pred = basis @ amps
    
    return pred
# EVOLVE-BLOCK-END
