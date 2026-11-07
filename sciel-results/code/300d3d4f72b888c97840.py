import numpy as np
from scipy.optimize import minimize

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params: 10 values = 5 centers + 5 widths
    # Use fixed positive weights to keep the model simple
    # RBF: sum_i w_i * exp(-((x - c_i)^2) / (2 * sigma_i^2))
    
    centers = params[:5]
    sigmas = np.abs(params[5:]) + 1e-3  # ensure positive widths
    
    # Fixed weights (all ones, could be optimized but we only have 10 params)
    weights = np.ones(5)
    
    # Compute RBF output
    # x shape: (N,), centers shape: (5,)
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    result = np.exp(-diff**2 / (2.0 * sigmas[np.newaxis, :]**2))
    result = result @ weights
    
    return result
