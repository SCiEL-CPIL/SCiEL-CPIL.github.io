# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale RBF basis function model.
    
    Strategy:
    - The domain is [0.1, 6.0].
    - We use 10 parameters.
    - Param 0: Intercept (constant term).
    - Params 1-9: Weights for 9 RBFs.
    
    RBF Centers (c_i) and Bandwidths (sigma_i) are fixed.
    We choose centers spread across the domain and a mix of sigmas:
    - Wide sigmas (e.g., 3.0) to capture global trends.
    - Medium sigmas (e.g., 1.0) to capture mid-scale features.
    - Narrow sigmas (e.g., 0.2) to capture local variations/sharp changes.
    
    This linear model in parameters allows BFGS to find the global optimum efficiently.
    The multi-scale nature addresses the limitation of the previous Fourier approach 
    which was limited to low frequencies.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Fixed centers and bandwidths
    # Domain: [0.1, 6.0]
    # Centers: Evenly spaced roughly, plus some for flexibility
    # Let's pick 9 centers:
    # 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5
    # We only have 9 slots for RBFs (params 1-9).
    # Let's space them out:
    centers = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5])
    
    # Bandwidths: Mix of scales
    # We want to cover global to local.
    # sigma = 2.5 (global), 1.5 (mid), 1.0 (mid), 0.8, 0.6, 0.5, 0.4, 0.3, 0.2
    # Let's assign:
    sigmas = np.array([2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    
    # Compute RBFs: exp(-((x - c)^2) / (2 * sigma^2))
    # x shape: (N,)
    # centers shape: (9,)
    # sigmas shape: (9,)
    
    # Efficient computation
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Pre-compute inverse variance for speed
    inv_var = 1.0 / (2.0 * sigmas**2)
    
    # rbf shape: (N, 9)
    rbf = np.exp(-(diff**2) * inv_var[np.newaxis, :])
    
    # Extract parameters
    # params[0] is intercept
    # params[1:10] are weights
    intercept = params[0]
    weights = params[1:10]
    
    # Prediction: intercept + sum(weights * rbf)
    y_pred = intercept + np.sum(rbf * weights[np.newaxis, :], axis=1)
    
    return y_pred
# EVOLVE-BLOCK-END
