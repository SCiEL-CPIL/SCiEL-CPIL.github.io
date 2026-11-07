import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis: dense RBFs for local features 
    plus global trigonometric and polynomial terms for periodic/trend behavior.
    
    Strategy:
    - The 10 parameters are split: 6 for RBF weights, 4 for global terms.
    - RBF centers are placed at 6 key points to capture local structure.
    - Global terms include sin(x), cos(x), x^2, and 1 to capture periodicity 
      and quadratic trends, which are common in hidden functions.
    - This hybrid approach allows the model to fit both local and global patterns,
      improving generalization over pure RBF or pure polynomial models.
    """
    x_flat = x.flatten()
    
    # RBF part: 6 centers
    centers = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    sigma = 0.35  # Slightly wider than before for better overlap
    dist_sq = (x_flat[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    gaussians = np.exp(-dist_sq / (2 * sigma ** 2))
    
    # Global terms: sin(x), cos(x), x^2, 1
    global_terms = np.column_stack([
        np.sin(x_flat),
        np.cos(x_flat),
        x_flat ** 2,
        np.ones_like(x_flat)
    ])
    
    # Combine basis: 6 RBF + 4 global = 10 columns
    basis = np.hstack([gaussians, global_terms])
    
    # Linear combination with params
    predictions = basis @ params
    
    return predictions.reshape(x.shape)
