"""Symbolic regression with linear-in-parameters basis (Chebyshev + RBF)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model using Chebyshev polynomials and RBF terms.
    
    Basis functions (10 total, matching params length):
    0: 1 (constant)
    1: T_1(x) = x
    2: T_2(x) = 2x^2 - 1
    3: T_3(x) = 4x^3 - 3x
    4: T_4(x) = 8x^4 - 8x^2 + 1
    5: T_5(x) = 16x^5 - 20x^3 + 5x
    6: RBF centered at -2.0, width 1.0
    7: RBF centered at 0.0, width 1.0
    8: RBF centered at 2.0, width 1.0
    9: RBF centered at 0.0, width 0.5
    
    All basis functions are smooth and differentiable, ensuring BFGS can
    converge to the global least-squares solution for this linear model.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.ravel()
    
    # Compute Chebyshev polynomials (scaled to [-3,3] domain)
    # T_0 = 1
    T0 = np.ones_like(x_flat)
    # T_1 = x
    T1 = x_flat
    # T_2 = 2x^2 - 1
    T2 = 2.0 * x_flat**2 - 1.0
    # T_3 = 4x^3 - 3x
    T3 = 4.0 * x_flat**3 - 3.0 * x_flat
    # T_4 = 8x^4 - 8x^2 + 1
    T4 = 8.0 * x_flat**4 - 8.0 * x_flat**2 + 1.0
    # T_5 = 16x^5 - 20x^3 + 5x
    T5 = 16.0 * x_flat**5 - 20.0 * x_flat**3 + 5.0 * x_flat
    
    # RBF terms: exp(-(x - center)^2 / (2 * width^2))
    # Center at -2.0, width 1.0
    rbf1 = np.exp(-((x_flat - (-2.0))**2) / (2.0 * 1.0**2))
    # Center at 0.0, width 1.0
    rbf2 = np.exp(-((x_flat - 0.0)**2) / (2.0 * 1.0**2))
    # Center at 2.0, width 1.0
    rbf3 = np.exp(-((x_flat - 2.0)**2) / (2.0 * 1.0**2))
    # Center at 0.0, width 0.5 (narrower)
    rbf4 = np.exp(-((x_flat - 0.0)**2) / (2.0 * 0.5**2))
    
    # Stack basis functions
    # Shape: (n_points, 10)
    basis = np.column_stack([
        T0, T1, T2, T3, T4, T5, rbf1, rbf2, rbf3, rbf4
    ])
    
    # params is length 10
    p = np.asarray(params, dtype=np.float64)
    
    # Prediction: basis @ params
    pred = basis @ p
    
    # Reshape to match input shape
    return pred.reshape(x.shape)
# EVOLVE-BLOCK-END
