import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a single hidden-layer RBF (Radial Basis Function) network.
    
    The 10 parameters act as output weights for 10 Gaussian RBFs centered at evenly spaced
    points in [0.1, 6]. This provides a flexible, smooth non-linear basis that can fit
    arbitrary smooth functions without the conditioning issues of high-degree polynomials.
    
    The RBF width is fixed to provide good coverage of the domain. The model is linear
    in the parameters, so BFGS will reliably find the global minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 10 evenly spaced centers in [0.1, 6]
    n_centers = 10
    centers = np.linspace(0.1, 6.0, n_centers)
    
    # Fixed width for the Gaussian RBFs
    # Width should be roughly the spacing between centers to avoid overlap issues
    # Spacing is (6.0 - 0.1) / (10 - 1) ≈ 0.6556
    # A width of ~0.5-0.7 works well; let's use 0.5 for slightly sharper features
    width = 0.5
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * width^2))
    # Shape: (len(x), n_centers)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    # Prediction is the linear combination: sum_j params[j] * phi_j(x)
    return rbf_values @ params
