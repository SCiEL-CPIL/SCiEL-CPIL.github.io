# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates the model using a combination of localized Gaussian basis functions
    and a low-order polynomial backbone to provide both local flexibility and
    global structure.
    
    params[0..9] are coefficients for:
    1, x, x^2,  exp(-((x-c0)/w)^2), exp(-((x-c1)/w)^2), ..., exp(-((x-c6)/w)^2)
    where c_i = -3 + 6*i/6 for i=0..6, and w is a fixed width.
    
    This hybrid approach avoids the ill-conditioning of pure polynomial or
    pure RBF bases, giving BFGS a smoother landscape to navigate.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial backbone: 1, x, x^2
    x2 = x * x
    
    # Gaussian basis functions centered at evenly spaced points
    # 7 centers: -3, -2, -1, 0, 1, 2, 3
    # Width chosen to provide overlap but maintain locality
    sigma = 1.2
    sigma2 = sigma * sigma
    two_sigma2 = 2.0 * sigma2
    
    centers = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    
    # Compute Gaussian basis values: shape (n_points, 7)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-diff * diff / two_sigma2)
    
    # Combine: p0*1 + p1*x + p2*x^2 + p3*g0 + ... + p9*g6
    y = p0 * 1.0 + p1 * x + p2 * x2
    for i in range(7):
        y += params[3 + i] * gaussians[:, i]
    
    return y
# EVOLVE-BLOCK-END
