import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fixed Gaussian RBF basis.
    
    Strategy:
    1. Scale inputs to [0, 1] to improve BFGS conditioning (inputs are in [0.5, 2]).
    2. Use 9 fixed Gaussian Radial Basis Functions (RBFs) centered at a 3x3x3 grid 
       in the scaled space, plus 1 constant term.
    3. This provides a universal approximator structure for smooth functions, 
       unlike the current quadratic polynomial which is limited to degree 2.
    4. The fixed centers and bandwidths prevent the optimizer from struggling 
       with non-convex center/bandwidth parameters, focusing BFGS on the linear 
       combination weights which is well-behaved for a fixed basis.
    """
    # Extract parameters
    # p[0] is the constant bias
    # p[1:10] are the weights for the 9 RBFs
    
    p = params
    
    # 1. Scale inputs to [0, 1]
    # Original range: [0.5, 2]
    # Scaled range: (x - 0.5) / 1.5 -> [0, 1]
    x_s = (x - 0.5) / 1.5
    y_s = (y - 0.5) / 1.5
    z_s = (z - 0.5) / 1.5
    
    # 2. Define fixed centers in the scaled [0, 1]^3 space
    # Centers at 0.25, 0.5, 0.75 for each dimension
    centers = np.array([
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.75],
        [0.25, 0.75, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.25],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
        [0.75, 0.75, 0.75],
        [0.50, 0.50, 0.50]
    ])
    
    # 3. Compute RBF values
    # Bandwidth sigma = 0.4 (chosen to provide overlap but distinct centers)
    sigma = 0.4
    
    # Vectorized computation for all 9 RBFs
    # x_s, y_s, z_s are arrays of shape (N,)
    # centers are (9, 3)
    
    # Calculate squared distances
    dx = x_s[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y_s[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    dz = z_s[:, np.newaxis] - centers[:, 2][np.newaxis, :]
    
    sq_dist = dx**2 + dy**2 + dz**2
    
    # Gaussian kernel: exp(-sq_dist / (2 * sigma^2))
    # To avoid underflow/overflow issues, we rely on the fact that sq_dist is bounded
    # Max dist in [0,1]^3 is sqrt(3) ~ 1.732, sq_dist ~ 3.
    # 2*sigma^2 = 2*0.16 = 0.32.
    # Max exponent: -3/0.32 ~ -9.3. exp(-9.3) is ~ 9e-5, well within float64 range.
    
    rbf_values = np.exp(-sq_dist / (2 * sigma**2))
    
    # rbf_values shape: (N, 9)
    
    # 4. Linear combination
    # p[0] is bias, p[1:10] are RBF weights
    bias = p[0]
    weights = p[1:10]
    
    # Dot product: (N, 9) dot (9,) -> (N,)
    pred = bias + rbf_values @ weights
    
    return pred
