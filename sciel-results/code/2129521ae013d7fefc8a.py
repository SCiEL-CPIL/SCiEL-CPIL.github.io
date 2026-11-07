import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a radial basis function (RBF) approach
    with Gaussian kernels centered at fixed, well-distributed points in the (x, v) phase space.
    
    The RBF network uses 10 Gaussian kernels centered at points within the training domain.
    Each kernel has a fixed width (sigma) to provide local interpolation. The parameters
    are the weights for each kernel, which BFGS can fit effectively.
    
    Centers are chosen to cover the training range:
    x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    
    We use a 3x3 grid of centers plus one at the origin for 10 total kernels.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Define 10 center points in the (x, v) phase space
    # Using a grid that covers the training range
    # x range: [-0.5728, 0.5275], v range: [-0.1754, 0.2821]
    
    # Create a 3x3 grid plus origin
    x_centers = np.array([-0.4, 0.0, 0.4])
    v_centers = np.array([-0.12, 0.0, 0.15])
    
    # Build list of centers
    centers = []
    for xc in x_centers:
        for vc in v_centers:
            centers.append((xc, vc))
    # Add origin as the 10th center
    centers.append((0.0, 0.0))
    
    # Fixed width for Gaussian kernels
    sigma = 0.3
    
    # Compute RBF contributions
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(10):
        xc, vc = centers[i]
        # Gaussian kernel: exp(-((x - xc)^2 + (v - vc)^2) / (2 * sigma^2))
        dx = x - xc
        dv = v - vc
        kernel = np.exp(-(dx * dx + dv * dv) / (2.0 * sigma * sigma))
        result += params[i] * kernel
    
    return result
