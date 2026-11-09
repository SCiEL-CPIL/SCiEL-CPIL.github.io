import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of three Gaussian Radial Basis Functions (RBFs) with optimizable centers, widths, and amplitudes.
    This structural change moves to a local basis function model which can capture
    non-smooth or localized features that global sinusoidal or polynomial bases might miss.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # RBF 1: center (p0, p1, p2), width p3, amplitude p6
    # dist_sq1 = (x-p0)^2 + (y-p1)^2 + (z-p2)^2
    dx1 = x - p0
    dy1 = y - p1
    dz1 = z - p2
    dist_sq1 = dx1**2 + dy1**2 + dz1**2
    # Use width as standard deviation in Gaussian: exp(-dist_sq / (2 * width^2))
    # To avoid division by zero if width is near 0, add a small epsilon
    w1 = np.maximum(np.abs(p3), 1e-6)
    term1 = p6 * np.exp(-dist_sq1 / (2 * w1**2))
    
    # RBF 2: center (p4, p5, p6) -- wait, p6 is used above. 
    # We have 10 params. 
    # Let's use:
    # RBF 1: center (p0, p1, p2), width p3, amp p4
    # RBF 2: center (p5, p6, p7), width p8, amp p9
    # This uses all 10 params.
    
    # Let's restart the logic with this mapping.
    
    # RBF 1
    c1x, c1y, c1z = p0, p1, p2
    w1 = np.maximum(np.abs(p3), 1e-6)
    a1 = p4
    d1 = (x - c1x)**2 + (y - c1y)**2 + (z - c1z)**2
    term1 = a1 * np.exp(-d1 / (2 * w1**2))
    
    # RBF 2
    c2x, c2y, c2z = p5, p6, p7
    w2 = np.maximum(np.abs(p8), 1e-6)
    a2 = p9
    d2 = (x - c2x)**2 + (y - c2y)**2 + (z - c2z)**2
    term2 = a2 * np.exp(-d2 / (2 * w2**2))
    
    pred = term1 + term2
    
    return pred
