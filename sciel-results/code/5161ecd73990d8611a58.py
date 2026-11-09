import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    # Check for non-finite parameters
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Use 3 local Gaussian basis functions
    # Each uses 3 parameters: center, width, amplitude
    # Total 9 parameters. The 10th parameter is a global constant offset.
    
    c1, w1, a1 = p[0], p[1], p[2]
    c2, w2, a2 = p[3], p[4], p[5]
    c3, w3, a3 = p[6], p[7], p[8]
    offset = p[9]
    
    # Ensure widths are positive and not too small to avoid numerical issues
    # Using softplus or abs + epsilon is safer than raw values for BFGS
    # Since BFGS starts at 1, w=1 is fine, but gradients might push it negative.
    # Let's use abs(w) + 1e-6 to ensure positive width.
    w1_safe = np.abs(w1) + 1e-6
    w2_safe = np.abs(w2) + 1e-6
    w3_safe = np.abs(w3) + 1e-6
    
    # Compute Gaussians
    # exp(-((x - c)^2) / (2 * w^2))
    g1 = np.exp(-((x - c1)**2) / (2 * w1_safe**2))
    g2 = np.exp(-((x - c2)**2) / (2 * w2_safe**2))
    g3 = np.exp(-((x - c3)**2) / (2 * w3_safe**2))
    
    y = a1 * g1 + a2 * g2 + a3 * g3 + offset
    
    return y
