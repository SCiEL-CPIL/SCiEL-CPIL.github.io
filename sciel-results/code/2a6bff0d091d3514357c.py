# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 3 global (1, t, t^2) + 7 Gaussian RBFs.
    x in [0.1, 6] mapped to t in [-1, 1] for global terms.
    RBFs have fixed centers and widths; only amplitudes are fitted.
    params: [c0, c1, c2, a1, a2, ..., a7]
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Map to [-1, 1] for global basis
    t = (2.0 * x - 6.1) / 5.9
    
    # Global basis: 1, t, t^2
    G = np.column_stack([np.ones(n), t, t * t])
    
    # RBF centers (7 points spread across [0.1, 6])
    centers = np.array([0.5, 1.3, 2.1, 2.9, 3.7, 4.5, 5.3])
    sigma = 0.7
    
    # Compute RBF values: exp(-0.5 * ((x - c)/sigma)^2)
    # Shape: (n, 7)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    R = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine: params[0:3] for global, params[3:10] for RBF
    pred = G @ params[0:3] + R @ params[3:10]
    
    return pred
# EVOLVE-BLOCK-END
