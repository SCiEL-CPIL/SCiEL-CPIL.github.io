"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process-like RBF basis with fixed centers and bandwidths.
    Provides high local flexibility to capture non-polynomial features
    while maintaining linear-in-parameters structure for global convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers spread across [0.1, 6]
    centers = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
    
    # Multiple bandwidths for multi-scale representation
    # Narrow bandwidths for local features, wider for global trends
    bandwidths = np.array([0.3, 0.5, 0.8, 1.2, 2.0])
    
    # Build RBF basis: for each center and bandwidth, compute exp(-((x - center)^2) / (2 * bandwidth^2))
    # We have 11 centers x 5 bandwidths = 55 potential basis functions, but we only have 10 params
    # Strategy: use 10 selected (center, bandwidth) pairs that cover the space well
    
    # Select 10 (center, bandwidth) combinations:
    # 5 narrow RBFs at key locations for local flexibility
    # 5 wider RBFs for global trend capture
    
    # Narrow RBFs (bandwidth ~0.3-0.5) at strategic centers
    rbf1 = np.exp(-((x - 0.5)**2) / (2 * 0.4**2))   # narrow at left
    rbf2 = np.exp(-((x - 1.5)**2) / (2 * 0.4**2))   # narrow
    rbf3 = np.exp(-((x - 2.5)**2) / (2 * 0.4**2))   # narrow at center-left
    rbf4 = np.exp(-((x - 3.5)**2) / (2 * 0.4**2))   # narrow at center-right
    rbf5 = np.exp(-((x - 5.0)**2) / (2 * 0.4**2))   # narrow at right
    
    # Wider RBFs (bandwidth ~1.0-2.0) for global trends
    rbf6 = np.exp(-((x - 1.0)**2) / (2 * 1.2**2))   # wide left
    rbf7 = np.exp(-((x - 3.0)**2) / (2 * 1.5**2))   # wide center
    rbf8 = np.exp(-((x - 5.0)**2) / (2 * 1.2**2))   # wide right
    rbf9 = np.exp(-((x - 2.0)**2) / (2 * 2.0**2))   # very wide left-center
    rbf10 = np.exp(-((x - 4.0)**2) / (2 * 2.0**2))  # very wide right-center
    
    # Stack into basis matrix
    basis = np.column_stack([rbf1, rbf2, rbf3, rbf4, rbf5,
                             rbf6, rbf7, rbf8, rbf9, rbf10])
    
    # Compute linear combination
    y = basis @ params
    
    return y

# EVOLVE-BLOCK-END
