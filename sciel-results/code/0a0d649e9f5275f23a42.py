"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: multiple RBFs with fixed centers/bandwidths for local flexibility
    + global trend terms for overall shape.
    
    Structural change from parent: replace single narrow RBF with multiple RBFs
    at different centers and bandwidths (Gaussian process-like). This provides
    high local flexibility to capture non-polynomial features while the global
    terms handle broad trends. The linear-in-parameters structure ensures BFGS
    can converge globally.
    
    Basis:
    - 5 RBFs at evenly spaced centers with varying bandwidths
    - 5 global terms: constant, centered linear, centered quadratic, sin, cos
    """
    x = np.asarray(x, dtype=np.float64)
    
    # --- Global trend basis (well-scaled) ---
    x_c = (x - 3.05) / 3.05  # centered and scaled to ~[-1, 1]
    
    b0 = np.ones_like(x)
    b1 = x_c
    b2 = x_c * x_c
    b3 = np.sin(x)
    b4 = np.cos(x)
    
    # --- Local RBF basis: 5 Gaussians at different centers and bandwidths ---
    # Centers evenly spaced across [0.1, 6]
    centers = np.array([0.5, 1.8, 3.05, 4.3, 5.6])
    # Bandwidths: mix of narrow (for sharp features) and wide (for smooth bumps)
    bandwidths = np.array([0.4, 0.7, 1.0, 0.7, 0.4])
    
    # b5: RBF at center 0.5, width 0.4
    b5 = np.exp(-((x - centers[0])**2) / (2 * bandwidths[0]**2))
    
    # b6: RBF at center 1.8, width 0.7
    b6 = np.exp(-((x - centers[1])**2) / (2 * bandwidths[1]**2))
    
    # b7: RBF at center 3.05, width 1.0
    b7 = np.exp(-((x - centers[2])**2) / (2 * bandwidths[2]**2))
    
    # b8: RBF at center 4.3, width 0.7
    b8 = np.exp(-((x - centers[3])**2) / (2 * bandwidths[3]**2))
    
    # b9: RBF at center 5.6, width 0.4
    b9 = np.exp(-((x - centers[4])**2) / (2 * bandwidths[4]**2))
    
    # Stack into basis matrix
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Compute linear combination
    y = basis @ params
    
    return y

# EVOLVE-BLOCK-END
