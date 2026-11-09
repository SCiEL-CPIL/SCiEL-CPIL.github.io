"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (global smoothness) + Gaussian RBFs (local flexibility).
    6 Chebyshev terms (T_0 through T_5) + 4 Gaussian RBFs with fixed centers.
    This provides both global polynomial approximation and local bump functions,
    allowing the model to capture both smooth trends and localized features.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Scale input to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Build basis matrix: 6 Chebyshev + 4 RBF = 10 columns
    basis = np.empty((n, 10), dtype=np.float64)
    
    # --- Chebyshev polynomial terms (T_0 to T_5) ---
    basis[:, 0] = 1.0           # T_0
    basis[:, 1] = t             # T_1
    
    for k in range(2, 6):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # --- Gaussian RBF terms with fixed centers ---
    # Centers spread across the domain, width chosen for local flexibility
    rbf_centers = np.array([-1.5, -0.5, 0.5, 1.5])
    rbf_width = 0.8
    
    for j, c in enumerate(rbf_centers):
        # RBF_j(x) = exp(-((x - c)^2) / (2 * width^2))
        basis[:, 6 + j] = np.exp(-((x - c) ** 2) / (2.0 * rbf_width ** 2))
    
    return basis @ params
# EVOLVE-BLOCK-END
