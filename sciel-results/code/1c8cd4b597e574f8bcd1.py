"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: polynomial terms (5) + adaptive sinusoids (3) + Gaussian RBF (2).
    Total: 10 parameters.
    
    Polynomial basis (indices 0-4):
      1, x, y, x^2, y^2
    
    Sinusoid basis (indices 5-7):
      sin(a*x + b*y), sin(c*x + d*y), sin(e*x + f*y)
      where frequencies are fixed but chosen to cover common patterns.
    
    RBF basis (indices 8-9):
      Two Gaussian RBFs at strategic locations.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    N = len(x)
    
    # Validate params
    if not np.all(np.isfinite(params)):
        return np.zeros(N)
    
    # --- Polynomial basis (5 terms) ---
    # 1, x, y, x^2, y^2
    poly_basis = np.column_stack([
        np.ones(N),
        x,
        y,
        x ** 2,
        y ** 2
    ])
    
    # --- Sinusoid basis (3 terms) ---
    # Use fixed frequencies that cover common patterns in [-1,1]
    # sin(x + y), sin(2*x), sin(2*y), sin(x - y), sin(3*x + 2*y)
    # Pick 3 diverse ones
    sin1 = np.sin(x + y)
    sin2 = np.sin(2.0 * x)
    sin3 = np.sin(2.0 * y)
    
    # Add more flexibility: sin(x - y) and sin(3x + 2y) as alternatives
    # Let's use: sin(x+y), sin(2x), sin(2y) -- these are orthogonal-ish
    sin_basis = np.column_stack([
        sin1,
        sin2,
        sin3
    ])
    
    # --- RBF basis (2 terms) ---
    # Two Gaussian RBFs at center and corners
    # Center: (0, 0), Corners: (0.5, 0.5)
    width = 0.6
    rbf1 = np.exp(-(x ** 2 + y ** 2) / (2.0 * width ** 2))
    rbf2 = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2.0 * width ** 2))
    
    rbf_basis = np.column_stack([
        rbf1,
        rbf2
    ])
    
    # Combine all basis functions
    basis = np.column_stack([poly_basis, sin_basis, rbf_basis])
    
    # Predict: basis @ params
    pred = basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros(N)
    
    return pred
# EVOLVE-BLOCK-END
