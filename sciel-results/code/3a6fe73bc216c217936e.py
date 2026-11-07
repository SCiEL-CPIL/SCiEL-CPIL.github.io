import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y.
    Uses a rich basis of polynomial, trigonometric, and Gaussian RBF terms,
    all linear in 10 parameters so BFGS can fit them reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # --- Polynomial basis (up to degree 3) ---
    b1 = np.ones_like(x)          # bias
    b2 = x                        # linear x
    b3 = y                        # linear y
    b4 = x * y                    # interaction
    b5 = x * x                     # quadratic x
    b6 = y * y                     # quadratic y
    b7 = x * x * y                 # cubic mixed
    b8 = x * y * y                 # cubic mixed
    
    # --- Trigonometric basis ---
    b9 = np.sin(np.pi * x)         # sinusoidal in x
    b10 = np.sin(np.pi * y)        # sinusoidal in y
    
    # Combine linearly
    pred = (p[0] * b1 + 
            p[1] * b2 + 
            p[2] * b3 + 
            p[3] * b4 + 
            p[4] * b5 + 
            p[5] * b6 + 
            p[6] * b7 + 
            p[7] * b8 + 
            p[8] * b9 + 
            p[9] * b10)
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
