import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    
    Uses a hybrid basis of Gaussian Radial Basis Functions (RBFs) and low-degree polynomials.
    This structure allows the model to capture both local non-linear features (via RBFs)
    and global trends (via polynomials). The model is linear in the 10 parameters,
    ensuring reliable convergence with BFGS.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # 1. Basis: Gaussian RBFs with fixed centers and widths
    # Centers chosen to cover the domain [-1, 1]x[-1, 1]
    # We use 5 RBFs with moderate width and 1 sharp RBF at the center
    
    # RBF 1: Center (0.0, 0.0), sigma = 0.8
    r1 = x**2 + y**2
    b0 = np.exp(-r1 / (2 * 0.8**2))
    
    # RBF 2: Center (0.5, 0.5), sigma = 0.6
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    b1 = np.exp(-r2 / (2 * 0.6**2))
    
    # RBF 3: Center (-0.5, -0.5), sigma = 0.6
    r3 = (x + 0.5)**2 + (y + 0.5)**2
    b2 = np.exp(-r3 / (2 * 0.6**2))
    
    # RBF 4: Center (0.5, -0.5), sigma = 0.6
    r4 = (x - 0.5)**2 + (y + 0.5)**2
    b3 = np.exp(-r4 / (2 * 0.6**2))
    
    # RBF 5: Center (-0.5, 0.5), sigma = 0.6
    r5 = (x + 0.5)**2 + (y - 0.5)**2
    b4 = np.exp(-r5 / (2 * 0.6**2))
    
    # RBF 6: Center (0.0, 0.0), sigma = 0.3 (Sharp peak for local features)
    b5 = np.exp(-r1 / (2 * 0.3**2))
    
    # 2. Basis: Low-degree Polynomials
    # These capture global linear and quadratic trends
    b6 = np.ones_like(x)       # Constant
    b7 = x                     # Linear x
    b8 = y                     # Linear y
    b9 = x * y                 # Interaction
    
    # Stack into a matrix
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    pred = basis @ p
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return pred
