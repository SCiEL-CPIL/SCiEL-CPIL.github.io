import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a 10-parameter Composite Basis.
    
    The model is a linear combination of:
    - Polynomial terms: 1, x, x^2, x^3
    - Trigonometric terms: sin(x), cos(x), sin(2x), cos(2x)
    - Hyperbolic tangent terms: tanh(x), tanh(2x)
    
    This diverse basis allows the model to capture polynomial trends,
    periodic behavior, and sigmoidal/step-like transitions, providing
    more flexibility than a single RBF family while remaining a
    linear-in-weights model that BFGS can solve robustly.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if params.shape[0] != 10:
        # Fallback if shape mismatch occurs (should not happen per contract)
        return np.zeros_like(x)
        
    # Construct the basis matrix
    # Shape: (N, 10)
    
    # 1. Polynomial basis
    p1 = np.ones_like(x)
    p2 = x
    p3 = x**2
    p4 = x**3
    
    # 2. Trigonometric basis
    t1 = np.sin(x)
    t2 = np.cos(x)
    t3 = np.sin(2 * x)
    t4 = np.cos(2 * x)
    
    # 3. Hyperbolic tangent basis
    h1 = np.tanh(x)
    h2 = np.tanh(2 * x)
    
    # Stack basis functions into a matrix
    # Rows are samples, columns are basis functions
    basis = np.column_stack((
        p1, p2, p3, p4, 
        t1, t2, t3, t4, 
        h1, h2
    ))
    
    # Linear combination
    # basis is (N, 10), params is (10,)
    # Result is (N,)
    prediction = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(x)
        
    return prediction
