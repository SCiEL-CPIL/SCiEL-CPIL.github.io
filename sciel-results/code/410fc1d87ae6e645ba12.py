import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev Polynomial basis of degree <= 3.
    
    The model is linear in 10 parameters, which are fitted by the external BFGS optimizer.
    The basis functions are chosen to be orthogonal on the domain [-1, 1], 
    ensuring a well-conditioned design matrix and fast convergence.
    
    Basis functions (10 terms spanning polynomials of degree <= 3):
    0: 1
    1: x
    2: y
    3: T2(x) = 2x^2 - 1
    4: T2(y) = 2y^2 - 1
    5: x*y
    6: T3(x) = 4x^3 - 3x
    7: T3(y) = 4y^3 - 3y
    8: T2(x)*T1(y) = (2x^2 - 1)y = 2x^2y - y
    9: T1(x)*T2(y) = x(2y^2 - 1) = 2xy^2 - x
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))

    # Compute Chebyshev polynomials
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    T0_x = np.ones(n)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones(n)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Construct the 10 basis vectors
    # Order:
    # 0: T0_x * T0_y
    # 1: T1_x * T0_y
    # 2: T0_x * T1_y
    # 3: T2_x * T0_y
    # 4: T0_x * T2_y
    # 5: T1_x * T1_y
    # 6: T3_x * T0_y
    # 7: T0_x * T3_y
    # 8: T2_x * T1_y
    # 9: T1_x * T2_y
    
    basis = np.column_stack([
        T0_x * T0_y,  # 1
        T1_x * T0_y,  # x
        T0_x * T1_y,  # y
        T2_x * T0_y,  # 2x^2 - 1
        T0_x * T2_y,  # 2y^2 - 1
        T1_x * T1_y,  # xy
        T3_x * T0_y,  # 4x^3 - 3x
        T0_x * T3_y,  # 4y^3 - 3y
        T2_x * T1_y,  # (2x^2-1)y
        T1_x * T2_y   # x(2y^2-1)
    ])
    
    # Linear combination
    out = basis @ p
    
    # Safety check
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
    return out
