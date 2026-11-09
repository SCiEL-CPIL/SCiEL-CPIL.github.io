import numpy as np

def _chebyshev_basis(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Build a basis of Chebyshev polynomials of the first kind mapped to [0.5, 2].
    
    For each input variable, we map t in [0.5, 2] to u in [-1, 1] via:
        u = (2*t - (0.5+2)) / (2 - 0.5) = (2*t - 2.5) / 1.5
    
    Then we compute T_0(u)=1, T_1(u)=u, T_2(u)=2u^2-1, T_3(u)=4u^3-3u.
    
    We build a tensor-product-like basis but select 10 terms that capture:
    - Constant
    - Linear in each variable (via T_1)
    - Quadratic in each variable (via T_2)
    - Interaction terms (products of T_1 from different variables)
    
    Specifically, the 10 basis functions are:
    0: T_0(x)*T_0(y)*T_0(z) = 1
    1: T_1(x)*T_0(y)*T_0(z)
    2: T_0(x)*T_1(y)*T_0(z)
    3: T_0(x)*T_0(y)*T_1(z)
    4: T_2(x)*T_0(y)*T_0(z)
    5: T_0(x)*T_2(y)*T_0(z)
    6: T_0(x)*T_0(y)*T_2(z)
    7: T_1(x)*T_1(y)*T_0(z)
    8: T_1(x)*T_0(y)*T_1(z)
    9: T_0(x)*T_1(y)*T_1(z)
    """
    # Map inputs to [-1, 1]
    ux = (2.0 * x - 2.5) / 1.5
    uy = (2.0 * y - 2.5) / 1.5
    uz = (2.0 * z - 2.5) / 1.5
    
    # Chebyshev polynomials of the first kind
    T0_x = np.ones_like(ux)
    T1_x = ux
    T2_x = 2.0 * ux**2 - 1.0
    
    T0_y = np.ones_like(uy)
    T1_y = uy
    T2_y = 2.0 * uy**2 - 1.0
    
    T0_z = np.ones_like(uz)
    T1_z = uz
    T2_z = 2.0 * uz**2 - 1.0
    
    # Build the 10 basis functions
    basis = np.column_stack([
        T0_x * T0_y * T0_z,
        T1_x * T0_y * T0_z,
        T0_x * T1_y * T0_z,
        T0_x * T0_y * T1_z,
        T2_x * T0_y * T0_z,
        T0_x * T2_y * T0_z,
        T0_x * T0_y * T2_z,
        T1_x * T1_y * T0_z,
        T1_x * T0_y * T1_z,
        T0_x * T1_y * T1_z
    ])
    
    return basis

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Chebyshev polynomial basis.
    
    The model is a linear combination of 10 Chebyshev-based basis functions.
    Chebyshev polynomials provide superior numerical conditioning compared to
    monomials, enabling BFGS to fit more complex functions stably.
    
    params: array of 10 coefficients for the basis functions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape[0] != 10:
        # Fallback: try to handle different param lengths gracefully
        p = np.resize(p, 10)
    
    # Build the Chebyshev basis
    B = _chebyshev_basis(x, y, z)
    
    # Linear prediction
    pred = B @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
