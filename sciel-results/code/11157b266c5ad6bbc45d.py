import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a 2D Chebyshev polynomial basis.
    
    This structural change replaces the monomial basis with an orthogonal 
    Chebyshev basis. This improves numerical conditioning for the BFGS 
    optimizer by reducing feature correlation, allowing it to find a 
    better local minimum for the nonlinear function.
    
    Inputs are mapped to [-1, 1] to utilize standard Chebyshev properties.
    Basis includes all terms T_i(x)T_j(v) for i,j in {0,1,2} (9 terms) 
    plus T_3(x)T_0(v) as the 10th term to capture higher-order position 
    nonlinearity.
    """
    # Training ranges provided in task contract
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Calculate center and half-range for mapping to [-1, 1]
    x_center = (x_max + x_min) / 2.0
    x_half_range = (x_max - x_min) / 2.0
    
    v_center = (v_max + v_min) / 2.0
    v_half_range = (v_max - v_min) / 2.0
    
    # Map inputs to [-1, 1]
    # Add a small epsilon to avoid division by zero if range is 0 (not expected here)
    u = (x - x_center) / (x_half_range + 1e-8)
    w = (v - v_center) / (v_half_range + 1e-8)
    
    # Clip to [-1, 1] to ensure stability, though inputs are in range
    u = np.clip(u, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0, T_1, T_2, T_3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0_u = np.ones_like(u)
    T1_u = u
    T2_u = 2 * u**2 - 1
    T3_u = 4 * u**3 - 3 * u
    
    T0_w = np.ones_like(w)
    T1_w = w
    T2_w = 2 * w**2 - 1
    
    # Construct basis matrix
    # Order:
    # 0: T0_u * T0_w
    # 1: T1_u * T0_w
    # 2: T2_u * T0_w
    # 3: T0_u * T1_w
    # 4: T1_u * T1_w
    # 5: T2_u * T1_w
    # 6: T0_u * T2_w
    # 7: T1_u * T2_w
    # 8: T2_u * T2_w
    # 9: T3_u * T0_w
    
    basis = np.column_stack((
        T0_u * T0_w,
        T1_u * T0_w,
        T2_u * T0_w,
        T0_u * T1_w,
        T1_u * T1_w,
        T2_u * T1_w,
        T0_u * T2_w,
        T1_u * T2_w,
        T2_u * T2_w,
        T3_u * T0_w
    ))
    
    return basis @ params
