import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis for nonlinear oscillator.
    Uses scaled inputs mapped to [-1, 1] for Chebyshev polynomials.
    This provides superior approximation properties for bounded domains.
    """
    # Scale inputs to [-1, 1] for Chebyshev polynomials
    # x range: [-0.5728, 0.5275], center = -0.02265, half_range = 0.55015
    # v range: [-0.1754, 0.2821], center = 0.05335, half_range = 0.22875
    
    x_center = -0.02265
    x_half = 0.55015
    v_center = 0.05335
    v_half = 0.22875
    
    X = (x - x_center) / x_half
    V = (v - v_center) / v_half
    
    # Compute Chebyshev polynomials up to degree 4 for X and V
    # T_0 = 1
    # T_1 = z
    # T_2 = 2z^2 - 1
    # T_3 = 4z^3 - 3z
    # T_4 = 8z^4 - 8z^2 + 1
    
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    
    V2 = V * V
    V3 = V2 * V
    V4 = V2 * V2
    
    # Chebyshev polynomials for X
    TX0 = np.ones_like(x)
    TX1 = X
    TX2 = 2 * X2 - 1
    TX3 = 4 * X3 - 3 * X
    TX4 = 8 * X4 - 8 * X2 + 1
    
    # Chebyshev polynomials for V
    TV0 = np.ones_like(v)
    TV1 = V
    TV2 = 2 * V2 - 1
    TV3 = 4 * V3 - 3 * V
    TV4 = 8 * V4 - 8 * V2 + 1
    
    # Basis terms:
    # 0: T_0(X) = 1
    # 1: T_1(X) = X
    # 2: T_2(X)
    # 3: T_3(X)
    # 4: T_1(V) = V
    # 5: T_2(V)
    # 6: T_3(V)
    # 7: X*V (cross term)
    # 8: T_2(X)*T_1(V)
    # 9: T_1(X)*T_2(V)
    
    t0 = TX0
    t1 = TX1
    t2 = TX2
    t3 = TX3
    t4 = TV1
    t5 = TV2
    t6 = TV3
    t7 = X * V
    t8 = TX2 * TV1
    t9 = TX1 * TV2
    
    T = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])
    
    return T @ params
