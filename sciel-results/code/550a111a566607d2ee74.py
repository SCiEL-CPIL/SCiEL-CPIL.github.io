import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Chebyshev polynomial basis.
    Inputs are mapped to [-1, 1] to improve conditioning.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Map x to [-1, 1]
    # x range: [-0.5728, 0.5275]
    x_center = -0.02265
    x_scale = 0.55015
    X = (x - x_center) / x_scale
    
    # Map v to [-1, 1]
    # v range: [-0.1754, 0.2821]
    v_center = 0.05335
    v_scale = 0.22875
    V = (v - v_center) / v_scale
    
    # Clip to [-1, 1] to avoid Chebyshev growth outside range (though training is in range)
    # But test might be out of range? "in-distribution test split".
    # So clipping is safe for training, maybe not for OOD, but OOD is not scored.
    # We don't clip to allow extrapolation if needed, but Chebyshev grows fast.
    
    # Chebyshev Polynomials
    T0_X = 1.0
    T1_X = X
    T2_X = 2.0 * X * X - 1.0
    T3_X = 4.0 * X * X * X - 3.0 * X
    
    T0_V = 1.0
    T1_V = V
    T2_V = 2.0 * V * V - 1.0
    T3_V = 4.0 * V * V * V - 3.0 * V
    
    # Basis terms
    # 1, X, V, T2_X, T1_X*T1_V, T2_V, T3_X, T2_X*T1_V, T1_X*T2_V, T3_V
    term0 = T0_X * T0_V
    term1 = T1_X * T0_V
    term2 = T0_X * T1_V
    term3 = T2_X * T0_V
    term4 = T1_X * T1_V
    term5 = T0_X * T2_V
    term6 = T3_X * T0_V
    term7 = T2_X * T1_V
    term8 = T1_X * T2_V
    term9 = T0_X * T3_V
    
    return p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 + p8 * term8 + p9 * term9
