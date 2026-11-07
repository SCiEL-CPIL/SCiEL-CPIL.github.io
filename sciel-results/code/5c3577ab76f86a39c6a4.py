import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a rational/transformed basis.
    
    This uses a different structural family: instead of pure polynomials or Chebyshev,
    we use:
    1. tanh-saturated terms for bounded nonlinearities
    2. Rational terms (x/(1+x^2) style) for peaked responses
    3. Mixed polynomial terms for flexibility
    
    The key insight is that oscillator dynamics often have bounded responses and 
    saturation effects that polynomials approximate poorly.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to roughly [-1, 1]
    x_center = -0.02265
    x_scale = 0.55015
    v_center = 0.05335
    v_scale = 0.22875
    
    X = (x - x_center) / x_scale
    V = (v - v_center) / v_scale
    
    # Clip to avoid extreme values
    X = np.clip(X, -5.0, 5.0)
    V = np.clip(V, -5.0, 5.0)
    
    # Precompute common terms
    X2 = X * X
    V2 = V * V
    X3 = X2 * X
    V3 = V2 * V
    
    # Basis terms - mixing different functional families:
    
    # b0: constant
    b0 = 1.0
    
    # b1: linear restoring force (x)
    b1 = X
    
    # b2: tanh-saturated cubic (bounded nonlinearity)
    b2 = np.tanh(X3)
    
    # b3: linear damping (v)
    b3 = V
    
    # b4: rational peaked term in x (like x/(1+x^2))
    b4 = X / (1.0 + X2)
    
    # b5: cross coupling x*v
    b5 = X * V
    
    # b6: tanh of v (saturated damping)
    b6 = np.tanh(V)
    
    # b7: quadratic cross term
    b7 = X2 * V
    
    # b8: rational term in v
    b8 = V / (1.0 + V2)
    
    # b9: mixed term x^2 * v^2 (higher order coupling)
    b9 = X2 * V2
    
    # Linear combination
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + 
              p4 * b4 + p5 * b5 + p6 * b6 + p7 * b7 + 
              p8 * b8 + p9 * b9)
    
    return result
