import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using a physics-inspired monomial basis.
    
    This basis includes terms corresponding to:
    - Linear and cubic restoring forces (Duffing oscillator)
    - Linear and cubic damping
    - Amplitude-dependent damping (Van der Pol oscillator)
    - Coupling terms
    
    Inputs are normalized to improve numerical conditioning for BFGS.
    """
    # Normalize inputs to approximately [-1, 1]
    # Training ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    X = x / 0.6
    V = v / 0.3
    
    # Precompute powers for efficiency and readability
    X2 = X * X
    X3 = X2 * X
    X5 = X3 * X2
    
    V2 = V * V
    V3 = V2 * V
    
    # Basis terms
    # 1. Constant
    t0 = 1.0
    # 2. Linear restoring force
    t1 = X
    # 3. Cubic restoring force
    t2 = X3
    # 4. Linear damping
    t3 = V
    # 5. Cubic damping
    t4 = V3
    # 6. Amplitude-dependent damping (Van der Pol: x^2 * v)
    t5 = X2 * V
    # 7. Linear coupling
    t6 = X * V
    # 8. Quadratic position bias
    t7 = X2
    # 9. Quadratic velocity bias
    t8 = V2
    # 9. Quintic restoring force (strong nonlinearity)
    t9 = X5
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Combine terms
    return (p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3 + p4 * t4 +
            p5 * t5 + p6 * t6 + p7 * t7 + p8 * t8 + p9 * t9)
