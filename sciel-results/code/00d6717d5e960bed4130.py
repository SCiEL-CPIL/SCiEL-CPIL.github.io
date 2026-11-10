import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear
    b1 = x
    # 3. Quadratic
    b2 = x * x
    # 4. Cubic
    b3 = b2 * x
    # 5. Quartic
    b4 = b2 * b2
    
    # 6. Sine
    b5 = np.sin(x)
    # 7. Cosine
    b6 = np.cos(x)
    # 8. Hyperbolic Sine
    b7 = np.sinh(x)
    # 9. Hyperbolic Cosine
    b8 = np.cosh(x)
    # 10. Gaussian decay
    b9 = np.exp(-x * x)
    
    # Linear combination
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
            p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
