import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear basis model for symbolic regression.
    
    The model is: y = sum_{i=0}^{9} params[i] * basis_i(x)
    
    Because the model is linear in `params`, the objective function (MSE) is 
    a convex quadratic function. The external BFGS optimizer will find the 
    global minimum efficiently regardless of the starting point.
    
    Basis functions are chosen to be smooth, bounded or well-behaved on [-3, 3],
    and to cover common functional symmetries:
    0: 1.0
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: tanh(x)
    7: exp(-x^2)
    8: x * sin(x)
    9: x * cos(x)
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure p has length 10, pad with zeros if necessary (defensive)
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    # Compute basis functions
    # These are all O(1) or O(log N) operations, very fast.
    
    # 0: Constant
    f0 = np.ones_like(x)
    
    # 1: Linear
    f1 = x
    
    # 2: Quadratic
    f2 = x ** 2
    
    # 3: Cubic
    f3 = x ** 3
    
    # 4: Sine
    f4 = np.sin(x)
    
    # 5: Cosine
    f5 = np.cos(x)
    
    # 6: Hyperbolic tangent (smooth step-like behavior)
    f6 = np.tanh(x)
    
    # 7: Gaussian bump (localized feature)
    f7 = np.exp(-x ** 2)
    
    # 8: Modulated sine (captures phase shifts/oscillations with amplitude growth)
    f8 = x * np.sin(x)
    
    # 9: Modulated cosine
    f9 = x * np.cos(x)
    
    # Linear combination
    # Using np.dot or sum of products. 
    # For small N (200) and small D (10), explicit summation is fine and readable.
    # Alternatively: 
    # basis = np.vstack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]).T
    # return basis @ p
    
    # Explicit sum for clarity and slight potential speedup in numpy broadcasting
    pred = (
        p[0] * f0 +
        p[1] * f1 +
        p[2] * f2 +
        p[3] * f3 +
        p[4] * f4 +
        p[5] * f5 +
        p[6] * f6 +
        p[7] * f7 +
        p[8] * f8 +
        p[9] * f9
    )
    
    # Ensure output is finite. If params go wild during optimization, 
    # this might produce inf/nan. 
    # However, BFGS usually handles this by rejecting steps. 
    # If we return non-finite, the evaluator penalizes with 1e30.
    # We can clip extreme values to avoid overflow issues in subsequent steps?
    # The task says "A prediction of the wrong shape or with non-finite error costs 1e30".
    # It's safer to let BFGS see the gradient of the true function. 
    # But if x is large, x^3 is small (27). exp(-x^2) is small. 
    # The values are bounded. No overflow expected for x in [-3, 3].
    
    return pred
