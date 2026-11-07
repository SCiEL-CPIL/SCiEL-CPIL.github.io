"""Symbolic regression: recover hidden function from 2 inputs using rich basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build a feature matrix from x, y using a rich set of basis functions.
    Returns shape (n_points, n_features).
    """
    x = x.astype(float)
    y = y.astype(float)
    n = x.shape[0]
    
    # Polynomial terms up to degree 3
    f = []
    f.append(np.ones(n))           # 0: constant
    f.append(x)                    # 1: x
    f.append(y)                    # 2: y
    f.append(x * y)                # 3: x*y
    f.append(x**2)                 # 4: x^2
    f.append(y**2)                 # 5: y^2
    f.append(x**3)                 # 6: x^3
    f.append(y**3)                 # 7: y^3
    f.append(x**2 * y)             # 8: x^2*y
    f.append(x * y**2)             # 9: x*y^2
    
    # Trigonometric terms
    f.append(np.sin(x))            # 10: sin(x)
    f.append(np.cos(x))            # 11: cos(x)
    f.append(np.sin(y))            # 12: sin(y)
    f.append(np.cos(y))            # 13: cos(y)
    f.append(np.sin(x + y))        # 14: sin(x+y)
    f.append(np.cos(x + y))        # 15: cos(x+y)
    f.append(np.sin(x - y))        # 16: sin(x-y)
    f.append(np.cos(x - y))        # 17: cos(x-y)
    
    # Exponential / radial terms
    f.append(np.exp(-x**2))        # 18: exp(-x^2)
    f.append(np.exp(-y**2))        # 19: exp(-y^2)
    f.append(np.exp(-(x**2 + y**2))) # 20: exp(-(x^2+y^2))
    
    # Logarithmic-ish (safe) and absolute value terms
    f.append(np.abs(x))            # 21: |x|
    f.append(np.abs(y))            # 22: |y|
    f.append(np.abs(x * y))        # 23: |x*y|
    
    # Sigmoid-like
    f.append(1.0 / (1.0 + np.exp(-5*x)))   # 24: sigmoid(5x)
    f.append(1.0 / (1.0 + np.exp(-5*y)))   # 25: sigmoid(5y)
    
    # Higher-order interactions
    f.append(x**2 * y**2)          # 26: x^2*y^2
    f.append(np.sin(x) * np.cos(y)) # 27: sin(x)*cos(y)
    f.append(np.cos(x) * np.sin(y)) # 28: cos(x)*sin(y)
    
    # Additional polynomial
    f.append(x**4)                 # 29: x^4
    f.append(y**4)                 # 30: y^4
    f.append(x**3 * y)             # 31: x^3*y
    f.append(x * y**3)             # 32: x*y^3
    f.append(x**2 * y**2 * x)      # 33: x^3*y^2
    f.append(x**2 * y**2 * y)      # 34: x^2*y^3
    
    # More trig
    f.append(np.sin(2*x))          # 35: sin(2x)
    f.append(np.cos(2*x))          # 36: cos(2x)
    f.append(np.sin(2*y))          # 37: sin(2y)
    f.append(np.cos(2*y))          # 38: cos(2y)
    
    # Radial and angular
    r = np.sqrt(x**2 + y**2)
    f.append(r)                    # 39: r
    f.append(r**2)                 # 40: r^2 (=x^2+y^2, already have exp but not linear)
    f.append(np.sin(r))            # 41: sin(r)
    f.append(np.cos(r))            # 42: cos(r)
    
    # Product of sigmoids
    f.append((1.0 / (1.0 + np.exp(-5*x))) * (1.0 / (1.0 + np.exp(-5*y)))) # 43
    
    # x*y * r
    f.append(x * y * r)            # 44
    
    # Clip to avoid extreme values
    feat_matrix = np.column_stack(f)
    # Replace any non-finite values
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feat_matrix


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a linear combination of rich basis functions.
    params: array of length 10 (as required by evaluator).
    We use the first 10 parameters to weight the first 10 basis functions,
    but we can also use a fixed mapping. Since params has length 10,
    we'll use those 10 to weight a selected subset of features.
    
    Actually, let's use all features but with only 10 free parameters.
    Strategy: use the 10 params to weight the 10 most important features,
    and set the rest to zero or fixed values.
    
    Better: Use the 10 params as weights for a carefully chosen 10-dim basis.
    """
    x = x.astype(float)
    y = y.astype(float)
    n = x.shape[0]
    
    # Build a 10-dimensional basis that covers the most likely function families
    # The 10 params will weight these 10 features
    b0 = np.ones(n)
    b1 = x
    b2 = y
    b3 = x * y
    b4 = x**2
    b5 = y**2
    b6 = np.sin(x)
    b7 = np.sin(y)
    b8 = np.exp(-(x**2 + y**2))
    b9 = x**2 * y**2
    
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    # Ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return pred
# EVOLVE-BLOCK-END
