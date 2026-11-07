# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Degree-3 bivariate polynomial model.
    Basis functions:
    1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    params[0:10] are the coefficients for these terms.
    This is a linear model in params, allowing BFGS to efficiently find the
    optimal coefficients (equivalent to linear least squares).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has length 10
    if p.size < 10:
        p = np.concatenate([p, np.ones(10 - p.size)])
    p = p[:10]

    # Ensure x is 2D (N, 2)
    if x.ndim == 1:
        # If x is 1D, it's ambiguous. But task says 2 inputs.
        # Assume standard format is (N, 2). If (2, N) is passed, we might need to transpose.
        # However, usually x is (N, D). Let's assume (N, 2).
        # If x.shape[0] == 2 and x.shape[1] != 2, maybe it's transposed?
        # Let's add a safety check for shape.
        if x.size % 2 == 0:
             x = x.reshape(-1, 2)
        else:
             # Fallback: pad or truncate? Just return zeros if shape is weird.
             return np.zeros(y.shape)

    # If x is (2, N) instead of (N, 2)?
    # If x.shape[0] == 2 and x.shape[1] != 2:
    #    x = x.T
    # Let's assume standard (N, 2). If the error persists, we might need to check.
    # The previous error was index out of bounds on axis 0 size 2.
    # If x was (2, 200), x - cx[m] (scalar) -> (2, 200).
    # The error was in cx[m]. So the shape of x wasn't the primary cause of the *index* error, the cx length was.
    # But let's be safe.
    if x.shape[0] == 2 and x.shape[1] != 2:
        x = x.T

    n = x.shape[0]
    if n == 0:
        return np.zeros(0)

    x1 = x[:, 0]
    x2 = x[:, 1]

    # Compute basis functions
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: xy
    # 5: y^2
    # 6: x^3
    # 7: x^2 y
    # 8: x y^2
    # 9: y^3
    
    # Vectorized calculation
    term0 = np.ones(n)
    term1 = x1
    term2 = x2
    term3 = x1 * x1
    term4 = x1 * x2
    term5 = x2 * x2
    term6 = term3 * x1
    term7 = term3 * x2
    term8 = x1 * term5
    term9 = term5 * x2

    # Combine with params
    # pred = p[0]*term0 + p[1]*term1 + ...
    pred = (p[0] * term0 + 
            p[1] * term1 + 
            p[2] * term2 + 
            p[3] * term3 + 
            p[4] * term4 + 
            p[5] * term5 + 
            p[6] * term6 + 
            p[7] * term7 + 
            p[8] * term8 + 
            p[9] * term9)
            
    return pred

# EVOLVE-BLOCK-END
