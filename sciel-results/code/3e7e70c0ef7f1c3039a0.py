import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y using a Fourier/trigonometric basis.
    The 10 parameters are coefficients for low-frequency sine and cosine terms.
    Since the model is linear in params, BFGS (starting from ones) should
    converge cleanly to the global least-squares solution for this basis.
    """
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        return np.zeros_like(x)

    # Basis functions:
    # 0: 1 (bias)
    # 1: cos(x)
    # 2: sin(x)
    # 3: cos(y)
    # 4: sin(y)
    # 5: cos(x+y)
    # 6: sin(x+y)
    # 7: cos(x-y)
    # 8: sin(x-y)
    # 9: sin(2x)*cos(2y)  -- captures cross-frequency interaction

    c1 = np.cos(x)
    s1 = np.sin(x)
    c2 = np.cos(y)
    s2 = np.sin(y)
    c3 = np.cos(x + y)
    s3 = np.sin(x + y)
    c4 = np.cos(x - y)
    s4 = np.sin(x - y)
    s5 = np.sin(2.0 * x) * np.cos(2.0 * y)

    pred = (p[0] +
            p[1] * c1 +
            p[2] * s1 +
            p[3] * c2 +
            p[4] * s2 +
            p[5] * c3 +
            p[6] * s3 +
            p[7] * c4 +
            p[8] * s4 +
            p[9] * s5)

    if time.time() - t0 > DEADLINE:
        return pred

    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
