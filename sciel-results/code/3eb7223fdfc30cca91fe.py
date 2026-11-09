import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10-term basis combining polynomial and trigonometric functions
    # 0: 1 (intercept)
    # 1: x
    # 2: y
    # 3: x*y
    # 4: x^2
    # 5: y^2
    # 6: sin(x + y)
    # 7: cos(x + y)
    # 8: sin(x - y)
    # 9: cos(x - y)

    # Compute the necessary components
    xy = x * y
    x2 = x * x
    y2 = y * y

    # Trigonometric terms of linear combinations
    sx = x + y
    sy = x - y

    sin_sx = np.sin(sx)
    cos_sx = np.cos(sx)
    sin_sy = np.sin(sy)
    cos_sy = np.cos(sy)

    # Build prediction as a linear combination
    pred = (params[0] * np.ones_like(x) +
            params[1] * x +
            params[2] * y +
            params[3] * xy +
            params[4] * x2 +
            params[5] * y2 +
            params[6] * sin_sx +
            params[7] * cos_sx +
            params[8] * sin_sy +
            params[9] * cos_sy)

    return pred
