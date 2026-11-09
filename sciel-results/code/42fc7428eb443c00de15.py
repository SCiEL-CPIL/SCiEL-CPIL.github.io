import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    # Basis: 1, x, y, x^2+y^2, x*y, sin(x), sin(y), cos(x+y), x^3+y^3, x^2*y^2
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x**2 + y**2
    term4 = x * y
    term5 = np.sin(x)
    term6 = np.sin(y)
    term7 = np.cos(x + y)
    term8 = x**3 + y**3
    term9 = x**2 * y**2

    return (p[0] * term0 +
            p[1] * term1 +
            p[2] * term2 +
            p[3] * term3 +
            p[4] * term4 +
            p[5] * term5 +
            p[6] * term6 +
            p[7] * term7 +
            p[8] * term8 +
            p[9] * term9)
