import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params[0] is coeff for x^0, params[1] for x^1, etc.
    # We use Horner's method or explicit sum for stability/speed.
    # Explicit sum is fine for degree 9.
    result = np.zeros_like(x)
    for i in range(10):
        result += params[i] * (x ** i)
    return result
