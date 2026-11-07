import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression equation using a hybrid basis of Chebyshev polynomials
    and trigonometric functions.
    
    The basis is designed to be well-conditioned (Chebyshev) and to capture
    both polynomial-like and oscillatory behaviors (trig terms).
    """
    n = x.shape[0]
    
    # Scale x to [-1, 1] for better numerical stability of Chebyshev polynomials
    t = x / 2.0
    
    # Allocate basis matrix
    basis = np.zeros((n, 10))
    
    if n == 0:
        return np.array([])
    
    # Columns 0-4: Chebyshev polynomials of the first kind T_k(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2t*T_{k-1}(t) - T_{k-2}(t)
    basis[:, 0] = 1.0
    basis[:, 1] = t
    for k in range(2, 5):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Columns 5-9: Trigonometric functions to capture oscillatory behavior
    # sin(x), cos(x), sin(2x), cos(2x), x*sin(x)
    # Using original x (in [-2, 2]) for frequencies 1 and 2
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.cos(x)
    basis[:, 7] = np.sin(2.0 * x)
    basis[:, 8] = np.cos(2.0 * x)
    basis[:, 9] = x * np.sin(x)
    
    # Linear combination
    pred = basis @ params
    
    # Safety guard: replace non-finite values
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
