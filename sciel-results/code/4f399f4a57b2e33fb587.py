"""Symbolic regression with 2 inputs: Chebyshev tensor-product basis (degree 3) with periodic augmentation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix.
    
    Strategy: Combine Chebyshev polynomial basis (well-conditioned, bounded on [-1,1])
    with periodic terms to capture both smooth polynomial and oscillatory behavior.
    
    Features:
    1. T0(x)T0(y) = 1
    2. T1(x)T0(y) = x
    3. T0(x)T1(y) = y
    4. T2(x)T0(y) = 2x^2 - 1
    5. T1(x)T1(y) = xy
    6. T0(x)T2(y) = 2y^2 - 1
    7. sin(x)
    8. sin(y)
    9. T3(x)T0(y) = 4x^3 - 3x
    10. T0(x)T3(y) = 4y^3 - 3y
    
    Chebyshev polynomials provide well-conditioned, bounded terms for polynomial
    approximation, while sin(x) and sin(y) capture periodic components that
    polynomials alone cannot represent efficiently.
    """
    n = x.shape[0]

    # Compute Chebyshev polynomials for x and y
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    x2 = x * x
    y2 = y * y
    
    Tx0 = np.ones(n)
    Tx1 = x
    Tx2 = 2.0 * x2 - 1.0
    Tx3 = 4.0 * x2 * x - 3.0 * x

    Ty0 = np.ones(n)
    Ty1 = y
    Ty2 = 2.0 * y2 - 1.0
    Ty3 = 4.0 * y2 * y - 3.0 * y

    # Construct the 10 features: mix of Chebyshev and periodic terms
    f0 = Tx0 * Ty0          # 1
    f1 = Tx1 * Ty0          # x
    f2 = Tx0 * Ty1          # y
    f3 = Tx2 * Ty0          # 2x^2 - 1
    f4 = Tx1 * Ty1          # xy
    f5 = Tx0 * Ty2          # 2y^2 - 1
    f6 = np.sin(x)          # sin(x)
    f7 = np.sin(y)          # sin(y)
    f8 = Tx3 * Ty0          # 4x^3 - 3x
    f9 = Tx0 * Ty3          # 4y^3 - 3y

    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
