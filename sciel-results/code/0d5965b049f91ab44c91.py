"""
Chebyshev Polynomial Basis approach.
Uses a full basis of Chebyshev polynomials in x and y up to total degree 3.
This provides 10 basis functions:
1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3 (mapped to Chebyshev for better conditioning).
Linear in the 10 parameters, allowing BFGS to optimize efficiently.
Global basis functions are often more robust than local RBFs with fixed centers
when the number of parameters is small.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_LIMIT = 150.0
_MARGIN = 10.0

def _elapsed_ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure 1-D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    n = x.size

    if not _elapsed_ok():
        # Safety fallback: return mean-ish constant
        return np.full(n, float(p[0]))

    # Chebyshev polynomials of the first kind:
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    # We construct a basis of total degree <= 3.
    # The terms are T_i(x) * T_j(y) where i + j <= 3.
    # i=0: T0(x)T0(y), T0(x)T1(y), T0(x)T2(y), T0(x)T3(y) -> 4 terms
    # i=1: T1(x)T0(y), T1(x)T1(y), T1(x)T2(y) -> 3 terms
    # i=2: T2(x)T0(y), T2(x)T1(y) -> 2 terms
    # i=3: T3(x)T0(y) -> 1 term
    # Total: 4 + 3 + 2 + 1 = 10 terms.

    # Compute Chebyshev values for x and y
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2 * x**2 - 1
    Tx3 = 4 * x**3 - 3 * x

    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2 * y**2 - 1
    Ty3 = 4 * y**3 - 3 * y

    # Construct the basis matrix (n, 10)
    # Order:
    # 0: T0(x)T0(y)
    # 1: T0(x)T1(y)
    # 2: T0(x)T2(y)
    # 3: T0(x)T3(y)
    # 4: T1(x)T0(y)
    # 5: T1(x)T1(y)
    # 6: T1(x)T2(y)
    # 7: T2(x)T0(y)
    # 8: T2(x)T1(y)
    # 9: T3(x)T0(y)
    
    basis = np.column_stack((
        Tx0 * Ty0,
        Tx0 * Ty1,
        Tx0 * Ty2,
        Tx0 * Ty3,
        Tx1 * Ty0,
        Tx1 * Ty1,
        Tx1 * Ty2,
        Tx2 * Ty0,
        Tx2 * Ty1,
        Tx3 * Ty0
    ))

    # Linear combination
    return basis @ p
# EVOLVE-BLOCK-END
