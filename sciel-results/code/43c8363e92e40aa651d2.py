"""
Chebyshev product basis for symbolic regression.
Uses T_i(x) * T_j(y) terms to form a well-conditioned, near-orthogonal
design matrix that efficiently captures smooth bivariate functions.
The 10 basis functions are chosen to span low-order Chebyshev products:
  T0(x)T0(y), T1(x)T0(y), T0(x)T1(y),
  T2(x)T0(y), T0(x)T2(y), T1(x)T1(y),
  T3(x)T0(y), T0(x)T3(y), T2(x)T1(y), T1(x)T2(y)
This avoids monomial collinearity while capturing polynomial and
near-periodic structure within 10 parameters.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 15.0


def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE


def _time_ok():
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _SAFETY_MARGIN


def _chebyshev_t(n: int, x: np.ndarray) -> np.ndarray:
    """Compute the n-th Chebyshev polynomial of the first kind T_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    t_nm2 = np.ones_like(x)
    t_nm1 = x.copy()
    for _ in range(2, n + 1):
        t_n = 2.0 * x * t_nm1 - t_nm2
        t_nm2, t_nm1 = t_nm1, t_n
    return t_nm1


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction using 10 learned Chebyshev-product parameters."""
    _init_deadline()

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    if params.shape[0] < 10:
        p = np.ones(10, dtype=np.float64)
        p[:min(10, len(params))] = params[:min(10, len(params))]
        params = p

    if not _time_ok():
        # Safety fallback: simple linear combination
        return params[0] + params[1] * x + params[2] * y

    n = len(x)
    xc = np.clip(x, -1.0, 1.0)
    yc = np.clip(y, -1.0, 1.0)

    # Precompute Chebyshev polynomials up to degree 3
    T0x = np.ones(n)
    T1x = xc.copy()
    T2x = 2.0 * xc * xc - 1.0
    T3x = 4.0 * xc * xc * xc - 3.0 * xc

    T0y = np.ones(n)
    T1y = yc.copy()
    T2y = 2.0 * yc * yc - 1.0
    T3y = 4.0 * yc * yc * yc - 3.0 * yc

    # Build the 10 Chebyshev product basis functions
    t0 = T0x * T0y       # 1
    t1 = T1x * T0y       # x
    t2 = T0x * T1y       # y
    t3 = T2x * T0y       # 2x^2 - 1
    t4 = T0x * T2y       # 2y^2 - 1
    t5 = T1x * T1y       # x*y
    t6 = T3x * T0y       # 4x^3 - 3x
    t7 = T0x * T3y       # 4y^3 - 3y
    t8 = T2x * T1y       # (2x^2-1)*y
    t9 = T1x * T2y       # x*(2y^2-1)

    # Stack into feature matrix (n, 10)
    F = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])

    # Ensure finite values
    F = np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=-1.0)

    # Linear combination
    pred = F @ params[:10]

    # Ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

    return pred
# EVOLVE-BLOCK-END
