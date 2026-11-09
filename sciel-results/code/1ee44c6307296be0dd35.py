"""
Hybrid Chebyshev-RBF basis: 7 Chebyshev product terms (well-conditioned, global smooth)
plus 3 RBF Gaussian kernels (localized, captures non-polynomial features).
The Chebyshev terms T_i(x)*T_j(y) are orthogonal on [-1,1] providing a stable
linear-in-parameters model. The RBF terms add localized flexibility.
All 10 basis functions are precomputed per call; the model is linear in params
so BFGS converges efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 5.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), constant_values=0.0)
    else:
        p = p[:10]
    
    # Safety: if close to deadline, return simple fallback
    if _time_remaining() < _SAFETY_MARGIN:
        return p[0] + p[1] * x + p[2] * y
    
    n = x.shape[0]
    
    # --- Chebyshev polynomials T_0..T_4 on [-1,1] ---
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    
    Tx0 = np.ones_like(x)
    Tx1 = x.copy()
    Tx2 = 2.0 * x * x - 1.0
    Tx3 = 4.0 * x * x * x - 3.0 * x
    Tx4 = 8.0 * x**4 - 8.0 * x * x + 1.0
    
    Ty0 = np.ones_like(y)
    Ty1 = y.copy()
    Ty2 = 2.0 * y * y - 1.0
    Ty3 = 4.0 * y * y * y - 3.0 * y
    Ty4 = 8.0 * y**4 - 8.0 * y * y + 1.0
    
    # 7 Chebyshev product terms:
    # b0 = T0(x)*T0(y) = 1  (intercept)
    # b1 = T1(x)*T0(y) = x
    # b2 = T0(x)*T1(y) = y
    # b3 = T1(x)*T1(y) = x*y
    # b4 = T2(x)*T0(y) = 2x^2-1
    # b5 = T0(x)*T2(y) = 2y^2-1
    # b6 = T1(x)*T2(y) = x*(2y^2-1)  (captures x^2*y type interactions via mixing)
    
    b0 = Tx0 * Ty0
    b1 = Tx1 * Ty0
    b2 = Tx0 * Ty1
    b3 = Tx1 * Ty1
    b4 = Tx2 * Ty0
    b5 = Tx0 * Ty2
    b6 = Tx1 * Ty2
    
    # 3 RBF Gaussian kernels at strategic centers:
    # Center at origin (0,0) - captures radial/global structure
    # Center at (0.5, 0.5) - captures positive quadrant structure
    # Center at (-0.5, -0.5) - captures negative quadrant structure
    # Width chosen to cover ~half the domain
    
    width = 0.7
    
    # RBF 1: center (0, 0)
    r2_1 = x * x + y * y
    b7 = np.exp(-r2_1 / (2.0 * width * width))
    
    # RBF 2: center (0.5, 0.5)
    dx2 = x - 0.5
    dy2 = y - 0.5
    r2_2 = dx2 * dx2 + dy2 * dy2
    b8 = np.exp(-r2_2 / (2.0 * width * width))
    
    # RBF 3: center (-0.5, -0.5)
    dx3 = x + 0.5
    dy3 = y + 0.5
    r2_3 = dx3 * dx3 + dy3 * dy3
    b9 = np.exp(-r2_3 / (2.0 * width * width))
    
    # Stack basis and compute prediction
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    pred = basis @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return pred
# EVOLVE-BLOCK-END
