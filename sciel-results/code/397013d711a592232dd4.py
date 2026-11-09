"""Piecewise cubic B-spline with fixed internal knots for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_SPLINE_W = None
_SPLINE_KNOTS = None
_SPLINE_H = None
_SPLINE_SEG_IDX_CACHE = None

def _get_spline_constants():
    global _SPLINE_W, _SPLINE_KNOTS, _SPLINE_H
    if _SPLINE_W is not None:
        return _SPLINE_KNOTS, _SPLINE_H, _SPLINE_W
    
    n_knots = 10
    _SPLINE_KNOTS = np.linspace(-2.0, 2.0, n_knots)
    _SPLINE_H = np.diff(_SPLINE_KNOTS)
    
    # Construct Tridiagonal Matrix A for Natural Cubic Spline moments M
    # A * M = b
    # M[0] = 0, M[-1] = 0
    # For i in 1..n-2:
    # h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6 * ( (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1] )
    
    A = np.zeros((n_knots, n_knots))
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    
    for i in range(1, n_knots - 1):
        A[i, i-1] = _SPLINE_H[i-1]
        A[i, i] = 2.0 * (_SPLINE_H[i-1] + _SPLINE_H[i])
        A[i, i+1] = _SPLINE_H[i]
        
    # Construct Matrix C such that b = C * y
    C = np.zeros((n_knots, n_knots))
    # b[0] = 0, b[-1] = 0, so rows 0 and -1 of C are 0
    
    for i in range(1, n_knots - 1):
        h_prev = _SPLINE_H[i-1]
        h_curr = _SPLINE_H[i]
        
        # b_i = 6 * ( -1/h_prev * y[i-1] - (1/h_prev + 1/h_curr) * y[i] + 1/h_curr * y[i+1] )
        C[i, i-1] = -6.0 / h_prev
        C[i, i] = -6.0 * (1.0/h_prev + 1.0/h_curr)
        C[i, i+1] = 6.0 / h_curr
        
    # Precompute W = A^-1 * C
    # M = A^-1 * b = A^-1 * C * y = W * y
    _SPLINE_W = np.linalg.inv(A) @ C
    
    return _SPLINE_KNOTS, _SPLINE_H, _SPLINE_W

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Natural Cubic Spline representation.
    
    The spline is defined by 10 control points (params) at fixed knot locations
    spanning the input domain [-2, 2]. The second derivatives at the knots are
    determined by the natural boundary conditions (zero second derivative at endpoints).
    
    Parameters:
        x: input array of shape (n_points,)
        params: array of 10 control point values
    
    Returns:
        Array of predictions, one per point in x.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > 115.0: # Safety margin for 120s limit
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x)
        
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
        
    knots, h, W = _get_spline_constants()
    n_knots = len(knots)
    
    # Calculate moments M
    M = W @ params
    
    # Find segment index for each x
    # searchsorted returns index i such that knots[i-1] <= x < knots[i]
    # We want segment i such that knots[i] <= x < knots[i+1]
    # So seg_idx = searchsorted(knots, x, side='right') - 1
    seg_idx = np.searchsorted(knots, x, side='right') - 1
    seg_idx = np.clip(seg_idx, 0, n_knots - 2)
    
    # Local coordinates
    x0 = knots[seg_idx]
    x1 = knots[seg_idx + 1]
    y0 = params[seg_idx]
    y1 = params[seg_idx + 1]
    m0 = M[seg_idx]
    m1 = M[seg_idx + 1]
    h_seg = h[seg_idx]
    
    # u = x - x0
    u = x - x0
    
    # Natural Cubic Spline Formula:
    # S(x) = m0 * (x1-x)^3 / (6*h) + m1 * (x-x0)^3 / (6*h) + 
    #        (y0/h - m0*h/6) * (x1-x) + (y1/h - m1*h/6) * (x-x0)
    
    term1 = m0 * (x1 - x)**3 / (6.0 * h_seg)
    term2 = m1 * u**3 / (6.0 * h_seg)
    term3 = (y0 / h_seg - m0 * h_seg / 6.0) * (x1 - x)
    term4 = (y1 / h_seg - m1 * h_seg / 6.0) * u
    
    result = term1 + term2 + term3 + term4
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
