"""Symbolic regression with 1 input using cubic B-spline basis."""
# EVOLVE-BLOCK-START
import numpy as np


def _eval_cubic_bspline_basis(x: np.ndarray, n_ctrl: int = 10) -> np.ndarray:
    """
    Evaluate n_ctrl cubic B-spline basis functions at points x.
    Uses a clamped uniform knot vector over [-3, 3].
    Returns array of shape (len(x), n_ctrl).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = n_ctrl
    p = 3  # degree
    
    # Knot vector for clamped cubic B-spline with n control points
    # Need n + p + 1 = n + 4 knots
    # Clamped: first p+1=4 knots at -3, last p+1=4 at 3
    # Interior: n - 4 knots, uniformly spaced
    if n <= 4:
        # Degenerate case, use simple basis
        basis = np.ones((len(x), n))
        for i in range(1, n):
            basis[:, i] = x ** i
        return basis
    
    n_int = n - 4  # number of interior knots
    if n_int > 0:
        interior = np.linspace(-3, 3, n_int + 2)[1:-1]
    else:
        interior = np.array([])
    
    knots = np.concatenate([
        np.full(p + 1, -3.0),
        interior,
        np.full(p + 1, 3.0)
    ])
    
    # Cox-de Boor recursion for B-spline basis
    # N_{i,0}(x) = 1 if knots[i] <= x < knots[i+1]
    # N_{i,p}(x) = (x - knots[i])/(knots[i+p] - knots[i]) * N_{i,p-1}(x) +
    #              (knots[i+p+1] - x)/(knots[i+p+1] - knots[i+1]) * N_{i+1,p-1}(x)
    
    m = len(x)
    basis = np.zeros((m, n))
    
    # Degree 0
    for i in range(n):
        if i < len(knots) - 1:
            mask = (x >= knots[i]) & (x < knots[i + 1])
            basis[:, i] = mask.astype(np.float64)
    
    # Higher degrees
    for d in range(1, p + 1):
        new_basis = np.zeros((m, n))
        for i in range(n - d):
            # First term
            denom1 = knots[i + d] - knots[i]
            if denom1 > 1e-12:
                term1 = (x - knots[i]) / denom1 * basis[:, i]
            else:
                term1 = np.zeros(m)
            
            # Second term
            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 > 1e-12 and (i + 1) < n:
                term2 = (knots[i + d + 1] - x) / denom2 * basis[:, i + 1]
            else:
                term2 = np.zeros(m)
            
            new_basis[:, i] = term1 + term2
        basis = new_basis
    
    # Handle the last basis function (rightmost)
    # For clamped B-splines, the last basis function should be 1 at x = 3
    if n > 0:
        basis[-1, :] = 0  # This isn't quite right, let me reconsider
    
    # Actually, the Cox-de Boor above should handle it. Let me verify the indexing.
    # The issue is that for clamped B-splines, we need to be careful at the boundaries.
    
    # Let me just use a direct approach: evaluate each basis function
    # using the recursive definition, handling edge cases.
    
    # Reset and redo more carefully
    basis = np.zeros((m, n))
    
    # Degree 0
    for i in range(n):
        if knots[i] < knots[i + 1]:
            basis[:, i] = ((x >= knots[i]) & (x < knots[i + 1])).astype(np.float64)
        elif i == n - 1:
            # Last interval, include right endpoint
            basis[:, i] = ((x >= knots[i]) & (x <= knots[i + 1])).astype(np.float64)
    
    for d in range(1, p + 1):
        new_basis = np.zeros((m, n))
        for i in range(n - d):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            
            if denom1 > 1e-12:
                term1 = (x - knots[i]) / denom1 * basis[:, i]
            else:
                term1 = np.zeros(m)
            
            if denom2 > 1e-12:
                term2 = (knots[i + d + 1] - x) / denom2 * basis[:, i + 1]
            else:
                term2 = np.zeros(m)
            
            new_basis[:, i] = term1 + term2
        basis = new_basis
    
    # Ensure partition of unity (should be automatic for clamped B-splines)
    # But due to numerical issues, let's normalize
    row_sums = basis.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    basis = basis / row_sums
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic B-spline with 10 control points.
    params (length 10): control point values defining the curve.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if len(params) != 10:
        params = np.resize(params, 10)
    
    basis = _eval_cubic_bspline_basis(x, n_ctrl=10)
    result = basis @ params
    
    result = np.where(np.isfinite(result), result, 0.0)
    return result
# EVOLVE-BLOCK-END
