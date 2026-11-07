"""Symbolic regression: fit a cubic B-spline basis via BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 135.0  # safety margin before 150s kill


def _remaining_time() -> float:
    """Return remaining seconds before safety deadline."""
    return _TIME_BUDGET - (time.time() - _START_TIME)


def _cubic_bspline_basis(x):
    """
    Compute a cubic B-spline basis matrix with uniform knots over [-2, 2].
    
    Uses 10 basis functions (cubic B-splines with degree 3).
    The knots are placed at -2, -1.2, -0.4, 0.4, 1.2, 2.0 (5 interior knots + 2 endpoints).
    For cubic B-splines (degree 3), the number of basis functions is n_knots - 1 - degree
    if we consider the support, but more simply: with K uniform intervals, we have K+degree basis functions.
    
    Let's use a simpler construction:
    Divide [-2, 2] into 7 intervals of width 4/7.
    Knots: -2, -2+4/7, ..., 2. That's 8 knots.
    For cubic B-splines, we need 4 extra knots on each side for the boundary, or we can just
    evaluate the basis functions directly using the Cox-de Boor recursion or a simpler approach.
    
    Alternative simpler approach: Use 10 overlapping cubic "bumps" (not standard B-splines)
    that are C^2 continuous and form a partition of unity or at least a stable basis.
    
    Actually, let's stick to a well-defined cubic B-spline basis.
    We'll use 10 basis functions. Let's define knots such that we get exactly 10 basis functions.
    Number of basis functions = number of knots - degree - 1.
    So if we want 10 basis functions with degree 3, we need 10 + 3 + 1 = 14 knots.
    Let's place 14 uniform knots over [-2, 2].
    Knots: -2, -2 + 4/13, -2 + 8/13, ..., 2.
    This gives 13 intervals, 14 knots.
    Basis functions: N_{i,3} for i = 0, ..., 10. (11 functions? No, 14-3-1 = 10).
    Yes, 10 basis functions.
    
    We will compute these using the Cox-de Boor recursion.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    degree = 3
    n_basis = 10
    
    # Define 14 uniform knots over [-2, 2]
    knots = np.linspace(-2.0, 2.0, 14)
    
    # Initialize basis matrix
    basis = np.zeros((n, n_basis), dtype=np.float64)
    
    # Degree 0 basis functions
    # N_{i,0}(x) = 1 if knots[i] <= x < knots[i+1] else 0
    # We need basis functions for i = 0 to 10 (11 functions) to start recursion up to degree 3?
    # Actually, for degree 3, we need N_{i,3} for i=0..10.
    # Recursion: N_{i,k} = ((x - t_i)/(t_{i+k} - t_i)) * N_{i,k-1} + ((t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1})) * N_{i+1,k-1}
    # For k=0, N_{i,0} is defined for i=0..13 (14 knots -> 13 intervals? No, 14 knots define 13 intervals).
    # Wait, standard B-spline definition:
    # If we have K knots t_0...t_{K-1}, and degree p, the number of basis functions is K - p - 1.
    # Here K=14, p=3, so 10 basis functions. Correct.
    # The basis functions are N_{0,3}, ..., N_{9,3}.
    # To compute N_{i,p}, we need N_{j,k} for j up to i+p and k up to p.
    # So we need to compute up to N_{9,3}.
    # This requires N_{j,0} for j=0..12 (since 9+3=12).
    # Let's compute all necessary intermediate basis functions.
    
    # Let's compute N_{i,k} for k=0..3 and i=0..(12-k)
    # We'll store them in a list of arrays or a 3D array.
    # For simplicity, let's compute column by column or level by level.
    
    # Level 0 (k=0)
    # N_{i,0} for i=0..12
    N = np.zeros((n, 13), dtype=np.float64) # Max i for k=0 is 12 (13 functions)
    for i in range(13):
        mask = (x >= knots[i]) & (x < knots[i+1])
        N[:, i] = mask.astype(np.float64)
    # Handle the right endpoint x == 2.0
    mask_end = (x == knots[13])
    N[:, 12] += mask_end.astype(np.float64) # Assign last interval to last basis function
    
    # Level 1 (k=1)
    # N_{i,1} for i=0..11
    N1 = np.zeros((n, 12), dtype=np.float64)
    for i in range(12):
        denom1 = knots[i+1] - knots[i]
        denom2 = knots[i+2] - knots[i+1]
        if denom1 > 1e-12:
            term1 = (x - knots[i]) / denom1 * N[:, i]
        else:
            term1 = np.zeros_like(x)
        if denom2 > 1e-12:
            term2 = (knots[i+2] - x) / denom2 * N[:, i+1]
        else:
            term2 = np.zeros_like(x)
        N1[:, i] = term1 + term2
        
    # Level 2 (k=2)
    # N_{i,2} for i=0..10
    N2 = np.zeros((n, 11), dtype=np.float64)
    for i in range(11):
        denom1 = knots[i+2] - knots[i]
        denom2 = knots[i+3] - knots[i+1]
        if denom1 > 1e-12:
            term1 = (x - knots[i]) / denom1 * N1[:, i]
        else:
            term1 = np.zeros_like(x)
        if denom2 > 1e-12:
            term2 = (knots[i+3] - x) / denom2 * N1[:, i+1]
        else:
            term2 = np.zeros_like(x)
        N2[:, i] = term1 + term2
        
    # Level 3 (k=3)
    # N_{i,3} for i=0..9
    N3 = np.zeros((n, 10), dtype=np.float64)
    for i in range(10):
        denom1 = knots[i+3] - knots[i]
        denom2 = knots[i+4] - knots[i+1]
        if denom1 > 1e-12:
            term1 = (x - knots[i]) / denom1 * N2[:, i]
        else:
            term1 = np.zeros_like(x)
        if denom2 > 1e-12:
            term2 = (knots[i+4] - x) / denom2 * N2[:, i+1]
        else:
            term2 = np.zeros_like(x)
        N3[:, i] = term1 + term2
        
    return N3


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the model: sum of params[i] * B_i(x) where B_i are cubic B-splines.
    This is linear in params.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if len(params) < 10:
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
        
    basis = _cubic_bspline_basis(x)
    
    # Compute predictions
    return basis @ params
# EVOLVE-BLOCK-END
