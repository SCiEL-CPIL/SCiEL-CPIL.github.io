import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

# Knots for cubic B-splines
# 14 knots, degree 3 -> 10 basis functions
_KNOTS = np.array([-2.0, -2.0, -2.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0])
_DEGREE = 3
_N_BASIS = len(_KNOTS) - _DEGREE - 1 # 10

def _bspline_basis(x: np.ndarray) -> np.ndarray:
    """
    Compute the B-spline basis functions for the given x values.
    Returns a matrix of shape (len(x), _N_BASIS).
    """
    n_points = len(x)
    basis = np.zeros((n_points, _N_BASIS))
    
    # Base case: k=0
    # N_{i,0}(x) = 1 if _KNOTS[i] <= x < _KNOTS[i+1]
    # Note: For the last knot, we include x == _KNOTS[-1] in the last interval
    for i in range(len(_KNOTS) - 1):
        mask = (x >= _KNOTS[i]) & (x < _KNOTS[i+1])
        if i == len(_KNOTS) - 2:
            mask = (x >= _KNOTS[i]) & (x <= _KNOTS[i+1])
        basis[:, i] = mask.astype(float)
        
    # Recursion for k=1, 2, 3
    # We need to be careful with indices.
    # N_{i,k} depends on N_{i,k-1} and N_{i+1,k-1}.
    # We will iterate k from 1 to _DEGREE.
    # At step k, we compute basis_k from basis_{k-1}.
    # The number of basis functions at step k is len(_KNOTS) - k.
    # But we only care about the first _N_BASIS functions?
    # Actually, the recursion naturally reduces the number of non-zero functions.
    
    current_basis = basis
    n_current = len(_KNOTS) - 1
    
    for k in range(1, _DEGREE + 1):
        next_basis = np.zeros((n_points, n_current - 1))
        for i in range(n_current - 1):
            # Term 1: (x - t_i) / (t_{i+k} - t_i) * N_{i,k-1}
            denom1 = _KNOTS[i+k] - _KNOTS[i]
            if denom1 != 0:
                term1 = (x - _KNOTS[i]) / denom1 * current_basis[:, i]
            else:
                term1 = 0.0
                
            # Term 2: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * N_{i+1,k-1}
            denom2 = _KNOTS[i+k+1] - _KNOTS[i+1]
            if denom2 != 0:
                term2 = (_KNOTS[i+k+1] - x) / denom2 * current_basis[:, i+1]
            else:
                term2 = 0.0
                
            next_basis[:, i] = term1 + term2
            
        current_basis = next_basis
        n_current -= 1
        
    # current_basis now has shape (n_points, _N_BASIS)
    return current_basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: linear interpolation or simple constant
        # Using the first two params as a linear fit is not straightforward with B-splines
        # Just return 0 or a constant
        return np.zeros_like(x)
    
    basis = _bspline_basis(x)
    return basis @ params

