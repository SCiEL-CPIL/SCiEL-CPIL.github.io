import numpy as np
import time

# Pre-compute knots for the B-spline basis
# 10 parameters -> 10 cubic B-spline basis functions
# Need 14 knots for 10 basis functions of degree 3
# Domain [-2, 2]
_KNOTS = np.linspace(-2.0, 2.0, 14)
# Pre-compute denominators for Cox-de Boor recursion to save time
# B_{i,k} depends on t_{i+k} - t_i
# For k=1: t_{i+1}-t_i
# For k=2: t_{i+2}-t_i
# For k=3: t_{i+3}-t_i
# We can just compute them on the fly or precompute arrays.
# Given the small size, on-the-fly is fine, but let's precompute for speed.

# B1 (degree 1) uses knots[i+1]-knots[i] and knots[i+2]-knots[i+1]
# B2 (degree 2) uses knots[i+2]-knots[i] and knots[i+3]-knots[i+1]
# B3 (degree 3) uses knots[i+3]-knots[i] and knots[i+4]-knots[i+1]

# Since knots are uniform, these differences are constant!
# delta = 4.0 / 13.0
_DELTA = 4.0 / 13.0

# Denominators:
# k=1: _DELTA, _DELTA
# k=2: 2*_DELTA, 2*_DELTA
# k=3: 3*_DELTA, 3*_DELTA

def _bspline_basis_values(x_arr: np.ndarray) -> np.ndarray:
    """
    Compute the 10 cubic B-spline basis functions for all points in x_arr.
    Returns a matrix of shape (N, 10).
    """
    N = x_arr.shape[0]
    basis = np.zeros((N, 10), dtype=np.float64)
    
    # We iterate over points to avoid complex vectorization of the conditional logic
    # N=200 is small, so this is fast.
    for j in range(N):
        x_val = x_arr[j]
        
        # Handle boundary exactly
        if x_val < -2.0 or x_val > 2.0:
            # Should not happen based on task description, but safe to return zeros or clamp
            # Task says inputs drawn from [-2, 2]
            pass
        
        # Determine which interval x_val falls into
        # knots: t_0 ... t_13
        # Intervals: [t_0, t_1), ..., [t_12, t_13]
        # Find m such that t_m <= x_val < t_{m+1}
        # Using searchsorted
        # We want the rightmost index m such that knots[m] <= x_val
        # If x_val == 2.0, searchsorted side='left' gives 14 (out of bounds for interval index 13)
        # We need to handle the right edge.
        
        # searchsorted returns index i such that knots[i-1] <= x < knots[i] for side='left'?
        # No, side='left' returns i such that knots[i-1] < x <= knots[i] is false.
        # Let's use:
        # idx = np.searchsorted(_KNOTS, x_val, side='right') - 1
        # This gives the largest index i such that _KNOTS[i] <= x_val.
        
        idx = np.searchsorted(_KNOTS, x_val, side='right') - 1
        
        # Clamp idx to valid range for intervals 0..12
        if idx < 0: idx = 0
        if idx > 12: idx = 12
        
        # Now we know x_val is in [KNOTS[idx], KNOTS[idx+1]]
        # The non-zero basis functions of degree 3 are B_{idx-3, 3}, B_{idx-2, 3}, B_{idx-1, 3}, B_{idx, 3}
        # Indices must be valid: 0..9
        
        # We will compute B_i^3 for i in [max(0, idx-3), min(9, idx)]
        
        # To do this efficiently, we can compute the basis values for the relevant range.
        # However, since the support is small, we can just compute all 10?
        # No, 10 is small. Let's compute the 4 relevant ones and store them.
        
        # Actually, for simplicity and safety against index errors in the recursion,
        # let's compute the full basis row using the recursion on the fly for the 4 candidates.
        
        # Candidates: i_start = max(0, idx - 3), i_end = min(9, idx)
        
        # We need B_i^0, B_i^1, B_i^2 for these i.
        # B_i^0 is 1 if i == idx (since x is in interval idx)
        # So B_idx^0 = 1.0, others 0.
        
        # Let's use a local array for the current point's basis calculation
        # We need B^0 for indices idx-3 to idx (max 4 values)
        # But B^1 needs B^0 of i and i+1.
        # It's easier to just compute the 4 values directly using the formula if we know the local structure.
        
        # Formula for B_i^3(x) given x in [t_m, t_{m+1}]:
        # Only i = m-3, m-2, m-1, m can be non-zero.
        
        # Let's define a helper to get B_i^k for specific i, k
        # But recursion is easier.
        
        # Let's create small arrays for the recursion steps for the relevant window.
        # Window for B^0: indices idx-3 to idx+1 (to support B^1 up to idx)
        # Actually, let's just compute B^0 for all 13 intervals? No, that's wasteful.
        
        # Let's stick to the loop over the 10 basis functions, checking if they are non-zero.
        # For i in 0..9:
        #   if x_val >= _KNOTS[i] and x_val < _KNOTS[i+4]:
        #      compute B_i^3(x_val)
        
        for i in range(10):
            # Check support
            if x_val >= _KNOTS[i] and (x_val < _KNOTS[i+4] or (i == 9 and x_val == _KNOTS[13])):
                # Compute B_i^3(x_val)
                # We need B_i^0, B_{i+1}^0, B_{i+2}^0, B_{i+3}^0?
                # No, we need the recursion.
                
                # Let's compute the 4 base values B_j^0 for j in [i, i+3]
                # B_j^0 = 1 if _KNOTS[j] <= x_val < _KNOTS[j+1]
                
                b0_0 = 1.0 if (_KNOTS[i] <= x_val < _KNOTS[i+1]) else 0.0
                b0_1 = 1.0 if (_KNOTS[i+1] <= x_val < _KNOTS[i+2]) else 0.0
                b0_2 = 1.0 if (_KNOTS[i+2] <= x_val < _KNOTS[i+3]) else 0.0
                b0_3 = 1.0 if (_KNOTS[i+3] <= x_val <= _KNOTS[i+4]) else 0.0 # Handle right edge for last one
                
                # Compute B^1
                # B_i^1 = (x - t_i)/d1 * B_i^0 + (t_{i+2} - x)/d2 * B_{i+1}^0
                # d1 = _DELTA, d2 = _DELTA
                # B_{i+1}^1 = (x - t_{i+1})/d1 * B_{i+1}^0 + (t_{i+3} - x)/d2 * B_{i+2}^0
                # B_{i+2}^1 = (x - t_{i+2})/d1 * B_{i+2}^0 + (t_{i+4} - x)/d2 * B_{i+3}^0
                
                b1_0 = ((x_val - _KNOTS[i]) / _DELTA) * b0_0 + ((_KNOTS[i+2] - x_val) / _DELTA) * b0_1
                b1_1 = ((x_val - _KNOTS[i+1]) / _DELTA) * b0_1 + ((_KNOTS[i+3] - x_val) / _DELTA) * b0_2
                b1_2 = ((x_val - _KNOTS[i+2]) / _DELTA) * b0_2 + ((_KNOTS[i+4] - x_val) / _DELTA) * b0_3
                
                # Compute B^2
                # B_i^2 = (x - t_i)/(2d) * B_i^1 + (t_{i+3} - x)/(2d) * B_{i+1}^1
                # B_{i+1}^2 = (x - t_{i+1})/(2d) * B_{i+1}^1 + (t_{i+4} - x)/(2d) * B_{i+2}^1
                
                b2_0 = ((x_val - _KNOTS[i]) / (2.0 * _DELTA)) * b1_0 + ((_KNOTS[i+3] - x_val) / (2.0 * _DELTA)) * b1_1
                b2_1 = ((x_val - _KNOTS[i+1]) / (2.0 * _DELTA)) * b1_1 + ((_KNOTS[i+4] - x_val) / (2.0 * _DELTA)) * b1_2
                
                # Compute B^3
                # B_i^3 = (x - t_i)/(3d) * B_i^2 + (t_{i+4} - x)/(3d) * B_{i+1}^2
                
                b3_0 = ((x_val - _KNOTS[i]) / (3.0 * _DELTA)) * b2_0 + ((_KNOTS[i+4] - x_val) / (3.0 * _DELTA)) * b2_1
                
                basis[j, i] = b3_0
    
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety
    
    # Compute basis matrix (N, 10)
    basis = _bspline_basis_values(x)
    
    # Prediction = basis @ params
    pred = basis.dot(params)
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        # If something went wrong, return a simple fallback
        # But basis should be finite.
        return np.zeros_like(x)
        
    return pred
