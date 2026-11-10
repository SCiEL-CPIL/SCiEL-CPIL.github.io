# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _union_length(intervals):
    """
    Calculate the total length of the union of a list of intervals.
    intervals: list of (start, end) tuples.
    """
    if not intervals:
        return 0.0
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    total_length = 0.0
    current_start, current_end = sorted_intervals[0]
    
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            # Overlapping or touching
            current_end = max(current_end, end)
        else:
            # Disjoint
            total_length += current_end - current_start
            current_start, current_end = start, end
            
    total_length += current_end - current_start
    return total_length

def _cost_3d_proxy(X):
    """
    Calculate the integral of the square of the 1D union length.
    This corresponds to the volume of the 3D union if X=Y due to separability.
    V = integral_0^1 L(c)^2 dc
    """
    N = 8
    indices = np.arange(N)
    # Use a fine grid for integration
    c_samples = np.linspace(0, 1, 200)
    total_cost = 0.0
    for c in c_samples:
        starts = X + c * indices / N
        ends = starts + 1.0 / N
        intervals = list(zip(starts, ends))
        L = _union_length(intervals)
        total_cost += L * L
    # Trapezoidal integration
    return total_cost / len(c_samples)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Initial guess: all zeros
    X0 = np.zeros(N)
    
    # Optimize X to minimize the 3D volume proxy
    try:
        res = minimize(_cost_3d_proxy, X0, method='Powell', options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-11})
        if res.success:
            X_opt = res.x
        else:
            # Fallback
            res2 = minimize(_cost_3d_proxy, X0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-11})
            X_opt = res2.x
    except Exception:
        X_opt = np.linspace(0, 1, N)

    # Symmetric solution
    Y_opt = X_opt

    # Construct the 64 positions
    i_indices = np.repeat(np.arange(N), N)
    j_indices = np.tile(np.arange(N), N)
    
    base_x = X_opt[i_indices]
    base_y = Y_opt[j_indices]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions
# EVOLVE-BLOCK-END
