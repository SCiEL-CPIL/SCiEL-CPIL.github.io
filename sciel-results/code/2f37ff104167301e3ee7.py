# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    This implementation starts from the classical Keich construction and refines
    the positions using a convex optimization solver (SLSQP) to minimize the
    area of the union, computed via quadrature.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Generate Keich baseline
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    # Precompute a_i
    a = (np.arange(1, n + 1)) / n
    
    def area_objective(x):
        # Use a coarser grid for optimization speed
        # The evaluator uses 20000 points, but 500-1000 should be sufficient for gradient estimation
        m = 1000
        y_vals = (np.arange(m) + 0.5) / (2 * m)
        w = (1 - y_vals) / n
        
        # L_i(y) = x_i + a_i * y
        # Shape: (n, m)
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals[np.newaxis, :]
        
        # Sort L along axis 0 for each y
        L_sorted = np.sort(L, axis=0)
        
        # Differences between consecutive sorted starts
        diffs = np.diff(L_sorted, axis=0)
        
        # Union length at each y:
        # w + sum(min(diff_j, w))
        # = w + sum(diff_j) - sum(max(diff_j - w, 0))
        # sum(diff_j) = L_sorted[-1] - L_sorted[0]
        # max(diff_j - w, 0) is the part of the gap not covered by the interval width
        
        # Note: w varies with y, so we need to broadcast w
        # diffs shape: (n-1, m)
        # w shape: (m,)
        
        gaps_exceeding_w = np.maximum(diffs - w[np.newaxis, :], 0)
        
        # Union length for each y
        # w + (L_sorted[-1] - L_sorted[0]) - sum(gaps_exceeding_w, axis=0)
        union_lengths = w + (L_sorted[-1] - L_sorted[0]) - np.sum(gaps_exceeding_w, axis=0)
        
        # Area is the mean of union lengths (approximating integral over [0,1])
        # Since y_vals are uniform midpoints, the mean is a good approximation of the integral
        return np.mean(union_lengths)
    
    # Optimize
    # SLSQP is a good choice for small scale convex problems
    result = minimize(
        area_objective, 
        x0, 
        method='SLSQP', 
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    if result.success:
        return result.x
    else:
        # Fallback to baseline if optimization fails
        return x0

# EVOLVE-BLOCK-END
