# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses the Keich construction as initialization, then refines with scipy optimization.
    """
    deadline = time.time() + 600.0
    n = 64
    
    # Keich construction as initial guess
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Quadrature heights
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n  # shape (n,)
    w = (1 - y) / n  # shape (20000,)
    
    def compute_area(x):
        """Compute the area of the union of triangles given base positions x."""
        # L_i = x_i + a_i * y, shape (n, 20000)
        L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
        # Sort L along axis 0
        L_sorted = np.sort(L, axis=0)  # shape (n, 20000)
        # Compute gaps between consecutive L values
        gaps = np.diff(L_sorted, axis=0)  # shape (n-1, 20000)
        # Union length = w + sum of min(gap, w) for consecutive pairs
        min_gaps = np.minimum(gaps, w[np.newaxis, :])  # shape (n-1, 20000)
        union_length = w[np.newaxis, :] + min_gaps.sum(axis=0)  # shape (20000,)
        area = np.mean(union_length)
        return area
    
    # Compute initial area
    initial_area = compute_area(x0)
    
    # Try to improve with scipy optimization
    best_x = x0.copy()
    best_area = initial_area
    
    try:
        # Use L-BFGS-B for constrained optimization
        result = minimize(
            compute_area,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 10000}
        )
        
        if result.success and result.fun < best_area:
            best_x = result.x
            best_area = result.fun
        
        # Check deadline
        if time.time() > deadline - 60:
            return best_x.astype(np.float64)
        
        # Try further refinement with more iterations
        result2 = minimize(
            compute_area,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 20000}
        )
        
        if result2.success and result2.fun < best_area:
            best_x = result2.x
            best_area = result2.fun
            
    except Exception:
        pass
    
    # Final validation
    final_area = compute_area(best_x)
    if not np.isfinite(final_area) or final_area <= 0:
        return x0.astype(np.float64)
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
