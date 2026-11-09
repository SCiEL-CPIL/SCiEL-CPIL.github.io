# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses Differential Evolution, a global stochastic optimization method,
    to minimize the union area of 64 triangles. This differs from the
    previous local search approach by exploring the global landscape.
    """
    n = 64
    
    # We fix x_0 = 0 to break translational symmetry.
    # We optimize x_1 ... x_{63}.
    # Bounds: The triangles should be close to each other. 
    # A reasonable bound is [-1, 1] for the relative positions.
    bounds = [(-1.0, 1.0)] * (n - 1)
    
    # Subset of heights for fast evaluation during optimization
    m_subset = np.arange(0, 20000, 20)  # 1000 heights
    y_subset = (m_subset + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    def area_objective(x_rel):
        """
        Compute union area using subset of heights.
        x_rel contains x_1 ... x_{63}. x_0 is fixed at 0.
        """
        # Reconstruct full x vector
        x = np.zeros(n)
        x[1:] = x_rel
        
        # L_i = x_i + a_i * y
        # Shape: (len(y_subset), n)
        L = x[None, :] + a[None, :] * y_subset[:, None]
        
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)
        
        # w at each y
        w = (1 - y_subset) / n  # shape (len(y_subset),)
        
        # Consecutive differences
        diffs = np.diff(L_sorted, axis=1)  # shape (len(y_subset), n-1)
        
        # Union length = w + sum(min(diffs, w))
        mins = np.minimum(diffs, w[:, None])
        union_lengths = w + np.sum(mins, axis=1)
        
        return np.mean(union_lengths)
    
    # Run Differential Evolution
    # popsize=10, maxiter=100
    # tol=1e-8
    # seed=42 for reproducibility
    result = differential_evolution(
        area_objective,
        bounds,
        method='differential',
        popsize=10,
        maxiter=100,
        tol=1e-8,
        seed=42,
        polish=True  # Use Nelder-Mead to polish the result at the end
    )
    
    x_rel_opt = result.x
    x_opt = np.zeros(n)
    x_opt[1:] = x_rel_opt
    
    # Verify with full 20000 heights
    m_full = np.arange(20000)
    y_full = (m_full + 0.5) / 20000.0
    L_full = x_opt[None, :] + a[None, :] * y_full[:, None]
    L_sorted = np.sort(L_full, axis=1)
    w_full = (1 - y_full) / n
    diffs_full = np.diff(L_sorted, axis=1)
    mins_full = np.minimum(diffs_full, w_full[:, None])
    union_lengths_full = w_full + np.sum(mins_full, axis=1)
    final_area = np.mean(union_lengths_full)
    
    # Sanity check
    if not np.isfinite(final_area) or final_area <= 0:
        # Fallback to a simple heuristic if optimization fails
        # e.g., all zeros
        return np.zeros(n)
    
    return x_opt.astype(np.float64)


# EVOLVE-BLOCK-END
