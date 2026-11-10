import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the base positions of 64 sheared tubes to minimize the union volume.
    Uses a differentiable approximation of the 1D union length via softplus gaps.
    """
    N = 8
    w = 1.0 / N
    
    # Initial guess: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    U0 = keich.copy()

    def softplus(x, beta=1.0):
        """
        Differentiable approximation of max(0, x).
        softplus(x) = (1/beta) * log(1 + exp(beta * x))
        """
        # Numerically stable softplus
        # If x > 0, log(1+exp(x)) ~ x
        # If x < 0, log(1+exp(x)) ~ exp(x)
        return np.where(x > 20, x, np.log1p(np.exp(x)))

    def union_length(U, c):
        """
        Computes the differentiable length of the union of N intervals.
        """
        starts = U + c * np.arange(N) / N
        ends = starts + w
        
        # Sort intervals by start point
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # The union length is the span minus the sum of the gaps.
        # Gap between interval k and k+1 is max(0, s_{k+1} - e_k).
        # We use softplus to make this differentiable.
        
        # Calculate gaps
        # s_sorted[1:] - e_sorted[:-1]
        gaps = s_sorted[1:] - e_sorted[:-1]
        soft_gaps = softplus(gaps)
        
        # Total span is max(end) - min(start)
        # Since sorted by start, min start is s_sorted[0]
        # Max end is NOT necessarily e_sorted[-1] if intervals are disjoint and later ones are shorter?
        # No, ends = starts + w. Since starts are sorted, e_sorted is NOT necessarily sorted if w is constant?
        # Yes, e_sorted[i] = s_sorted[i] + w. Since s_sorted is sorted, e_sorted is sorted.
        # So max end is e_sorted[-1].
        
        span = e_sorted[-1] - s_sorted[0]
        total_gap = np.sum(soft_gaps)
        
        return span - total_gap

    def objective(U):
        """
        Minimizes the integral of L(c)^2 over c in [0, 1].
        """
        # Use a fine grid for integration
        # We can use a slightly coarser grid if the function is smooth, but 100 is fine.
        c_vals = np.linspace(0, 1, 100)
        total = 0.0
        for c in c_vals:
            l = union_length(U, c)
            total += l * l
        return total / len(c_vals)

    # Optimize U
    bounds = [(-1.0, 1.0)] * N
    res = minimize(objective, U0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    U_opt = res.x

    # Construct the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = U_opt[i_flat]
    base_y = U_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Sanity check: ensure finite
    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions
