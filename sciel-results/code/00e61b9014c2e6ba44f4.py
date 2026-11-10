# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses a refined 1D construction that minimizes the integral of the square
    of the 1D union length (equivalently, minimizes the integral of the product
    of x and y union lengths under the symmetry x = y). This accounts for
    covariance between x and y projections.
    """
    t_start = time.time()
    deadline = 630.0  # seconds
    
    N = 8
    w = 1.0 / N  # width of each tube cross-section
    
    # The 1D problem: we have N intervals of width w, where interval i at height c
    # occupies [x_i + c*i/N, x_i + c*i/N + w].
    # The union length at height c is L(c) = length of union of these N intervals.
    # We want to minimize integral_0^1 L(c)^2 dc (by symmetry, this equals
    # integral_0^1 Lx(c)*Ly(c) dc when x and y use the same construction).
    
    # Strategy: Use a parameterized family and optimize.
    # The Keich construction gives a good starting point.
    # Let's use a more general approach: parameterize by a set of "offsets"
    # and use local optimization.
    
    # Keich binary construction for N = 2^k
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # The 1D union length at height c for base positions x_0, ..., x_{N-1}:
    # Interval i is [x_i + c*i/N, x_i + c*i/N + w]
    # We need to compute the length of the union of these N intervals.
    
    def compute_union_length(x_bases, c_values, N=N, w=w):
        """Compute the union length of N intervals at given heights."""
        # x_bases: shape (N,)
        # c_values: shape (M,)
        # Returns: shape (M,) union lengths
        M = len(c_values)
        # For each c, we have N intervals. Compute union length.
        # To be efficient, we can sort intervals by left endpoint and sweep.
        
        # Left endpoints: x_i + c*i/N for i = 0,...,N-1
        # Shape: (M, N)
        lefts = x_bases[None, :] + c_values[:, None] * np.arange(N)[None, :] / N
        rights = lefts + w
        
        # For each row, sort by left endpoint and compute union
        # This is O(M * N log N)
        order = np.argsort(lefts, axis=1)
        sorted_lefts = np.take_along_axis(lefts, order, axis=1)
        sorted_rights = np.take_along_axis(rights, order, axis=1)
        
        # Sweep to compute union length
        # Start with first interval
        union_len = np.zeros(M, dtype=np.float64)
        current_right = sorted_rights[:, 0].copy()
        union_len[:] = current_right - sorted_lefts[:, 0]
        
        for i in range(1, N):
            new_left = sorted_lefts[:, i]
            new_right = sorted_rights[:, i]
            
            # If this interval extends beyond current_right, add the extension
            extension = np.maximum(0, new_right - np.maximum(current_right, new_left))
            union_len += extension
            current_right = np.maximum(current_right, new_right)
        
        return union_len
    
    def objective(x_bases, c_grid=None):
        """Compute integral of L(c)^2 dc using quadrature."""
        if c_grid is None:
            c_grid = np.linspace(0, 1, 200)
        L = compute_union_length(x_bases, c_grid)
        return np.trapz(L ** 2, c_grid)
    
    # Optimize the 1D positions
    # Start from Keich and use Nelder-Mead or gradient-based optimization
    
    x0 = keich.copy()
    
    # Try to improve using scipy optimize
    try:
        from scipy.optimize import minimize, differential_evolution
        
        # Use a coarse grid first for speed
        c_coarse = np.linspace(0, 1, 50)
        
        def obj_coarse(x_bases):
            L = compute_union_length(x_bases, c_coarse)
            return np.trapz(L ** 2, c_coarse)
        
        # Nelder-Mead from Keich start
        res_nm = minimize(obj_coarse, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        x_opt = res_nm.x
        val_opt = res_nm.fun
        val_keich = obj_coarse(x0)
        
        # Check time
        if time.time() - t_start > deadline - 120:
            # Not enough time, use Keich
            x_final = x0
        else:
            # Try differential evolution for global optimization
            # Bounds: positions should be in a reasonable range
            # The intervals shift by at most (N-1)/N = 7/8 over height 1
            # So positions should be in roughly [-1, 1]
            bounds = [(-1.5, 1.5)] * N
            
            def obj_de(x_bases):
                L = compute_union_length(x_bases, c_coarse)
                return np.trapz(L ** 2, c_coarse)
            
            # Use DE with moderate population
            res_de = differential_evolution(obj_de, bounds, maxiter=200,
                                           popsize=15, tol=1e-8, seed=42,
                                           polish=True)
            
            if res_de.fun < val_opt:
                x_opt = res_de.x
                val_opt = res_de.fun
            
            # Polish with Nelder-Mead on finer grid
            c_fine = np.linspace(0, 1, 200)
            
            def obj_fine(x_bases):
                L = compute_union_length(x_bases, c_fine)
                return np.trapz(L ** 2, c_fine)
            
            res_polish = minimize(obj_fine, x_opt, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            x_opt = res_polish.x
            val_opt = res_polish.fun
            
            x_final = x_opt
    except Exception:
        x_final = x0
    
    # Check deadline
    if time.time() - t_start > deadline - 60:
        x_final = x0  # Fall back to Keich if running out of time
    
    # Build the 2D positions: base_x depends only on i, base_y depends only on j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_final[i_flat]
    base_y = x_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        # Fall back to Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
