# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Minimizes the volume of the union of 64 sheared tubes in 3D.
    
    Strategy:
    1. Exploit separability to find an excellent initial guess by solving the 1D
       analog problem optimally.
    2. Use a deterministic, exact volume estimator based on integrating the 2D
       union area of the cross-sections.
    3. Perform a full 128-variable optimization using scipy's Powell method,
       starting from the separable solution.
    """
    N = 8
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0

    # Precompute tube indices
    k_idx = np.arange(N * N)
    i_idx = (k_idx // N).astype(np.float64)
    j_idx = (k_idx % N).astype(np.float64)
    width = 1.0 / N

    def union_length_1d(z, c):
        """
        Calculate the length of the union of intervals for a given height c.
        z: array of shape (N,)
        c: scalar
        """
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + width
        
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        total_len = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        
        for k in range(1, N):
            if s_sorted[k] <= cur_end:
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
            else:
                total_len += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
        total_len += cur_end - cur_start
        return total_len

    def cost_1d(z):
        """
        Minimize the integral of the 1D union length over c in [0, 1].
        z: array of shape (N-1,), representing z_1 to z_{N-1}. z_0 is fixed to 0.
        """
        full_z = np.zeros(N)
        full_z[1:] = z
        
        # Use high-resolution quadrature for accuracy
        c_samples = np.linspace(0, 1, 501)
        lengths = np.array([union_length_1d(full_z, c) for c in c_samples])
        integral = np.trapz(lengths, c_samples)
        return integral

    # 1. Solve the 1D separable problem
    # We optimize z_1..z_7. z_0 is fixed at 0.
    # Initial guess: Keich construction or random
    rng = np.random.default_rng(42)
    
    # Keich construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Center keich
    keich -= keich.mean()
    
    best_z_1d = None
    best_cost_1d = np.inf
    
    # Run a few optimizations to ensure we find a good local minimum
    # The 1D problem is likely convex or has few local minima
    for run in range(3):
        if time.time() > deadline - safety_margin:
            break
            
        if run == 0:
            start = keich[1:] - keich[0]
        else:
            start = rng.uniform(-0.5, 0.5, N - 1)
            
        try:
            res = scipy.optimize.minimize(cost_1d, start, method='Nelder-Mead', 
                                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_cost_1d:
                best_cost_1d = res.fun
                best_z_1d = res.x
        except Exception:
            pass
            
    if best_z_1d is None:
        best_z_1d = keich[1:] - keich[0]
        
    # Construct full 1D positions
    z_full = np.zeros(N)
    z_full[1:] = best_z_1d
    
    # Build initial 64x2 positions using separable structure
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = z_full[i_flat]
    base_y = z_full[j_flat]
    pos_init = np.stack([base_x, base_y], axis=1)
    
    # Center the whole configuration
    pos_init[:, 0] -= pos_init[:, 0].mean()
    pos_init[:, 1] -= pos_init[:, 1].mean()

    def union_area_2d(squares):
        """
        Calculate the area of the union of axis-aligned squares.
        squares: array of shape (M, 4) -> [x_min, x_max, y_min, y_max]
        """
        if len(squares) == 0:
            return 0.0
            
        x_coords = np.unique(np.concatenate([squares[:, 0], squares[:, 1]]))
        
        if len(x_coords) < 2:
            return 0.0
            
        total_area = 0.0
        
        # Iterate through x-slabs
        for i in range(len(x_coords) - 1):
            x1 = x_coords[i]
            x2 = x_coords[i+1]
            width_x = x2 - x1
            
            # Find squares that cover this x-slab
            # A square [xs, xe, ys, ye] covers [x1, x2] if xs <= x1 and xe >= x2
            # Since x1, x2 are from the set of starts/ends, we can just check
            # if the square's x-range overlaps with (x1, x2).
            # Actually, if we use unique coords, the slab [x1, x2] is fully covered
            # by a square if square_x_min <= x1 and square_x_max >= x2.
            
            active_indices = np.where((squares[:, 0] <= x1) & (squares[:, 1] >= x2))[0]
            
            if len(active_indices) == 0:
                continue
                
            # Get y-intervals for active squares
            y_intervals = squares[active_indices, 2:4]
            
            # Merge y-intervals to find total length
            y_intervals = y_intervals[np.argsort(y_intervals[:, 0])]
            
            total_y_len = 0.0
            cur_y_start = y_intervals[0, 0]
            cur_y_end = y_intervals[0, 1]
            
            for k in range(1, len(y_intervals)):
                if y_intervals[k, 0] <= cur_y_end:
                    if y_intervals[k, 1] > cur_y_end:
                        cur_y_end = y_intervals[k, 1]
                else:
                    total_y_len += cur_y_end - cur_y_start
                    cur_y_start = y_intervals[k, 0]
                    cur_y_end = y_intervals[k, 1]
            total_y_len += cur_y_end - cur_y_start
            
            total_area += width_x * total_y_len
            
        return total_area

    def compute_volume_exact(pos):
        """
        Compute the volume by integrating the 2D union area over c in [0, 1].
        pos: shape (64, 2)
        """
        x = pos[:, 0]
        y = pos[:, 1]
        
        # Use high-resolution quadrature for c
        # The area function is piecewise polynomial, so trapz is good enough
        # with sufficient points.
        c_samples = np.linspace(0, 1, 101)
        
        areas = np.zeros(len(c_samples))
        for idx, c in enumerate(c_samples):
            # Compute square coordinates at height c
            x_starts = x + c * i_idx / N
            x_ends = x_starts + width
            y_starts = y + c * j_idx / N
            y_ends = y_starts + width
            
            # squares: [x_min, x_max, y_min, y_max]
            squares = np.stack([x_starts, x_ends, y_starts, y_ends], axis=1)
            
            areas[idx] = union_area_2d(squares)
            
        volume = np.trapz(areas, c_samples)
        return volume

    # 2. Full 3D Optimization
    # Use Powell method as it is derivative-free and handles high dimensions well
    # for smooth functions.
    
    best_pos = pos_init
    best_V = compute_volume_exact(best_pos)
    print(f"Initial V (Exact): {best_V:.6f}")
    
    # Define the objective function for the optimizer
    # It must take a flat array of 128 floats
    def objective(flat_pos):
        if time.time() > deadline - safety_margin:
            return 1e9 # Timeout guard
        pos = flat_pos.reshape(N*N, 2)
        # Ensure no NaNs/Infs
        if not np.all(np.isfinite(pos)):
            return 1e9
        V = compute_volume_exact(pos)
        if not np.isfinite(V) or V <= 0:
            return 1e9
        return V

    # Run optimization
    # Powell is robust but can be slow. We set maxiter to a reasonable number.
    # We can also try Nelder-Mead if Powell is too slow, but Powell is often better for >20 dims.
    
    try:
        res = scipy.optimize.minimize(
            objective, 
            pos_init.flatten(), 
            method='Powell',
            options={
                'maxiter': 500, 
                'maxfev': 20000,
                'xtol': 1e-8,
                'ftol': 1e-8,
                'disp': False
            }
        )
        
        if res.fun < best_V:
            best_V = res.fun
            best_pos = res.x.reshape(N*N, 2)
            print(f"Optimized V (Exact): {best_V:.6f}")
            
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Final check with the exact MC estimator as specified in the task
    # The task uses a specific MC estimator for scoring.
    # Our optimization used an exact (quadrature) estimator.
    # We should verify the score using the task's MC estimator.
    
    def final_volume_mc(pos):
        x = pos[:, 0]
        y = pos[:, 1]

        x_max = x + i_idx / N + width
        x_min = x
        y_max = y + j_idx / N + width
        y_min = y

        X0 = x_min.min()
        X1 = x_max.max()
        Y0 = y_min.min()
        Y1 = y_max.max()

        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return np.inf

        rng = np.random.default_rng(12345)
        n = 500000
        a = rng.uniform(X0, X1, n)
        b = rng.uniform(Y0, Y1, n)
        c = rng.uniform(0, 1, n)

        x_starts = x[:, None] + c[None, :] * i_idx[:, None] / N
        x_ends = x_starts + width
        y_starts = y[:, None] + c[None, :] * j_idx[:, None] / N
        y_ends = y_starts + width

        inside_x = (a[None, :] >= x_starts) & (a[None, :] <= x_ends)
        inside_y = (b[None, :] >= y_starts) & (b[None, :] <= y_ends)
        inside = inside_x & inside_y
        any_inside = inside.any(axis=0)
        fraction = any_inside.mean()
        V = fraction * box_area
        if V <= 0:
            return np.inf
        return V

    final_V = final_volume_mc(best_pos)
    if np.isfinite(final_V) and final_V > 0:
        print(f"Final V (MC): {final_V:.6f}")
    else:
        print("Final V invalid, using initial")
        best_pos = pos_init
        final_V = final_volume_mc(best_pos)

    return best_pos

# EVOLVE-BLOCK-END
