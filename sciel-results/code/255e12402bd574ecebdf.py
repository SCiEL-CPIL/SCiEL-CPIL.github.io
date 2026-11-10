# EVOLVE-BLOCK-START
import numpy as np
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses a separable 1D optimization approach: if base_x depends only on i and
    base_y only on j, the union volume factorizes as Lx * Ly. We optimize each
    1D problem independently to minimize the product.
    """
    t_start = time.time()
    N = 8
    width = 1.0 / N  # 1/8
    
    def one_d_union_length(positions_1d, N):
        """Compute union length of N sheared intervals in 1D.
        Interval i at height c: [pos[i] + c*i/N, pos[i] + c*i/N + 1/N]
        We need to find the measure of the set {(x,c) : x in some interval at height c}
        projected onto x-axis, then integrate over c... 
        Actually, the union in 2D (x,c) has area = union_length_in_1D_at_each_c integrated over c.
        But since we factorize, the union area in (x,c) for x-part times (y,c) for y-part...
        Let me reconsider.
        
        The 3D union volume with separable design:
        V = integral_0^1 [measure of union of x-intervals at height c] * [measure of union of y-intervals at height c] dc
        
        So we need to compute for each 1D problem the function f(c) = measure of union at height c,
        then V = integral_0^1 f_x(c) * f_y(c) dc.
        """
        # For a given 1D problem with N intervals of width w, sheared by i/N,
        # at height c, interval i is [pos[i] + c*i/N, pos[i] + c*i/N + w]
        # We want the union measure at each c.
        
        # Sample c values and compute union length
        c_samples = np.linspace(0, 1, 201)
        union_lens = np.zeros(len(c_samples))
        for idx, c in enumerate(c_samples):
            intervals = [(positions_1d[i] + c * i / N, positions_1d[i] + c * i / N + width) for i in range(N)]
            intervals.sort(key=lambda x: x[0])
            # Merge intervals
            if not intervals:
                union_lens[idx] = 0
                continue
            merged = [list(intervals[0])]
            for s, e in intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            union_lens[idx] = sum(e - s for s, e in merged)
        
        return c_samples, union_lens
    
    def compute_volume_3d(base_x, base_y):
        """Compute the 3D union volume using Monte Carlo with fixed seed."""
        N = 8
        width = 1.0 / N
        i_vals = np.repeat(np.arange(N), N)
        j_vals = np.tile(np.arange(N), N)
        
        x0 = np.min(base_x)
        x1 = np.max(base_x[i_vals // N * N + i_vals % N] + (i_vals + 1) / N)  # This is wrong
        # Let me redo: for tube k with i=k//8, j=k%8:
        # x range: [base_x[k], base_x[k] + i/8 + 1/8] at max c=1
        # So x1 = max over k of (base_x[k] + i_k/8 + 1/8) where i_k = k//8
        i_all = np.arange(64) // N
        j_all = np.arange(64) % N
        x1 = np.max(base_x + (i_all + 1) / N)
        y1 = np.max(base_y + (j_all + 1) / N)
        
        X0 = np.min(base_x)
        Y0 = np.min(base_y)
        
        box_area = (x1 - X0) * (y1 - Y0)
        if box_area <= 1e-9:
            return None
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, x1, 500000)
        b = rng.uniform(Y0, y1, 500000)
        c = rng.uniform(0, 1, 500000)
        
        # Check if point (a,b,c) is in any tube
        # Tube k: x in [base_x[k] + c*i_k/N, base_x[k] + c*i_k/N + width]
        #         y in [base_y[k] + c*j_k/N, base_y[k] + c*j_k/N + width]
        inside = np.zeros(500000, dtype=bool)
        for k in range(64):
            i_k = k // N
            j_k = k % N
            x_lo = base_x[k] + c * i_k / N
            x_hi = x_lo + width
            y_lo = base_y[k] + c * j_k / N
            y_hi = y_lo + width
            mask = (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
            inside |= mask
            if inside.all():
                break
        
        frac = np.mean(inside)
        V = frac * box_area
        return V
    
    # Optimize 1D positions
    # The 1D problem: N intervals of width w=1/8, sheared by i/N.
    # At height c, interval i is at [p_i + c*i/N, p_i + c*i/N + w]
    # We want to minimize integral_0^1 f(c)^2 dc (since V = integral f_x(c)*f_y(c) dc,
    # and if we use same positions for x and y, V = integral f(c)^2 dc)
    
    def objective_1d(params):
        """params: array of 8 base positions. Returns integral of f(c)^2 dc."""
        positions_1d = params
        c_samples = np.linspace(0, 1, 101)
        f = np.zeros(len(c_samples))
        for idx, c in enumerate(c_samples):
            intervals = [(positions_1d[i] + c * i / N, positions_1d[i] + c * i / N + width) for i in range(N)]
            intervals.sort(key=lambda x: x[0])
            merged = [list(intervals[0])]
            for s, e in intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            f[idx] = sum(e - s for s, e in merged)
        return np.trapz(f ** 2, c_samples)
    
    # Use a different initial design: place intervals to maximize overlap
    # Idea: at c=0, all intervals should overlap as much as possible
    # At c=1, interval i is at [p_i + i/8, p_i + i/8 + 1/8]
    # To minimize union, we want intervals at each c to cluster together
    
    # Try: p_i = -i/16 so that at c=0.5, interval i is at [-i/16 + i/16, -i/16 + i/16 + 1/8] = [0, 1/8]
    # All intervals at c=0.5 overlap at [0, 1/8]
    
    # Better: use optimization from a few starting points
    best_params = None
    best_obj = np.inf
    
    # Starting points
    starts = [
        np.linspace(0, -0.5, N),
        np.linspace(0, 0.5, N),
        -np.arange(N) / 16.0,
        np.arange(N) / 16.0,
        np.zeros(N),
    ]
    
    # Use Nelder-Mead or gradient-free optimization
    from scipy.optimize import minimize
    
    for start in starts:
        try:
            result = minimize(objective_1d, start, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            continue
        
        if time.time() - t_start > 580:
            break
    
    if best_params is None:
        best_params = -np.arange(N) / 16.0
    
    # Clip to reasonable range
    best_params = np.clip(best_params, -2.0, 2.0)
    
    # Build full 64x2 positions using separable design
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_params[i_flat]
    base_y = best_params[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Verify it's valid
    if not np.all(np.isfinite(positions)):
        # Fallback to simple design
        base_x = -np.arange(N)[i_flat] / 16.0
        base_y = -np.arange(N)[j_flat] / 16.0
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
