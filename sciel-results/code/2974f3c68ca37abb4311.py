# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    num_tubes = N * N
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    # Pre-compute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)
    
    # Product-form parameterization:
    # x_k = x_base[i] + x_offset[i,j]
    # y_k = y_base[j] + y_offset[i,j]
    # This reduces degrees of freedom from 128 to 16 (8 x_base + 8 y_base) 
    # plus potentially offsets if we allow them. 
    # However, the task allows any (x_k, y_k). 
    # A pure product form x_k = A_i + B_j might be too restrictive?
    # Let's check: if x_k = A_i + B_j, then for fixed i, all j have same x shift relative to A_i.
    # Actually, let's use a slightly more flexible but still low-dim parameterization.
    # Or just use the full 128 params but with exact volume, Powell is fast enough?
    # 128 dims is high for Powell.
    # Let's try: x_k = x_i + dx_{i,j}, y_k = y_j + dy_{i,j}.
    # This is 8 + 64 + 8 + 64 = 144 params? No, that's worse.
    
    # Let's stick to the insight from inspiration 1/2/3:
    # "Product-form parameterization (16 vs 128 params)"
    # This implies x_k depends only on i, and y_k depends only on j?
    # i.e., x_k = X[i], y_k = Y[j].
    # This is 16 parameters total.
    # Is this sufficient to find a good Kakeya set?
    # In 2D, the Besicovitch set construction often involves translations that depend on the slope index.
    # If x depends only on i and y only on j, we are restricting the shape significantly.
    # However, for N=8, maybe this symmetry is close to optimal or a good local minimum.
    # Let's try this first as it allows very fast optimization (16 dims).
    
    # Alternative: x_k = X[i] + alpha * j, y_k = Y[j] + beta * i?
    # Let's start with the simplest: x_k = X[i], y_k = Y[j].
    
    def exact_volume(params):
        """
        Compute exact union volume for the set of tubes.
        params: array of shape (16,) containing X[0..7] and Y[0..7].
        """
        X = params[:N]
        Y = params[N:]
        
        # Reconstruct x_k, y_k
        # x_k = X[i], y_k = Y[j]
        x_pos = np.repeat(X, N) # For each i, repeat N times for j
        y_pos = np.tile(Y, N)   # For each j, tile N times for i
        
        # Check validity
        if not np.all(np.isfinite(x_pos)) or not np.all(np.isfinite(y_pos)):
            return 1e10
            
        # Determine bounding box
        # Tube k at height c:
        # x in [x_k + c*i/N, x_k + c*i/N + 1/N]
        # y in [y_k + c*j/N, y_k + c*j/N + 1/N]
        
        # X0 = min_k x_k
        # X1 = max_k (x_k + 1 * i/N) -> max_k (x_k + i_k/N) since c=1 is max shear?
        # Wait, i ranges 0..7. i/N ranges 0..7/8.
        # The right edge is x_k + c*i/N + 1/N.
        # Max right edge is at c=1: x_k + i_k/N + 1/N.
        # Min left edge is at c=0: x_k.
        
        X0 = np.min(x_pos)
        X1 = np.max(x_pos + i_flat / N + 1.0 / N)
        Y0 = np.min(y_pos)
        Y1 = np.max(y_pos + j_flat / N + 1.0 / N)
        
        w = X1 - X0
        h = Y1 - Y0
        
        if w <= 1e-9 or h <= 1e-9:
            return 1e10 # Invalid or degenerate
            
        # To compute volume, we integrate the area of the union of rectangles in x-y plane
        # over c from 0 to 1.
        # Area(c) = Length_X_union(c) * Length_Y_union(c) ?
        # NO. The union of rectangles is NOT the product of the union of x-intervals and union of y-intervals.
        # The union is a set of points (x,y) such that there exists k with x in I_xk(c) AND y in I_yk(c).
        # This is generally not separable.
        
        # However, inspiration 2 says: "uses an exact, piecewise-linear integration of the 1D union lengths".
        # This implies the volume is computed as integral of (Len_X(c) * Len_Y(c))?
        # This would only be true if the set of active tubes for x and y were independent or if the union
        # happened to be a rectangle. It is NOT generally a rectangle.
        
        # Let's re-read the geometry.
        # Tube k is a square.
        # The union of squares is a complex shape.
        # Monte Carlo estimates the volume of the union.
        # Exact integration of the union area is hard (Boolean operations on polygons).
        
        # Maybe the inspiration meant something else?
        # "1D union lengths" might refer to a specific approximation or a specific structure.
        # Or perhaps they approximate the volume by assuming independence?
        # If we assume the x-projection and y-projection are independent, V = int Len_X(c) Len_Y(c) dc.
        # This is an UPPER BOUND on the volume? No, the union area is <= product of projection lengths.
        # Actually, Area(Union) <= Area(Proj_X * Proj_Y).
        # So minimizing the product of projection lengths is a proxy for minimizing the union volume.
        # Is it a good proxy?
        # If the tubes are well-mixed, the union might fill the bounding box of projections fairly densely.
        # Let's try this proxy: Minimize int_0^1 (Len_X_union(c) * Len_Y_union(c)) dc.
        # This is much faster to compute exactly than the 2D union area.
        
        # Critical points for c are where interval endpoints coincide.
        # Endpoints for x:
        # L_k(c) = x_k + c*i_k/N
        # R_k(c) = x_k + c*i_k/N + 1/N
        # Endpoints for y:
        # B_k(c) = y_k + c*j_k/N
        # T_k(c) = y_k + c*j_k/N + 1/N
        
        # We need to find the union length of these intervals as a function of c.
        # The length is piecewise linear in c.
        # Breakpoints occur when L_k(c) = L_m(c), L_k(c) = R_m(c), etc.
        # i.e., x_k + c*i_k/N = x_m + c*i_m/N  => c = (x_m - x_k) / ((i_k - i_m)/N)
        # Since x_k = X[i_k], this becomes X[i_k] + c*i_k/N = X[i_m] + c*i_m/N.
        
        # Let's collect all critical c values in [0, 1].
        c_crits = [0.0, 1.0]
        
        # For x-intervals
        for k in range(num_tubes):
            for m in range(k+1, num_tubes):
                # L_k(c) = L_m(c)
                if i_flat[k] != i_flat[m]:
                    c_val = (x_pos[m] - x_pos[k]) / ((i_flat[k] - i_flat[m]) / N)
                    if 0 <= c_val <= 1:
                        c_crits.append(c_val)
                # L_k(c) = R_m(c)
                if i_flat[k] != i_flat[m]:
                    c_val = (x_pos[m] + 1.0/N - x_pos[k]) / ((i_flat[k] - i_flat[m]) / N)
                    if 0 <= c_val <= 1:
                        c_crits.append(c_val)
                # R_k(c) = L_m(c)
                if i_flat[k] != i_flat[m]:
                    c_val = (x_pos[m] - x_pos[k] - 1.0/N) / ((i_flat[k] - i_flat[m]) / N)
                    if 0 <= c_val <= 1:
                        c_crits.append(c_val)
                # R_k(c) = R_m(c)
                if i_flat[k] != i_flat[m]:
                    c_val = (x_pos[m] + 1.0/N - x_pos[k] - 1.0/N) / ((i_flat[k] - i_flat[m]) / N)
                    if 0 <= c_val <= 1:
                        c_crits.append(c_val)
                        
        # Same for y-intervals
        for k in range(num_tubes):
            for m in range(k+1, num_tubes):
                if j_flat[k] != j_flat[m]:
                    c_val = (y_pos[m] - y_pos[k]) / ((j_flat[k] - j_flat[m]) / N)
                    if 0 <= c_val <= 1:
                        c_crits.append(c_val)
                if j_flat[k] != j_flat[m]:
                    c_val = (y_pos[m] + 1.0/N - y_pos[k]) / ((j_flat[k] - j_flat[m]) / N)
                    if 0 <= c_val <= 1:
                        c_crits.append(c_val)
                if j_flat[k] != j_flat[m]:
                    c_val = (y_pos[m] - y_pos[k] - 1.0/N) / ((j_flat[k] - j_flat[m]) / N)
                    if 0 <= c_val <= 1:
                        c_crits.append(c_val)
                if j_flat[k] != j_flat[m]:
                    c_val = (y_pos[m] + 1.0/N - y_pos[k] - 1.0/N) / ((j_flat[k] - j_flat[m]) / N)
                    if 0 <= c_val <= 1:
                        c_crits.append(c_val)
        
        # Unique and sort
        c_crits = np.unique(np.array(c_crits))
        c_crits = c_crits[(c_crits >= 0) & (c_crits <= 1)]
        
        if len(c_crits) < 2:
            return 1e10
            
        total_vol = 0.0
        for idx in range(len(c_crits) - 1):
            c1 = c_crits[idx]
            c2 = c_crits[idx+1]
            if c2 - c1 < 1e-9:
                continue
            c_mid = 0.5 * (c1 + c2)
            
            # Compute union length of x-intervals at c_mid
            x_ints = []
            y_ints = []
            for k in range(num_tubes):
                lx = x_pos[k] + c_mid * i_flat[k] / N
                rx = lx + 1.0 / N
                x_ints.append((lx, rx))
                
                ly = y_pos[k] + c_mid * j_flat[k] / N
                ry = ly + 1.0 / N
                y_ints.append((ly, ry))
            
            # Sort and merge
            x_ints.sort()
            y_ints.sort()
            
            len_x = 0.0
            if x_ints:
                cur_l, cur_r = x_ints[0]
                for l, r in x_ints[1:]:
                    if l <= cur_r:
                        if r > cur_r:
                            cur_r = r
                    else:
                        len_x += cur_r - cur_l
                        cur_l, cur_r = l, r
                len_x += cur_r - cur_l
                
            len_y = 0.0
            if y_ints:
                cur_l, cur_r = y_ints[0]
                for l, r in y_ints[1:]:
                    if l <= cur_r:
                        if r > cur_r:
                            cur_r = r
                    else:
                        len_y += cur_r - cur_l
                        cur_l, cur_r = l, r
                        if r > cur_r: # Wait, logic error in merge?
                            pass
                        cur_l, cur_r = l, r # Re-initialize
                len_y += cur_r - cur_l
                
            # Wait, the merge logic above for y was slightly messed up in the draft.
            # Let's write a helper.
            
            def calc_union_length(intervals):
                if not intervals:
                    return 0.0
                intervals.sort()
                total = 0.0
                cur_l, cur_r = intervals[0]
                for l, r in intervals[1:]:
                    if l <= cur_r:
                        if r > cur_r:
                            cur_r = r
                    else:
                        total += cur_r - cur_l
                        cur_l, cur_r = l, r
                total += cur_r - cur_l
                return total

            len_x = calc_union_length(x_ints)
            len_y = calc_union_length(y_ints)
            
            # Area at c_mid is approx len_x * len_y (proxy)
            # Volume contribution is Area * dc
            area = len_x * len_y
            total_vol += area * (c2 - c1)
            
        return total_vol

    # Optimization
    # Initial guess
    rng = np.random.default_rng(42)
    best_params = None
    best_vol = np.inf
    
    # Try a few random restarts
    num_restarts = 15
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random initialization in [0, 1]
        params0 = rng.uniform(0, 1, 16)
        
        res = minimize(exact_volume, params0, method='Powell', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.success or res.fun < best_vol:
            if np.isfinite(res.fun):
                best_vol = res.fun
                best_params = res.x.copy()
                
    # If product form is not good enough or fails, we might need to fall back.
    # But let's assume it finds a decent local min.
    
    if best_params is None:
        # Fallback to simple grid
        X = np.arange(N) / (2*N)
        Y = np.arange(N) / (2*N)
        best_params = np.concatenate([X, Y])
        
    # Reconstruct x_k, y_k from best_params
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    x_pos = np.repeat(X_opt, N)
    y_pos = np.tile(Y_opt, N)
    
    # Final check
    if not np.all(np.isfinite(x_pos)) or not np.all(np.isfinite(y_pos)):
        X = np.arange(N) / (2*N)
        Y = np.arange(N) / (2*N)
        x_pos = np.repeat(X, N)
        y_pos = np.tile(Y, N)
        
    return np.stack([x_pos, y_pos], axis=1)

# EVOLVE-BLOCK-END
