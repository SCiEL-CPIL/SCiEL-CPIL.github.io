# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_3d_volume(params):
    """
    Computes the exact volume of the union of 64 sheared boxes.
    params: array of shape (16,). First 8 are x_base[0..7], next 8 are y_base[0..7].
    Tube k (k=0..63) corresponds to i=k//8, j=k%8.
    Box at height c:
      x in [x_i + c*i/8, x_i + c*i/8 + 1/8]
      y in [y_j + c*j/8, y_j + c*j/8 + 1/8]
      z in [0, 1]
    
    We integrate the area of the union of rectangles in the xy-plane over c from 0 to 1.
    The area A(c) is a piecewise linear function of c.
    Breakpoints occur when the relative order of any two rectangle edges changes.
    Edge x_start_i = x_i + c*i/8.
    Edge x_end_i = x_i + c*i/8 + 1/8.
    Same for y.
    
    We find all critical c values where any x-edge or y-edge crosses another.
    Then we compute A(c) at these critical points and integrate linearly.
    """
    N = 8
    x_base = params[:N]
    y_base = params[N:]
    
    # Collect critical c values
    # x-edges: for each i, start_i(c) = x_i + c*i/N, end_i(c) = x_i + c*i/N + 1/N
    # y-edges: for each j, start_j(c) = y_j + c*j/N, end_j(c) = y_j + c*j/N + 1/N
    
    critical_cs = set([0.0, 1.0])
    
    # x-crossings
    for i in range(N):
        for j in range(N):
            # start_i(c) = start_j(c) => x_i + c*i/N = x_j + c*j/N => c = N*(x_j-x_i)/(i-j)
            if i != j:
                c = N * (x_base[j] - x_base[i]) / (i - j)
                if 0.0 <= c <= 1.0:
                    critical_cs.add(c)
            # start_i(c) = end_j(c) => x_i + c*i/N = x_j + c*j/N + 1/N => c = N*(x_j + 1/N - x_i)/(i-j)
            if i != j:
                c = N * (x_base[j] + 1.0/N - x_base[i]) / (i - j)
                if 0.0 <= c <= 1.0:
                    critical_cs.add(c)
            # end_i(c) = start_j(c) => x_i + c*i/N + 1/N = x_j + c*j/N => c = N*(x_j - x_i - 1/N)/(i-j)
            if i != j:
                c = N * (x_base[j] - x_base[i] - 1.0/N) / (i - j)
                if 0.0 <= c <= 1.0:
                    critical_cs.add(c)
            # end_i(c) = end_j(c) => same as start_i = start_j
            
    # y-crossings
    for i in range(N):
        for j in range(N):
            if i != j:
                c = N * (y_base[j] - y_base[i]) / (i - j)
                if 0.0 <= c <= 1.0:
                    critical_cs.add(c)
            if i != j:
                c = N * (y_base[j] + 1.0/N - y_base[i]) / (i - j)
                if 0.0 <= c <= 1.0:
                    critical_cs.add(c)
            if i != j:
                c = N * (y_base[j] - y_base[i] - 1.0/N) / (i - j)
                if 0.0 <= c <= 1.0:
                    critical_cs.add(c)

    critical_cs = np.array(sorted(list(critical_cs)))
    
    def _area_at_c(c_val):
        # Get x-intervals
        x_ints = []
        for i in range(N):
            s = x_base[i] + c_val * i / N
            e = s + 1.0 / N
            x_ints.append((s, e))
        
        # Get y-intervals
        y_ints = []
        for j in range(N):
            s = y_base[j] + c_val * j / N
            e = s + 1.0 / N
            y_ints.append((s, e))
        
        # Merge x-intervals
        x_ints.sort(key=lambda x: x[0])
        merged_x = []
        for s, e in x_ints:
            if not merged_x:
                merged_x.append([s, e])
            else:
                if s <= merged_x[-1][1]:
                    merged_x[-1][1] = max(merged_x[-1][1], e)
                else:
                    merged_x.append([s, e])
        
        # Merge y-intervals
        y_ints.sort(key=lambda y: y[0])
        merged_y = []
        for s, e in y_ints:
            if not merged_y:
                merged_y.append([s, e])
            else:
                if s <= merged_y[-1][1]:
                    merged_y[-1][1] = max(merged_y[-1][1], e)
                else:
                    merged_y.append([s, e])
        
        # Calculate area of union of rectangles
        # The union of rectangles is the sum of areas of merged_x segments intersected with merged_y segments?
        # No, that's the area of the union of the *projections*.
        # The actual union of rectangles is NOT simply the product of the union of projections.
        # Example: two disjoint squares. Union of projections is one long strip. Product is large. Union of squares is small.
        # We need the area of the union of the 64 rectangles.
        
        # Since N=8, we have 8x8=64 rectangles.
        # We can compute the area of the union of these rectangles directly.
        # A robust way: sweep line in x, maintaining active y-intervals.
        # Or, since the rectangles are axis-aligned, we can use a grid? No, continuous.
        
        # Let's use the standard algorithm for union of rectangles.
        # 1. Collect all x-coordinates of vertical edges.
        # 2. Collect all y-coordinates of horizontal edges.
        # 3. Create a grid of cells defined by these coordinates.
        # 4. For each cell, check if it is covered by at least one rectangle.
        # 5. Sum the areas of covered cells.
        
        # This is efficient because the number of unique x-coords is at most 2*N = 16.
        # Number of unique y-coords is at most 2*N = 16.
        # Total cells: 15 * 15 = 225.
        # For each cell, check against 64 rectangles. 225 * 64 = 14400 ops. Very fast.
        
        x_coords = set()
        for s, e in merged_x: # Wait, merged_x are the union of x-projections.
            # We need the actual rectangle edges, not the merged projections.
            pass
            
        # Let's just collect all rectangle edges.
        rect_edges_x = []
        rect_edges_y = []
        rects = []
        for i in range(N):
            for j in range(N):
                x_s = x_base[i] + c_val * i / N
                x_e = x_s + 1.0 / N
                y_s = y_base[j] + c_val * j / N
                y_e = y_s + 1.0 / N
                rects.append((x_s, x_e, y_s, y_e))
                rect_edges_x.extend([x_s, x_e])
                rect_edges_y.extend([y_s, y_e])
                
        unique_x = sorted(list(set(rect_edges_x)))
        unique_y = sorted(list(set(rect_edges_y)))
        
        area = 0.0
        for ix in range(len(unique_x) - 1):
            x1, x2 = unique_x[ix], unique_x[ix+1]
            for iy in range(len(unique_y) - 1):
                y1, y2 = unique_y[iy], unique_y[iy+1]
                # Check if this cell [x1, x2] x [y1, y2] is covered by any rectangle
                covered = False
                for rx_s, rx_e, ry_s, ry_e in rects:
                    if rx_s <= x1 and x2 <= rx_e and ry_s <= y1 and y2 <= ry_e:
                        covered = True
                        break
                if covered:
                    area += (x2 - x1) * (y2 - y1)
        return area

    total_vol = 0.0
    for idx in range(len(critical_cs) - 1):
        c1 = critical_cs[idx]
        c2 = critical_cs[idx+1]
        dc = c2 - c1
        if dc <= 0:
            continue
        A1 = _area_at_c(c1)
        A2 = _area_at_c(c2)
        total_vol += 0.5 * (A1 + A2) * dc
        
    return total_vol

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    # Initial guess: Keich construction for both x and y
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # We can scale the Keich positions to fit better?
    # The Keich positions are in [0, 1].
    # Let's try a few initializations:
    # 1. Keich
    # 2. Keich scaled by 0.5
    # 3. Random
    
    best_params = np.concatenate([keich, keich])
    best_vol = _compute_3d_volume(best_params)
    
    # Optimization
    # Bounds: x_i, y_i should be in a reasonable range.
    # The tubes have width 1/8. The shearing is at most 1/8.
    # If we pack them tightly, they should be within [0, 1] or slightly outside.
    # Let's use bounds [-0.5, 1.5] for all.
    bounds = [(-0.5, 1.5)] * (2 * N)
    
    # Finite difference gradient
    # L-BFGS-B with numerical gradient is robust.
    
    def objective(params):
        if time.time() > deadline:
            return best_vol
        return _compute_3d_volume(params)
    
    # Run L-BFGS-B
    try:
        res = minimize(
            objective,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.fun < best_vol:
            best_params = res.x
            best_vol = res.fun
    except Exception:
        pass
    
    # If we have time, try a second run with a different starting point
    if time.time() < deadline - 60:
        # Try a more compact initial guess
        # Maybe cluster them?
        # Or use differential evolution for a short time to find a better basin?
        # DE is slow. Let's just try a perturbed Keich.
        np.random.seed(42)
        perturbed = best_params + np.random.normal(0, 0.1, 2*N)
        # Clip to bounds
        perturbed = np.clip(perturbed, -0.5, 1.5)
        
        try:
            res2 = minimize(
                objective,
                perturbed,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if res2.fun < best_vol:
                best_params = res2.x
                best_vol = res2.fun
        except Exception:
            pass
            
    # Construct output
    x_base = best_params[:N]
    y_base = best_params[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_base[i_flat]
    base_y = y_base[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
