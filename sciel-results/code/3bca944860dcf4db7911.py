"""
Circle packing in perimeter-4 rectangle: 21 circles.
Uses a grid-based cellular decomposition approach:
1. Partition the rectangle into a grid of cells.
2. Assign one circle to each cell.
3. Each circle's radius is constrained by its cell's inradius (distance to nearest edge).
4. Optimize cell boundaries and aspect ratio to maximize sum of radii.
This inherently guarantees disjoint interiors since circles are confined to non-overlapping cells.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def check_validity(x, y, r, w, tol=1e-9):
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        
        # Boundary checks
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
            
        # Overlap checks
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum * r_sum - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    def compute_cell_inradii(grid_w, grid_h, cols, rows, x_edges, y_edges):
        """
        Given a grid partition defined by x_edges (length cols+1) and y_edges (length rows+1),
        compute the inradius of each cell (distance from cell center to nearest edge).
        Returns an array of shape (rows, cols) with inradii, and arrays of cell centers.
        """
        inradii = np.zeros((rows, cols))
        centers_x = np.zeros((rows, cols))
        centers_y = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                cx = (x_edges[j] + x_edges[j+1]) / 2.0
                cy = (y_edges[i] + y_edges[i+1]) / 2.0
                centers_x[i, j] = cx
                centers_y[i, j] = cy
                # Inradius is min distance to cell boundaries
                inr = min(cx - x_edges[j], x_edges[j+1] - cx, cy - y_edges[i], y_edges[i+1] - cy)
                inradii[i, j] = inr
        
        return inradii, centers_x, centers_y

    def objective_cell(params, cols, rows, w):
        """
        Objective function for optimizing cell boundaries.
        params: flattened array of internal x boundaries and internal y boundaries.
        x_edges: [0, p0, p1, ..., p_{cols-2}, w]
        y_edges: [0, q0, q1, ..., q_{rows-2}, h]
        """
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return 1e10
        
        # Extract internal boundaries
        x_int = params[:cols-1]
        y_int = params[cols-1:cols-1+rows-1]
        
        # Build full edge arrays
        x_edges = np.concatenate([[0.0], x_int, [w]])
        y_edges = np.concatenate([[0.0], y_int, [h]])
        
        # Check monotonicity
        if np.any(np.diff(x_edges) <= 0) or np.any(np.diff(y_edges) <= 0):
            return 1e10
        
        inradii, cx, cy = compute_cell_inradii(w, h, cols, rows, x_edges, y_edges)
        
        # We have cols*rows cells but only N=21 circles.
        # We need to assign circles to cells. 
        # Strategy: Use the largest N cells. Or better, design the grid to have exactly N cells?
        # No, grid is rectangular. cols*rows >= N.
        # If cols*rows > N, we have empty cells. We can merge or just ignore empty cells.
        # But wait, if we have empty cells, the circles in adjacent non-empty cells might be close.
        # The inradius constraint ensures circle fits in cell.
        # If two circles are in adjacent cells, their distance is at least the sum of their distances to the shared boundary?
        # Yes, if circle A is in cell (i,j) and circle B in (i,j+1),
        # x_A <= x_edges[j+1] - r_A and x_B >= x_edges[j+1] - r_B.
        # So x_B - x_A >= r_A + r_B. Thus no overlap in x, so no overlap at all.
        # This is the key property of grid packing!
        
        # So we just need to pick which cells to fill.
        # To maximize sum of radii, we should fill the cells with the largest inradii.
        # But which cells to leave empty?
        # If we leave a cell empty, its neighbors might be able to have larger radii? No, radii are fixed by cell geometry.
        # So we just pick the top N inradii.
        
        total_r = 0.0
        # Flatten inradii
        flat_inr = inradii.flatten()
        
        # We need to select N cells.
        # To keep it simple and differentiable, we can just sum the top N values?
        # Top-k is not differentiable.
        # Alternative: Assume a regular grid where we fill all cells if possible, or use a specific pattern.
        # For N=21, common grids:
        # 3x7, 7x3, 4x6 (24 cells, 3 empty), 5x5 (25 cells, 4 empty), 6x4, etc.
        
        # Let's just sum the top N inradii for the objective.
        # To make it differentiable, we can use a soft-max or just assume we fill a specific subset.
        # Let's assume we fill the first N cells in row-major order? No, that might be suboptimal.
        # Let's try to make the grid have exactly N cells by allowing some cells to be "merged" or "skipped".
        # Actually, the simplest differentiable approach:
        # If cols*rows == N, sum all inradii.
        # If cols*rows > N, we have a choice.
        # Let's restrict to cases where cols*rows == N or cols*rows is slightly larger and we use a heuristic.
        
        # For now, let's assume cols*rows == N for the optimization variables to be well-defined.
        # If cols*rows != N, we return a large penalty.
        if cols * rows != N:
            return 1e10
            
        return -np.sum(flat_inr)

    def run_grid_optimization(cols, rows, w_init, time_budget):
        """
        Optimize a specific grid layout (cols x rows) with initial width w_init.
        """
        h_init = 2.0 - w_init
        
        # Initial guess for internal boundaries: uniform
        x_int_init = np.linspace(0, w_init, cols+1)[1:-1]
        y_int_init = np.linspace(0, h_init, rows+1)[1:-1]
        params_init = np.concatenate([x_int_init, y_int_init])
        
        # Bounds for internal boundaries
        # x_int: 0 < x_i < w
        # y_int: 0 < y_i < h
        # We need to optimize w as well?
        # Let's fix w for now and optimize boundaries. Then we can optimize w separately.
        
        bounds = [(1e-6, w_init - 1e-6)] * (cols - 1) + [(1e-6, h_init - 1e-6)] * (rows - 1)
        
        best_val = 0.0
        best_params = params_init.copy()
        
        # Use scipy minimize
        try:
            res = minimize(
                objective_cell,
                params_init,
                args=(cols, rows, w_init),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if res.success or res.fun < 0:
                best_params = res.x
                val = -res.fun
                if val > best_val:
                    best_val = val
        except Exception:
            pass
        
        # Construct the solution from best_params
        x_int = best_params[:cols-1]
        y_int = best_params[cols-1:cols-1+rows-1]
        
        x_edges = np.concatenate([[0.0], x_int, [w_init]])
        y_edges = np.concatenate([[0.0], y_int, [h_init]])
        
        inradii, cx, cy = compute_cell_inradii(w_init, h_init, cols, rows, x_edges, y_edges)
        
        # Since cols*rows == N, we use all cells
        x = cx.flatten()
        y = cy.flatten()
        r = inradii.flatten()
        
        # Final validity check
        valid, val_final = check_validity(x, y, r, w_init)
        if valid:
            return x, y, r, w_init, val_final
        else:
            return None

    best_value = 0.0
    best_result = None

    # Try different grid configurations
    # N = 21 = 3*7, 7*3
    # Also try 4*6=24 (with 3 empty cells, but our current objective requires exact match)
    # Let's stick to exact matches first: 3x7, 7x3
    # Also 1x21, 21x1 (unlikely to be optimal but valid)
    
    grid_configs = [
        (3, 7),
        (7, 3),
        (1, 21),
        (21, 1),
    ]
    
    # Also try non-integer grids by allowing "merged" cells?
    # For now, just try these.
    
    # We should also optimize the aspect ratio w.
    # Let's sample a range of w values for each grid.
    
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    for cols, rows in grid_configs:
        if time.time() > deadline - 10.0:
            break
            
        for w0 in aspect_ratios:
            if time.time() > deadline - 5.0:
                break
            
            time_budget = deadline - time.time() - 5.0
            if time_budget < 1.0:
                continue
                
            result = run_grid_optimization(cols, rows, w0, time_budget)
            if result is not None:
                x, y, r, w, val = result
                if val > best_value:
                    best_value = val
                    best_result = (x, y, r, w)

    # If we have a result, return it
    if best_result is not None:
        x, y, r, w = best_result
        # Shrink slightly to ensure strict inequality
        r_final = r * (1 - 1e-9)
        return np.column_stack((x, y, r_final))

    # Fallback: Simple grid in unit square
    # Use 3x7 grid in 1x1 square
    cols, rows = 3, 7
    w = 1.0
    h = 1.0
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        row, col = divmod(i, cols)
        cx = (col + 0.5) * (w / cols)
        cy = (row + 0.5) * (h / rows)
        x[i] = cx
        y[i] = cy
        cell_w = w / cols
        cell_h = h / rows
        r[i] = min(cell_w, cell_h) * 0.5 # Inradius of uniform cell
        
    return np.column_stack((x, y, r * (1 - 1e-9)))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
