<<<<<<< SEARCH
"""Bootstrap: direct grid-based circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    def validate_and_update(positions, radii, w, h):
        """Validate packing and update best if valid."""
        nonlocal best_result, best_score
        # Check boundaries
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-9:
                return
            if positions[i, 0] + radii[i] > w + 1e-9:
                return
            if positions[i, 1] - radii[i] < -1e-9:
                return
            if positions[i, 1] + radii[i] > h + 1e-9:
                return
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return
        score = float(np.sum(radii))
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()
    
    def compute_optimal_grid_packing(rows, cols, w, h):
        """
        For a given grid of rows x cols, place N circles in row-major order.
        The last row may be incomplete. We optimize the radius by finding the 
        maximum r such that all circles fit without overlap and within bounds.
        
        For a grid with equal cell sizes, the optimal radius is limited by:
        1. The minimum cell dimension / 2 (if circles are centered in cells)
        2. The spacing between adjacent circles
        
        Actually, for a regular grid with cols columns and rows rows,
        the x-spacing is w/cols and y-spacing is h/rows.
        The maximum radius is min(w/cols, h/rows) / 2 if all cells are filled.
        But if the last row is incomplete, we can potentially do better.
        
        Let's think more carefully. For a complete grid of R rows and C cols:
        - Circle centers at (x_i, y_i) where x_i = (i + 0.5) * w/C, y_i = (j + 0.5) * h/R
        - The limiting constraint is the minimum distance between any two centers,
          which for adjacent circles is min(w/C, h/R).
        - So max r = min(w/C, h/R) / 2.
        
        For an incomplete last row with K < C circles:
        - The circles in the last row are still at x positions (0.5, 1.5, ..., K-0.5) * w/C
        - But we could shift them to be more spread out. However, for simplicity,
          let's first try the regular grid placement.
        
        Actually, we can do better. If the last row has only K circles, we have 
        more horizontal space. The optimal placement for the last row would be to 
        spread them evenly across the full width, which gives spacing w/K instead of w/C.
        
        Let's handle this case specifically.
        """
        if rows * cols < N:
            return None, None
        
        positions = np.zeros((N, 2))
        
        for i in range(N):
            row = i // cols
            col = i % cols
            
            if row == rows - 1:
                # Last row - check if it's incomplete
                count_in_last_row = N - (rows - 1) * cols
                if count_in_last_row < cols:
                    # Spread the K circles evenly across the width
                    k = count_in_last_row
                    x = (col + 0.5) * w / k
                else:
                    x = (col + 0.5) * w / cols
            else:
                x = (col + 0.5) * w / cols
            
            y = (row + 0.5) * h / rows
            
            positions[i] = [x, y]
        
        # Compute maximum valid radius
        # The radius is limited by:
        # 1. Distance to boundaries
        # 2. Distance to other circles
        min_r = float('inf')
        
        # Boundary constraints
        for i in range(N):
            min_r = min(min_r, positions[i, 0], w - positions[i, 0],
                       positions[i, 1], h - positions[i, 1])
        
        # Pairwise distance constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_r = min(min_r, dist / 2.0)
        
        radii = np.full(N, min_r)
        return positions, radii
    
    # Try various grid configurations
    # rows * cols >= N, and rows, cols should be reasonable
    # We also need w + h = 2 (perimeter = 4)
    # For each aspect ratio, w = ar * h, so h = 2/(ar+1), w = 2*ar/(ar+1)
    
    # Generate candidate aspect ratios
    # Based on grid dimensions: if rows=r, cols=c, the "natural" aspect ratio is c/r
    # But we should also explore nearby values
    
    candidate_configs = []
    for rows in range(1, 10):
        for cols in range(1, 10):
            if rows * cols >= N:
                ar = cols / rows
                candidate_configs.append((rows, cols, ar))
    
    # Also add some non-integer aspect ratios for flexibility
    for ar in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.33, 2.5, 3.0]:
        # Find best grid for this aspect ratio
        best_rows, best_cols = None, None
        for rows in range(1, 10):
            for cols in range(1, 10):
                if rows * cols >= N:
                    grid_ar = cols / rows
                    if best_rows is None or abs(grid_ar - ar) < abs(best_cols / best_rows - ar):
                        best_rows, best_cols = rows, cols
        if best_rows is not None:
            candidate_configs.append((best_rows, best_cols, ar))
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for rows, cols, ar in candidate_configs:
        key = (rows, cols, round(ar, 3))
        if key not in seen:
            seen.add(key)
            unique_configs.append((rows, cols, ar))
    
    for rows, cols, ar in unique_configs:
        if time.time() > deadline:
            break
        
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        positions, radii = compute_optimal_grid_packing(rows, cols, w, h)
        if positions is None:
            continue
        
        validate_and_update(positions, radii, w, h)
    
    # Also try a few aspect ratios not directly from grid ratios
    extra_ars = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.75, 1.95, 2.15, 2.45, 2.75, 2.95]
    for ar in extra_ars:
        if time.time() > deadline:
            break
        
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Find best grid for this aspect ratio
        best_rows, best_cols = None, None
        best_diff = float('inf')
        for rows in range(1, 10):
            for cols in range(1, 10):
                if rows * cols >= N:
                    grid_ar = cols / rows
                    diff = abs(grid_ar - ar)
                    if diff < best_diff:
                        best_diff = diff
                        best_rows, best_cols = rows, cols
        
        if best_rows is not None:
            positions, radii = compute_optimal_grid_packing(best_rows, best_cols, w, h)
            if positions is not None:
                validate_and_update(positions, radii, w, h)
    
    # Fallback
    if best_result is None:
        # Square packing in 1x1 rectangle
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
=======
"""Bootstrap: direct grid-based circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    def validate_and_update(positions, radii, w, h):
        """Validate packing and update best if valid."""
        nonlocal best_result, best_score
        # Check boundaries
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-9:
                return
            if positions[i, 0] + radii[i] > w + 1e-9:
                return
            if positions[i, 1] - radii[i] < -1e-9:
                return
            if positions[i, 1] + radii[i] > h + 1e-9:
                return
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return
        score = float(np.sum(radii))
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()
    
    def compute_optimal_grid_packing(rows, cols, w, h):
        """
        For a given grid of rows x cols, place N circles in row-major order.
        The last row may be incomplete. We optimize the radius by finding the 
        maximum r such that all circles fit without overlap and within bounds.
        """
        if rows * cols < N:
            return None, None
        
        positions = np.zeros((N, 2))
        
        for i in range(N):
            row = i // cols
            col = i % cols
            
            if row == rows - 1:
                count_in_last_row = N - (rows - 1) * cols
                if count_in_last_row < cols:
                    k = count_in_last_row
                    x = (col + 0.5) * w / k
                else:
                    x = (col + 0.5) * w / cols
            else:
                x = (col + 0.5) * w / cols
            
            y = (row + 0.5) * h / rows
            
            positions[i] = [x, y]
        
        min_r = float('inf')
        
        for i in range(N):
            min_r = min(min_r, positions[i, 0], w - positions[i, 0],
                       positions[i, 1], h - positions[i, 1])
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_r = min(min_r, dist / 2.0)
        
        radii = np.full(N, min_r)
        return positions, radii
    
    def polish_with_slsqp(positions, radii, w, h, time_budget=5.0):
        """
        Use SLSQP to polish the packing by optimizing positions and radii.
        Objective: maximize sum of radii
        Constraints: 
          - Each circle within [0,w] x [0,h]
          - No overlaps: dist(i,j) >= r_i + r_j
        """
        if time.time() > deadline - time_budget:
            return positions, radii, False
        
        n = len(radii)
        # Variables: x_0, y_0, x_1, y_1, ..., x_{n-1}, y_{n-1}, r_0, r_1, ..., r_{n-1}
        x0 = np.concatenate([positions.flatten(), radii])
        
        def objective(x):
            return -np.sum(x[n:])  # minimize negative sum of radii
        
        constraints = []
        
        # Boundary constraints: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[n+i]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (w - x[i]) - x[n+i]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n//2 + i] - x[n+i]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (h - x[n//2 + i]) - x[n+i]})
        
        # Non-negativity of radii
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i]})
        
        # Overlap constraints: dist(i,j) >= r_i + r_j
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: math.sqrt((x[i]-x[j])**2 + (x[n//2+i]-x[n//2+j])**2) - (x[n+i] + x[n+j])
                })
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((0, w))  # x_i
        for i in range(n):
            bounds.append((0, h))  # y_i
        for i in range(n):
            bounds.append((0, min(w, h)/2))  # r_i
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if result.success:
                new_positions = result.x[:n*2].reshape(n, 2)
                new_radii = result.x[n*2:]
                # Validate
                valid = True
                for i in range(n):
                    if new_positions[i, 0] - new_radii[i] < -1e-8:
                        valid = False
                        break
                    if new_positions[i, 0] + new_radii[i] > w + 1e-8:
                        valid = False
                        break
                    if new_positions[i, 1] - new_radii[i] < -1e-8:
                        valid = False
                        break
                    if new_positions[i, 1] + new_radii[i] > h + 1e-8:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            dx = new_positions[i, 0] - new_positions[j, 0]
                            dy = new_positions[i, 1] - new_positions[j, 1]
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist < new_radii[i] + new_radii[j] - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    return new_positions, new_radii, True
        except Exception:
            pass
        
        return positions, radii, False
    
    # Try various grid configurations
    candidate_configs = []
    for rows in range(1, 10):
        for cols in range(1, 10):
            if rows * cols >= N:
                ar = cols / rows
                candidate_configs.append((rows, cols, ar))
    
    for ar in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.33, 2.5, 3.0]:
        best_rows, best_cols = None, None
        for rows in range(1, 10):
            for cols in range(1, 10):
                if rows * cols >= N:
                    grid_ar = cols / rows
                    if best_rows is None or abs(grid_ar - ar) < abs(best_cols / best_rows - ar):
                        best_rows, best_cols = rows, cols
        if best_rows is not None:
            candidate_configs.append((best_rows, best_cols, ar))
    
    seen = set()
    unique_configs = []
    for rows, cols, ar in candidate_configs:
        key = (rows, cols, round(ar, 3))
        if key not in seen:
            seen.add(key)
            unique_configs.append((rows, cols, ar))
    
    # First pass: find best grid packing
    best_grid_positions = None
    best_grid_radii = None
    best_grid_w = None
    best_grid_h = None
    
    for rows, cols, ar in unique_configs:
        if time.time() > deadline - 10.0:
            break
        
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        positions, radii = compute_optimal_grid_packing(rows, cols, w, h)
        if positions is None:
            continue
        
        validate_and_update(positions, radii, w, h)
        
        if positions is not None and (best_grid_positions is None or float(np.sum(radii)) > float(np.sum(best_grid_radii))):
            best_grid_positions = positions.copy()
            best_grid_radii = radii.copy()
            best_grid_w = w
            best_grid_h = h
    
    # Also try extra aspect ratios
    extra_ars = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.75, 1.95, 2.15, 2.45, 2.75, 2.95]
    for ar in extra_ars:
        if time.time() > deadline - 10.0:
            break
        
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        best_rows, best_cols = None, None
        best_diff = float('inf')
        for rows in range(1, 10):
            for cols in range(1, 10):
                if rows * cols >= N:
                    grid_ar = cols / rows
                    diff = abs(grid_ar - ar)
                    if diff < best_diff:
                        best_diff = diff
                        best_rows, best_cols = rows, cols
        
        if best_rows is not None:
            positions, radii = compute_optimal_grid_packing(best_rows, best_cols, w, h)
            if positions is not None:
                validate_and_update(positions, radii, w, h)
                if positions is not None and (best_grid_positions is None or float(np.sum(radii)) > float(np.sum(best_grid_radii))):
                    best_grid_positions = positions.copy()
                    best_grid_radii = radii.copy()
                    best_grid_w = w
                    best_grid_h = h
    
    # Polish the best grid packing with SLSQP
    if best_grid_positions is not None and time.time() < deadline - 5.0:
        polished_pos, polished_rad, polished = polish_with_slsqp(
            best_grid_positions, best_grid_radii, best_grid_w, best_grid_h,
            time_budget=10.0
        )
        if polished:
            validate_and_update(polished_pos, polished_rad, best_grid_w, best_grid_h)
    
    # Fallback
    if best_result is None:
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I need to fix a bug in the SLSQP polish function - the index calculation for y-coordinates is wrong. Let me correct that:

