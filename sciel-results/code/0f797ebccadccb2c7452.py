"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    circles = np.array(circles)
    for cx, cy, r in circles:
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _make_hexagonal_grid(n, w, h):
    """Create a hexagonal packing of n circles in rectangle w x h."""
    circles = []
    # Hexagonal packing: rows offset by half spacing
    # For 21 circles, try 5 rows of 4-5 circles
    # Let's try a hexagonal arrangement
    # Row 0: 5 circles, Row 1: 4 circles (offset), Row 2: 5, Row 3: 4, Row 4: 3
    # Total: 5+4+5+4+3 = 21
    
    row_counts = [5, 4, 5, 4, 3]
    total_rows = len(row_counts)
    
    # Estimate radius from hexagonal packing
    # In hexagonal packing, horizontal spacing = 2r, vertical spacing = sqrt(3)*r
    # Width: max row count * 2r <= w
    # Height: (total_rows - 1) * sqrt(3) * r + 2r <= h
    
    # Solve for r
    max_cols = max(row_counts)
    # w >= max_cols * 2r => r <= w / (2 * max_cols)
    # h >= (total_rows - 1) * sqrt(3) * r + 2r => r <= h / ((total_rows - 1) * sqrt(3) + 2)
    r1 = w / (2 * max_cols)
    r2 = h / ((total_rows - 1) * np.sqrt(3) + 2)
    r = min(r1, r2) * 0.95  # slight shrink for safety
    
    if r <= 0:
        r = 0.05
    
    y_spacing = np.sqrt(3) * r
    x_spacing = 2 * r
    
    y_start = r  # start at bottom
    for row_idx, count in enumerate(row_counts):
        y = y_start + row_idx * y_spacing
        # Center the row
        row_width = (count - 1) * x_spacing
        x_start = (w - row_width) / 2
        for col in range(count):
            if len(circles) >= n:
                break
            x = x_start + col * x_spacing
            circles.append((x, y, r))
    
    # If we have fewer than n, add more
    while len(circles) < n:
        circles.append((w / 2, h / 2, r * 0.5))
    
    circles = circles[:n]
    return circles

def _make_grid(n, w, h, cols, rows):
    """Create a rectangular grid of n circles in rectangle w x h."""
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5) / cols * w
            cy = (i + 0.5) / rows * h
            r = 0.5 / max(cols, rows) * min(w, h) * 0.95
            circles.append((cx, cy, r))
    circles = circles[:n]
    while len(circles) < n:
        circles.append((w / 2, h / 2, 0.05))
    return circles

def _penalty_objective(x, w, h, n, mu):
    """
    Penalty-based objective for circle packing.
    x: array of shape (3*n,) with [cx, cy, r] for each circle
    Returns: negative sum of radii + penalty for violations
    """
    circles = x.reshape(n, 3)
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Sum of radii
    total_r = np.sum(r)
    
    # Penalty for circles outside rectangle
    penalty = 0.0
    for i in range(n):
        if r[i] <= 0:
            penalty += 100 * (1.0 - r[i])
            continue
        # Left wall
        if cx[i] - r[i] < 0:
            penalty += mu * (cx[i] - r[i]) ** 2
        # Right wall
        if cx[i] + r[i] > w:
            penalty += mu * (cx[i] + r[i] - w) ** 2
        # Bottom wall
        if cy[i] - r[i] < 0:
            penalty += mu * (cy[i] - r[i]) ** 2
        # Top wall
        if cy[i] + r[i] > h:
            penalty += mu * (cy[i] + r[i] - h) ** 2
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += mu * (min_dist - dist) ** 2
    
    return -total_r + penalty

def _relax_circles(circles, w, h, max_iter=500, tol=1e-9):
    """
    Simple relaxation method to resolve overlaps.
    """
    circles = [list(c) for c in circles]
    n = len(circles)
    
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i][2] + circles[j][2]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    # Move both circles apart
                    nx = dx / dist
                    ny = dy / dist
                    # Move each by half the overlap
                    circles[i][0] += 0.5 * overlap * nx
                    circles[i][1] += 0.5 * overlap * ny
                    circles[j][0] -= 0.5 * overlap * nx
                    circles[j][1] -= 0.5 * overlap * ny
                    moved = True
            # Keep inside rectangle
            cx, cy, r = circles[i]
            if cx - r < 0:
                circles[i][0] = r
                moved = True
            if cx + r > w:
                circles[i][0] = w - r
                moved = True
            if cy - r < 0:
                circles[i][1] = r
                moved = True
            if cy + r > h:
                circles[i][1] = h - r
                moved = True
        if not moved:
            break
    
    return [tuple(c) for c in circles]

def _slsqp_optimize(circles, w, h, deadline):
    """
    Use SLSQP to maximize sum of radii with constraints.
    """
    n = len(circles)
    x0 = []
    for cx, cy, r in circles:
        x0.extend([cx, cy, r])
    x0 = np.array(x0)
    
    # Bounds: cx in [0, w], cy in [0, h], r > 0
    bounds = []
    for i in range(n):
        bounds.append((0, w))      # cx
        bounds.append((0, h))      # cy
        bounds.append((1e-6, min(w, h) / 2))  # r
    
    # Constraints: non-overlapping
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(i, j):
                def constraint(x):
                    cx_i = x[i * 3]
                    cy_i = x[i * 3 + 1]
                    r_i = x[i * 3 + 2]
                    cx_j = x[j * 3]
                    cy_j = x[j * 3 + 1]
                    r_j = x[j * 3 + 2]
                    dx = cx_i - cx_j
                    dy = cy_i - cy_j
                    dist = np.sqrt(dx * dx + dy * dy)
                    return dist - (r_i + r_j)
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    # Objective: maximize sum of radii
    def objective(x):
        return -sum(x[i * 3 + 2] for i in range(n))
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        
        if res.success or res.fun < 0:
            best_circles = []
            for i in range(n):
                best_circles.append((res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]))
            if _validate(best_circles, w, h):
                return best_circles
    except Exception:
        pass
    
    return None

def _lbfgsb_optimize(circles, w, h, deadline):
    """
    Use L-BFGS-B with penalty method to maximize sum of radii.
    """
    n = len(circles)
    x0 = []
    for cx, cy, r in circles:
        x0.extend([cx, cy, r])
    x0 = np.array(x0)
    
    # Progressive penalty
    mu = 1.0
    best_x = x0.copy()
    best_obj = _penalty_objective(x0, w, h, n, mu)
    
    for mu_val in [1.0, 10.0, 100.0, 1000.0]:
        def obj(x):
            return _penalty_objective(x, w, h, n, mu_val)
        
        try:
            res = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Extract circles
    best_circles = []
    for i in range(n):
        best_circles.append((best_x[i * 3], best_x[i * 3 + 1], best_x[i * 3 + 2]))
    
    if _validate(best_circles, w, h):
        return best_circles
    
    return None

def _high_precision_polish(circles, w, h, deadline):
    """
    Final high-precision polish using SLSQP with tight tolerances.
    """
    n = len(circles)
    x0 = []
    for cx, cy, r in circles:
        x0.extend([cx, cy, r])
    x0 = np.array(x0)
    
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-6, min(w, h) / 2))
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(i, j):
                def constraint(x):
                    cx_i = x[i * 3]
                    cy_i = x[i * 3 + 1]
                    r_i = x[i * 3 + 2]
                    cx_j = x[j * 3]
                    cy_j = x[j * 3 + 1]
                    r_j = x[j * 3 + 2]
                    dx = cx_i - cx_j
                    dy = cy_i - cy_j
                    dist = np.sqrt(dx * dx + dy * dy)
                    return dist - (r_i + r_j)
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    def objective(x):
        return -sum(x[i * 3 + 2] for i in range(n))
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        
        best_circles = []
        for i in range(n):
            best_circles.append((res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]))
        if _validate(best_circles, w, h):
            return best_circles
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    best_circles = None
    best_score = 0.0
    
    # Try various aspect ratios
    aspect_ratios = [
        (1.0, 1.0),
        (0.9, 1.1),
        (1.1, 0.9),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.7, 1.3),
        (1.3, 0.7),
        (0.75, 1.25),
        (1.25, 0.75),
    ]
    
    for w, h in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Try hexagonal grid
        hex_circles = _make_hexagonal_grid(N, w, h)
        if _validate(hex_circles, w, h):
            score = _total_radius(hex_circles)
            if score > best_score:
                best_score = score
                best_circles = hex_circles
            
            # Relax
            relaxed = _relax_circles(hex_circles, w, h)
            if _validate(relaxed, w, h):
                score = _total_radius(relaxed)
                if score > best_score:
                    best_score = score
                    best_circles = relaxed
            
            # L-BFGS-B optimization
            if time.time() < deadline - 10.0:
                opt = _lbfgsb_optimize(relaxed, w, h, deadline)
                if opt is not None:
                    score = _total_radius(opt)
                    if score > best_score:
                        best_score = score
                        best_circles = opt
            
            # SLSQP polish
            if time.time() < deadline - 5.0:
                polished = _slsqp_optimize(best_circles, w, h, deadline)
                if polished is not None:
                    score = _total_radius(polished)
                    if score > best_score:
                        best_score = score
                        best_circles = polished
        
        # Try rectangular grids
        for cols, rows in [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3)]:
            if time.time() > deadline:
                break
            
            grid_circles = _make_grid(N, w, h, cols, rows)
            if _validate(grid_circles, w, h):
                score = _total_radius(grid_circles)
                if score > best_score:
                    best_score = score
                    best_circles = grid_circles
                
                # Relax
                relaxed = _relax_circles(grid_circles, w, h)
                if _validate(relaxed, w, h):
                    score = _total_radius(relaxed)
                    if score > best_score:
                        best_score = score
                        best_circles = relaxed
                
                # L-BFGS-B
                if time.time() < deadline - 10.0:
                    opt = _lbfgsb_optimize(relaxed, w, h, deadline)
                    if opt is not None:
                        score = _total_radius(opt)
                        if score > best_score:
                            best_score = score
                            best_circles = opt
    
    # Final high-precision polish if time allows
    if best_circles is not None and time.time() < deadline - 3.0:
        w, h = best_circles[0][0] + best_circles[0][2], best_circles[0][1] + best_circles[0][2]
        # Actually, we need to track w, h separately
        # Let's just do a final polish on the best circles
        # We need to know the rectangle dimensions
        # For now, skip this if we don't track w, h
    
    if best_circles is not None:
        return np.array(best_circles)
    
    # Fallback
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
