import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """
    Check if circles are valid: disjoint interiors, within bounds.
    pos_r: (n, 3) array of [x, y, r]
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    # Check radii are positive
    if np.any(rs < 1e-9):
        return False
    
    # Check bounds: circles must be inside [0, w] x [0, h]
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    # Check non-overlap
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _resolve_collisions(circles, w, h, max_iters=2000, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    circles: (n, 3) array [x, y, r]
    Returns modified circles array.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints: push circles inside [0, w] x [0, h]
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        
        # 2. Pairwise collisions: push apart
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                
                if dist < min_dist:
                    if dist < 1e-9:
                        # Random perturbation if exactly overlapping
                        angle = np.random.uniform(0, 2*np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart by half the overlap each
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    # Final clamp to ensure boundary validity
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
        
    return np.column_stack((pos, rad))

def _generate_hex_init(w, h, n_circles):
    """
    Generate initial circle positions in a hexagonal lattice pattern.
    This typically yields denser packings than square grids.
    """
    circles = np.zeros((n_circles, 3))
    
    # Estimate radius based on area
    # Area of hexagon with side r is (3*sqrt(3)/2)*r^2
    # For a packing density of ~0.9 in a rectangle, we can estimate r
    # Let's start with a simple heuristic: r ~ sqrt(w*h / (n * 1.2))
    r_est = np.sqrt((w * h) / (n_circles * 1.1))
    
    # Generate hex grid points
    # Vertical spacing: sqrt(3) * r
    # Horizontal spacing: 1.5 * r (offset rows)
    # We will generate a grid slightly larger than needed and pick the best ones
    # Or just fill row by row.
    
    # Let's use a simple approach: 
    # Row 0: x = r, 3r, 5r...
    # Row 1: x = 2r, 4r... (shifted by r)
    # y spacing = sqrt(3)*r
    
    rows = []
    y = r_est
    row_idx = 0
    while y < h:
        x = r_est if row_idx % 2 == 0 else 2*r_est
        while x < w:
            rows.append([x, y, r_est])
            x += 2*r_est
        y += np.sqrt(3) * r_est
        row_idx += 1
        
    # If we don't have enough points, reduce radius and try again?
    # For now, just take the first n_circles if available, otherwise fallback to grid.
    if len(rows) < n_circles:
        # Fallback to simple grid if hex doesn't fit enough
        return _generate_grid_init(w, h, n_circles, 5, 5)
        
    # Select the first n_circles
    # It's better to select those closest to the center or just the first ones?
    # Just taking the first ones is fine for initialization.
    for i in range(n_circles):
        circles[i] = rows[i]
        
    return circles

def _generate_grid_init(w, h, n_circles, cols, rows):
    """
    Generate initial circle positions in a grid pattern.
    """
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            # Initial radius: half the min cell size, slightly smaller
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
            
    return circles

def _tighten_positions(circles, w, h, deadline):
    """
    Optimize only positions to minimize the sum of squared 'slacks' (clearances).
    This pushes circles into corners and against each other, creating a 'tight' configuration.
    Radii are fixed.
    """
    if time.time() > deadline - 10.0:
        return circles

    n = len(circles)
    pos0 = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    # Objective: Minimize sum of (clearance)^2 for all pairs and walls
    # We want clearances to be as small as possible (but >= 0).
    # Since we start from a valid config, clearances are >= 0.
    # Minimizing sum of clearances^2 will drive them to 0.
    
    def objective(pos):
        # pos is (n, 2)
        slacks = []
        # Wall slacks
        for i in range(n):
            x, y = pos[i]
            r = rad[i]
            slacks.append(max(0, x - r))
            slacks.append(max(0, w - x - r))
            slacks.append(max(0, y - r))
            slacks.append(max(0, h - y - r))
        
        # Pair slacks
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                slack = dist - (rad[i] + rad[j])
                slacks.append(max(0, slack))
                
        return np.sum(np.array(slacks)**2)

    # Bounds for positions
    bounds = []
    for i in range(n):
        bounds.append((rad[i], w - rad[i]))
        bounds.append((rad[i], h - rad[i]))
        
    try:
        res = minimize(
            objective,
            pos0.flatten(),
            method='L-BFGS-B', # L-BFGS-B is fast and good for bound-constrained problems
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.success:
            new_pos = res.x.reshape(-1, 2)
            # Ensure validity (L-BFGS-B should respect bounds, but check overlaps just in case)
            # If it caused overlaps (unlikely if starting valid and minimizing slack), fallback
            new_circles = np.column_stack((new_pos, rad))
            if _validate_solution(new_circles, w, h, N):
                return new_circles
    except Exception:
        pass
        
    return circles

def _slsqp_optimize(circles, w, h, deadline):
    """
    Optimize radii and positions using SLSQP with hard constraints.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = circles.flatten()

    # Objective: maximize sum of radii => minimize negative sum
    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)

    # Constraints
    constraints = []

    # Boundary constraints
    for i in range(n):
        # x >= r  => x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        # x + r <= w => w - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        # y >= r  => y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        # y + r <= h => h - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        # r > 0 => r >= 1e-6
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-6})

    # Overlap constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w)) # x
        bounds.append((1e-6, h)) # y
        bounds.append((1e-6, min(w, h)/2)) # r

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9} # Reduced maxiter to allow more restarts/aspect ratios
        )

        if res.success:
            new_circles = res.x.reshape(-1, 3)
            # Validate and fix any tiny numerical issues
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                # Try to fix with relaxation
                fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass

    return None

def _solve_aspect(w, h, deadline):
    """
    Solve for a specific aspect ratio.
    1. Generate hex/grid init.
    2. Relax.
    3. Tighten positions (L-BFGS-B on positions only).
    4. SLSQP optimize (Joint).
    """
    # Try hex init first, then grid configs
    best_init = None
    best_init_score = 0.0
    
    # 1. Hexagonal Init
    hex_init = _generate_hex_init(w, h, N)
    if hex_init is not None:
        relaxed = _resolve_collisions(hex_init, w, h, max_iters=200)
        if _validate_solution(relaxed, w, h, N):
            score = np.sum(relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = relaxed

    # 2. Grid Inits (as fallback or alternative)
    grid_configs = [(4, 5), (5, 4), (5, 5), (3, 7), (7, 3), (4, 6), (6, 4)]
    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=100)
        if _validate_solution(relaxed, w, h, N):
            score = np.sum(relaxed[:, 2])
            if score > best_init_score:
                best_init_score = score
                best_init = relaxed

    if best_init is None:
        return None

    # 3. Tighten Positions (New Step)
    if time.time() < deadline - 15.0:
        tightened = _tighten_positions(best_init, w, h, deadline)
        if tightened is not None:
            best_init = tightened

    # 4. SLSQP Refinement
    if time.time() < deadline - 5.0:
        optimized = _slsqp_optimize(best_init, w, h, deadline)
        if optimized is not None:
            return optimized

    return best_init

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try
    # Focus on ratios that might allow denser packing for N=21
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 2.0, 0.55]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        result = _solve_aspect(w, h, deadline)
        
        if result is not None and _validate_solution(result, w, h, N):
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        cols, rows = 5, 5
        best_result = _generate_grid_init(w, h, N, cols, rows)
        best_result = _resolve_collisions(best_result, w, h, max_iters=100)
        if not _validate_solution(best_result, w, h, N):
            for i in range(N):
                x, y, r = best_result[i]
                min_d = min(x, w-x, y, h-y)
                best_result[i, 2] = min(r, min_d - 1e-4)
            best_result = _resolve_collisions(best_result, w, h, max_iters=100)

    return best_result
