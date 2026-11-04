import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import linprog, minimize

N = 26
DEADLINE_MARGIN = 12.0

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    best_total = 0.0
    best_xy = None
    best_r = None
    best_valid = False

    def _eval_packing(xy, r):
        """Validate packing and return (total_radius, is_valid)."""
        n = len(xy)
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if x - ri < -1e-8 or 1.0 - x - ri < -1e-8 or y - ri < -1e-8 or 1.0 - y - ri < -1e-8:
                return 0.0, False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-8:
                    return 0.0, False
        return float(np.sum(r)), True

    def _build_constraint_matrix(xy):
        """
        Build A and b for the LP: maximize sum(r_i)
        s.t. r_i + r_j <= dist_ij  (for adjacent pairs via Delaunay + wall-adjacent)
             r_i <= min(x_i, 1-x_i, y_i, 1-y_i)  (wall constraint)
             r_i >= 0
        """
        n = len(xy)
        # Delaunay triangulation for circle-circle constraints
        try:
            tri = Delaunay(xy)
            edges = set()
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = simplex[a], simplex[b]
                        edges.add((min(i, j), max(i, j)))
        except Exception:
            edges = set()
        
        # Add wall constraints
        constraints = []  # list of (coeffs, rhs) where coeffs is sparse-ish
        
        # Circle-circle constraints from Delaunay
        for (i, j) in edges:
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            # r_i + r_j <= dist
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraints.append((row, dist))
        
        # Also add some non-Delaunay pairs that might be close
        # (to catch cases where Delaunay misses a close pair)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in edges:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # Only add if distance is small (potential overlap)
                if dist < 0.4:
                    row = np.zeros(n)
                    row[i] = 1.0
                    row[j] = 1.0
                    constraints.append((row, dist))
        
        # Wall constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        for i in range(n):
            wall_max = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            row = np.zeros(n)
            row[i] = 1.0
            constraints.append((row, wall_max))
        
        # Build A, b for linprog (minimize -sum(r), s.t. A_ub @ r <= b_ub, r >= 0)
        m = len(constraints)
        A_ub = np.zeros((m, n))
        b_ub = np.zeros(m)
        for k, (row, rhs) in enumerate(constraints):
            A_ub[k, :] = row
            b_ub[k] = rhs
        
        c = -np.ones(n)  # minimize -sum(r) = maximize sum(r)
        bounds = [(0.0, None)] * n
        
        return A_ub, b_ub, c, bounds

    def _solve_radii_lp(xy):
        """Solve LP to maximize sum of radii given fixed centers."""
        try:
            A_ub, b_ub, c, bounds = _build_constraint_matrix(xy)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success and res.x is not None:
                r = np.maximum(res.x, 0.0)
                # Clamp to wall
                for i in range(N):
                    wall_max = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                    r[i] = min(r[i], wall_max)
                return r
        except Exception:
            pass
        # Fallback: greedy assignment
        r = np.zeros(N)
        for i in range(N):
            r[i] = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            for j in range(i):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                r[i] = min(r[i], max(0.0, dist - r[j]))
        return r

    def _grow_radii(xy, r, deadline):
        """
        Iteratively grow radii using LP and then move centers to increase room.
        """
        n = N
        total, valid = _eval_packing(xy, r)
        if not valid:
            # Try to repair
            r = _solve_radii_lp(xy)
            total, valid = _eval_packing(xy, r)
            if not valid:
                return xy, r, total, valid
        
        best_total = total
        best_xy = xy.copy()
        best_r = r.copy()
        
        for it in range(30):
            if time.time() > deadline:
                break
            
            # Re-solve LP with current centers
            new_r = _solve_radii_lp(xy)
            new_total, new_valid = _eval_packing(xy, new_r)
            if new_valid and new_total > best_total:
                best_total = new_total
                best_xy = xy.copy()
                best_r = new_r.copy()
            
            # Move each center toward the "empty" direction to create room
            for i in range(n):
                if time.time() > deadline:
                    break
                
                # Compute pressure from each direction
                # Move circle i in the direction that maximizes the minimum clearance
                # Simple: try 8 directions + stay
                best_move = 0
                best_new_xy = xy.copy()
                step = 0.02
                directions = [(0,0), (1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
                for (dx, dy) in directions:
                    if dx == 0 and dy == 0:
                        continue
                    new_x = xy[i, 0] + dx * step
                    new_y = xy[i, 1] + dy * step
                    # Keep in bounds
                    ri = best_r[i]
                    if new_x < ri or new_x > 1.0 - ri or new_y < ri or new_y > 1.0 - ri:
                        continue
                    test_xy = xy.copy()
                    test_xy[i, 0] = new_x
                    test_xy[i, 1] = new_y
                    test_r = _solve_radii_lp(test_xy)
                    test_total, test_valid = _eval_packing(test_xy, test_r)
                    if test_valid and test_total > best_total + 1e-10:
                        best_total = test_total
                        best_new_xy = test_xy
                        best_move = 1
            
            if best_move:
                xy = best_new_xy
                r = _solve_radii_lp(xy)
                total, valid = _eval_packing(xy, r)
                if not valid:
                    xy = best_xy.copy()
                    r = best_r.copy()
                    break
            
            # Check if no improvement in last few iterations
            current_total, current_valid = _eval_packing(xy, r)
            if current_valid and current_total > best_total:
                best_total = current_total
                best_xy = xy.copy()
                best_r = r.copy()
        
        return best_xy, best_r, best_total, best_valid

    def _delaunay_objective(params, xy_fixed_mask, deadline):
        """
        Objective for moving centers: minimize negative sum of LP-optimal radii.
        """
        n = N
        xy = np.zeros((n, 2))
        xy[:] = params.reshape(n, 2)
        
        # Keep some centers fixed if masked
        if xy_fixed_mask is not None:
            xy[xy_fixed_mask] = original_xy[xy_fixed_mask]
        
        # Solve LP
        r = _solve_radii_lp(xy)
        
        # Penalty for invalid
        total, valid = _eval_packing(xy, r)
        if not valid:
            return 1000.0 - total
        return -total

    def _optimize_centers(xy, r, deadline):
        """
        Optimize center positions using L-BFGS-B with LP-based objective.
        """
        n = N
        params = xy.flatten()
        bounds = []
        for i in range(n):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        
        global original_xy
        original_xy = xy.copy()
        
        best_params = params.copy()
        best_total = np.sum(r)
        best_valid = True
        
        try:
            result = minimize(
                _delaunay_objective,
                params,
                args=(None, deadline),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            new_xy = result.x.reshape(n, 2)
            new_r = _solve_radii_lp(new_xy)
            new_total, new_valid = _eval_packing(new_xy, new_r)
            if new_valid and new_total > best_total:
                best_total = new_total
                best_params = result.x.copy()
        except Exception:
            pass
        
        final_xy = best_params.reshape(n, 2)
        final_r = _solve_radii_lp(final_xy)
        final_total, final_valid = _eval_packing(final_xy, final_r)
        
        return final_xy, final_r, final_total, final_valid

    def _init_hex():
        """Hexagonal packing initialization."""
        positions = []
        # Hexagonal lattice
        spacing = 0.22
        rows = int(1.0 / (spacing * 0.866)) + 1
        for i in range(rows):
            y = i * spacing * 0.866
            offset = (i % 2) * spacing * 0.5
            j = 0
            while offset + j * spacing <= 1.0 and len(positions) < N:
                x = offset + j * spacing
                if 0 <= x <= 1.0 and 0 <= y <= 1.0:
                    positions.append([x, y, 0.1])
                j += 1
        while len(positions) < N:
            positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.05])
        return np.array(positions[:N])

    def _init_grid(side):
        """Grid initialization."""
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.45 / side
                    positions.append([x, y, r])
        while len(positions) < N:
            positions.append([0.5, 0.5, 0.02])
        return np.array(positions[:N])

    def _init_random():
        """Random initialization."""
        positions = []
        for _ in range(N):
            positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.05])
        return np.array(positions)

    def _init_corner():
        """Corner-biased initialization for large circles."""
        positions = []
        corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        for cx, cy in corners:
            positions.append([cx, cy, 0.14])
        # Fill middle
        for _ in range(N - 4):
            positions.append([np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7), 0.08])
        return np.array(positions[:N])

    # Run multiple strategies
    strategies = [
        lambda: _init_hex(),
        lambda: _init_grid(5),
        lambda: _init_grid(6),
        lambda: _init_corner(),
        lambda: _init_random(),
    ]

    # Multi-start with diverse initializations
    for restart in range(8):
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        
        strat_idx = (restart * 3) % len(strategies)
        try:
            init_pos = strategies[strat_idx]()
            xy = init_pos[:, :2].copy()
            
            # Ensure centers are in bounds
            xy = np.clip(xy, 0.02, 0.98)
            
            # Step 1: Solve LP for initial radii
            r = _solve_radii_lp(xy)
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()
            
            # Step 2: Grow radii iteratively
            if time.time() < deadline - DEADLINE_MARGIN + 5:
                out_xy, out_r, out_total, out_valid = _grow_radii(xy, r, deadline - DEADLINE_MARGIN + 5)
                if out_valid and out_total > best_total:
                    best_total = out_total
                    best_valid = True
                    best_xy = out_xy.copy()
                    best_r = out_r.copy()
            
            # Step 3: Optimize centers
            if time.time() < deadline - DEADLINE_MARGIN + 10 and best_valid:
                opt_xy, opt_r, opt_total, opt_valid = _optimize_centers(best_xy, best_r, deadline - DEADLINE_MARGIN + 10)
                if opt_valid and opt_total > best_total:
                    best_total = opt_total
                    best_valid = True
                    best_xy = opt_xy.copy()
                    best_r = opt_r.copy()
                    
                # Step 4: Final grow after center optimization
                if time.time() < deadline - DEADLINE_MARGIN + 15 and best_valid:
                    out_xy2, out_r2, out_total2, out_valid2 = _grow_radii(best_xy, best_r, deadline - DEADLINE_MARGIN + 15)
                    if out_valid2 and out_total2 > best_total:
                        best_total = out_total2
                        best_valid = True
                        best_xy = out_xy2.copy()
                        best_r = out_r2.copy()
                        
        except Exception:
            continue

    # Final fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
