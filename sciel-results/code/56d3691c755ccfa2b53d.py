import numpy as np
import time
from scipy.optimize import linprog, minimize


def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    # Boundary check
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _sum_sides(sides):
    """Objective: sum of side lengths."""
    return np.sum(sides)


def _flatten(centers, sides):
    """Flatten centers and sides into a single vector."""
    n = len(sides)
    return np.concatenate([centers.flatten(), sides])


def _unflatten(x):
    """Unflatten a vector back into centers and sides."""
    n = 10
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    return centers, sides


def _penalty(x, tol=1e-7, weight=1e4):
    """Penalty for constraint violations on flattened vector."""
    centers, sides = _unflatten(x)
    n = len(sides)
    pen = 0.0
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += weight * (1.0 - s)
        if cx - s/2 < -tol:
            pen += weight * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            pen += weight * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            pen += weight * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            pen += weight * ((cy + s/2) - (1 + tol))
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if abs(dx) < t and abs(dy) < t:
                pen += weight * ((t - abs(dx)) + (t - abs(dy)))
    return pen


def _objective(x, tol=1e-7, weight=1e4):
    """Combined objective: maximize sum of sides minus penalty (negated for minimization)."""
    centers, sides = _unflatten(x)
    return -np.sum(sides) + _penalty(x, tol, weight)


def _make_2big_8small():
    """Create initial config: 2 large squares in top row, 8 small in bottom 2 rows."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 2 big squares in top: each 0.5 x 0.5
    s_big = 0.5
    centers[0] = [0.25, 0.75]
    centers[1] = [0.75, 0.75]
    sides[0] = s_big
    sides[1] = s_big
    
    # 8 small squares in bottom: 4 per row, each 0.25 x 0.25
    s_small = 0.25
    for i in range(4):
        # Row 1 (y=0.375)
        centers[2+i] = [(i+0.5)*s_small, 0.375]
        sides[2+i] = s_small
        # Row 2 (y=0.125)
        centers[6+i] = [(i+0.5)*s_small, 0.125]
        sides[6+i] = s_small
    
    return centers, sides


def _make_3big_7small():
    """Create initial config: 3 large squares on top, 7 smaller below."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 3 big squares in top row
    s_big = 1.0 / 3
    for i in range(3):
        centers[i] = [(i + 0.5) * s_big, 1.0 - s_big/2]
        sides[i] = s_big
    
    # 7 small squares in bottom area
    # Try a 4+3 arrangement
    s_small = 0.2
    # Row 1: 4 squares
    for i in range(4):
        centers[3+i] = [(i + 0.5) * s_small, 0.6]
        sides[3+i] = s_small
    # Row 2: 3 squares
    for i in range(3):
        centers[7+i] = [(i + 0.5) * s_small + 0.1, 0.3]
        sides[7+i] = s_small
    
    return centers, sides


def _make_grid_fallback():
    """Fallback grid configuration."""
    n = 10
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _solve_lp_with_separations(sep_matrix, tol=1e-7):
    """
    Solve LP given a separation matrix.
    sep_matrix[i,j] = 0 means horizontal separation (i left of j or vice versa)
    sep_matrix[i,j] = 1 means vertical separation (i below j or vice versa)
    We use the convention: for each pair, we pick one direction.
    
    Variables: x_i, y_i (centers), s_i (sides) for i=0..9
    Total: 30 variables
    
    Constraints:
    1. Boundary: x_i - s_i/2 >= 0, x_i + s_i/2 <= 1, etc.
    2. Separation: for each pair (i,j):
       If horizontal: |x_i - x_j| >= (s_i + s_j)/2
       If vertical: |y_i - y_j| >= (s_i + s_j)/2
       
    We linearize by choosing direction: x_j - x_i >= (s_i + s_j)/2 or x_i - x_j >= (s_i + s_j)/2
    """
    n = 10
    # Variables: [x_0, y_0, s_0, x_1, y_1, s_1, ..., x_9, y_9, s_9]
    nvar = 3 * n
    
    # Objective: maximize sum(s_i) -> minimize -sum(s_i)
    c = np.zeros(nvar)
    for i in range(n):
        c[3*i + 2] = -1.0  # coefficient for s_i
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_vals = []
    
    # Boundary constraints:
    # x_i - s_i/2 >= 0  ->  -x_i + s_i/2 <= 0
    # x_i + s_i/2 <= 1  ->   x_i + s_i/2 <= 1
    # y_i - s_i/2 >= 0  ->  -y_i + s_i/2 <= 0
    # y_i + s_i/2 <= 1  ->   y_i + s_i/2 <= 1
    for i in range(n):
        # -x_i + s_i/2 <= 0
        row = np.zeros(nvar)
        row[3*i] = -1.0
        row[3*i+2] = 0.5
        A_rows.append(row)
        b_vals.append(0.0)
        
        # x_i + s_i/2 <= 1
        row = np.zeros(nvar)
        row[3*i] = 1.0
        row[3*i+2] = 0.5
        A_rows.append(row)
        b_vals.append(1.0)
        
        # -y_i + s_i/2 <= 0
        row = np.zeros(nvar)
        row[3*i+1] = -1.0
        row[3*i+2] = 0.5
        A_rows.append(row)
        b_vals.append(0.0)
        
        # y_i + s_i/2 <= 1
        row = np.zeros(nvar)
        row[3*i+1] = 1.0
        row[3*i+2] = 0.5
        A_rows.append(row)
        b_vals.append(1.0)
    
    # Separation constraints:
    # For pair (i, j) with i < j:
    # If horizontal: need x_j - x_i >= (s_i+s_j)/2 OR x_i - x_j >= (s_i+s_j)/2
    # We pick one direction based on sep_matrix.
    # sep_matrix[i,j] = 0: horizontal, assume j is to the right: x_j - x_i >= (s_i+s_j)/2
    #                     -> -x_j + x_i + (s_i+s_j)/2 <= 0
    # sep_matrix[i,j] = 1: vertical, assume j is above: y_j - y_i >= (s_i+s_j)/2
    #                     -> -y_j + y_i + (s_i+s_j)/2 <= 0
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(nvar)
            if sep_matrix[i, j] == 0:
                # Horizontal: x_j - x_i >= (s_i+s_j)/2
                row[3*i] = -1.0    # -x_i
                row[3*j] = 1.0     # +x_j
                row[3*i+2] = 0.5   # +s_i/2
                row[3*j+2] = 0.5   # +s_j/2
            else:
                # Vertical: y_j - y_i >= (s_i+s_j)/2
                row[3*i+1] = -1.0  # -y_i
                row[3*j+1] = 1.0   # +y_j
                row[3*i+2] = 0.5   # +s_i/2
                row[3*j+2] = 0.5   # +s_j/2
            A_rows.append(row)
            b_vals.append(0.0)
    
    if not A_rows:
        return None, None
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: x_i in [0,1], y_i in [0,1], s_i in [0.001, 1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, 1.0))  # y_i
        bounds.append((0.001, 1.0))  # s_i
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x = result.x
            centers = np.zeros((n, 2))
            sides = np.zeros(n)
            for i in range(n):
                centers[i, 0] = x[3*i]
                centers[i, 1] = x[3*i+1]
                sides[i] = x[3*i+2]
            return centers, sides
        else:
            return None, None
    except Exception:
        return None, None


def _generate_separation_patterns(n=10, n_patterns=20, seed=42):
    """
    Generate diverse separation patterns.
    Pattern is a 10x10 matrix where sep[i,j] in {0,1} for i<j.
    0 = horizontal separation, 1 = vertical separation.
    """
    rng = np.random.RandomState(seed)
    patterns = []
    
    # Pattern 1: All horizontal (grid-like)
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            p[i, j] = 0
    patterns.append(p)
    
    # Pattern 2: All vertical
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            p[i, j] = 1
    patterns.append(p)
    
    # Pattern 3: Row-based (first 5 horizontal among themselves, last 5 horizontal, vertical between groups)
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if i < 5 and j < 5:
                p[i, j] = 0
            elif i >= 5 and j >= 5:
                p[i, j] = 0
            else:
                p[i, j] = 1
    patterns.append(p)
    
    # Pattern 4: Column-based
    p = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if i % 2 == j % 2:
                p[i, j] = 1
            else:
                p[i, j] = 0
    patterns.append(p)
    
    # Random patterns
    for _ in range(n_patterns - len(patterns)):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                p[i, j] = rng.randint(0, 2)
        patterns.append(p)
    
    return patterns


def _optimize_with_lp(centers_init, sides_init, tol=1e-7, time_budget=30):
    """Try multiple separation patterns and solve LPs."""
    start = time.time()
    best_centers, best_sides = centers_init.copy(), sides_init.copy()
    best_val = np.sum(sides_init)
    
    patterns = _generate_separation_patterns(n=10, n_patterns=30, seed=42)
    
    for pattern in patterns:
        if time.time() - start > time_budget:
            break
        
        c, s = _solve_lp_with_separations(pattern, tol)
        if c is not None and s is not None:
            if _is_feasible(c, s, tol):
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_centers = c
                    best_sides = s
    
    return best_centers, best_sides


def _local_optimize(centers, sides, max_iter=5000, tol=1e-7):
    """Coordinate descent local optimization (kept as fallback)."""
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _sum_sides(sides) - _penalty(_flatten(centers, sides), tol)
    
    step_size = 0.01
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            for delta in [step_size, -step_size]:
                new_cx = centers[i, 0] + delta
                new_centers = centers.copy()
                new_centers[i, 0] = new_cx
                new_obj = _sum_sides(sides) - _penalty(_flatten(new_centers, sides), tol)
                if new_obj > best_obj and _is_feasible(new_centers, sides, tol):
                    best_centers = new_centers
                    best_sides = sides.copy()
                    best_obj = new_obj
                    centers = new_centers
                    improved = True
            for delta in [step_size, -step_size]:
                new_cy = centers[i, 1] + delta
                new_centers = centers.copy()
                new_centers[i, 1] = new_cy
                new_obj = _sum_sides(sides) - _penalty(_flatten(new_centers, sides), tol)
                if new_obj > best_obj and _is_feasible(new_centers, sides, tol):
                    best_centers = new_centers
                    best_sides = sides.copy()
                    best_obj = new_obj
                    centers = new_centers
                    improved = True
            for delta in [step_size, -step_size]:
                new_s = sides[i] + delta
                if new_s <= 0:
                    continue
                new_sides = sides.copy()
                new_sides[i] = new_s
                new_obj = _sum_sides(new_sides) - _penalty(_flatten(centers, new_sides), tol)
                if new_obj > best_obj and _is_feasible(centers, new_sides, tol):
                    best_centers = centers.copy()
                    best_sides = new_sides
                    best_obj = new_obj
                    sides = new_sides
                    improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_centers, best_sides


def _optimize_from_config(centers, sides, tol=1e-7, time_budget=60):
    """Try LP-based separation optimization and local polish."""
    start = time.time()
    
    # Try LP with separation patterns
    if time_budget > 10:
        lp_budget = min(time_budget * 0.6, 40)
        c_lp, s_lp = _optimize_with_lp(centers, sides, tol, time_budget=lp_budget)
        if _is_feasible(c_lp, s_lp, tol):
            val_lp = np.sum(s_lp)
            val_cur = np.sum(sides)
            if val_lp > val_cur:
                centers, sides = c_lp, s_lp
    
    # Final local polish
    if time.time() - start < time_budget:
        centers, sides = _local_optimize(centers, sides, max_iter=2000, tol=tol)
    
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize the sum of side lengths.
    Uses LP-based separation pattern enumeration and local polish.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    best_centers, best_sides = None, None
    best_value = 0.0
    
    # Try multiple initial configurations
    initial_configs = [
        _make_2big_8small(),
        _make_3big_7small(),
        _make_grid_fallback(),
    ]
    
    # Add some random perturbations of the best configs
    np.random.seed(42)
    for _ in range(3):
        c, s = _make_2big_8small()
        c += np.random.normal(0, 0.02, c.shape)
        s += np.random.normal(0, 0.01, s.shape)
        s = np.clip(s, 0.05, 1.0)
        initial_configs.append((c, s))
    
    for idx, (centers, sides) in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Verify initial feasibility
        if not _is_feasible(centers, sides):
            continue
        
        current_val = np.sum(sides)
        
        # Allocate time budget based on remaining time
        remaining = deadline - time.time()
        time_budget = min(50, remaining * 0.3)
        
        if time_budget < 5:
            break
        
        # Optimize from this configuration
        opt_centers, opt_sides = _optimize_from_config(
            centers, sides, tol=1e-7, time_budget=time_budget
        )
        
        if _is_feasible(opt_centers, opt_sides):
            opt_val = np.sum(opt_sides)
            if opt_val > best_value:
                best_value = opt_val
                best_centers = opt_centers
                best_sides = opt_sides
    
    # If no feasible result found, use grid fallback
    if best_centers is None:
        best_centers, best_sides = _make_grid_fallback()
    
    # Final check
    if not _is_feasible(best_centers, best_sides):
        best_centers, best_sides = _make_grid_fallback()
    
    return best_centers, best_sides
