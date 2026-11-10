# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600  # Safety margin before 630s kill
    
    # Try multiple approaches and keep the best valid result
    candidates = []
    
    # Approach 1: Structured layouts with SLSQP optimization
    if time.time() - start_time < deadline - 60:
        layouts = _generate_structured_layouts(n, tol)
        for idx, (c0, s0, sep_dirs) in enumerate(layouts):
            if time.time() - start_time > deadline - 60:
                break
            try:
                c_opt, s_opt = _optimize_layout_slsqp(c0, s0, sep_dirs, n, tol, time_limit=25)
                if c_opt is not None and _check_valid(c_opt, s_opt, tol):
                    candidates.append((c_opt, s_opt, np.sum(s_opt)))
            except Exception as e:
                pass
    
    # Approach 2: Heuristic + local search
    if time.time() - start_time < deadline - 30:
        c_h, s_h = _heuristic_init()
        if _check_valid(c_h, s_h, tol):
            candidates.append((c_h, s_h, np.sum(s_h)))
    
    # Local polish on the best candidate
    if candidates:
        candidates.sort(key=lambda x: -x[2])
        best_c, best_s, best_val = candidates[0]
        
        # Aggressive local improvement
        if time.time() - start_time < deadline - 15:
            best_c, best_s, best_val = _local_improve_aggressive(
                best_c, best_s, tol, time_limit=int(deadline - time.time() - 15)
            )
        
        # Final high-precision polish
        if time.time() - start_time < deadline - 5:
            best_c, best_s, best_val = _final_polish(
                best_c, best_s, tol, time_limit=int(deadline - time.time() - 5)
            )
        
        return best_c, best_s
    
    # Fallback: always return something valid
    c_f, s_f = _heuristic_init()
    return c_f, s_f


def _generate_structured_layouts(n, tol):
    """Generate several structured layout topologies."""
    layouts = []
    
    # Layout 1: 2 large squares on bottom, 8 small in 2x4 grid on top
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # Large squares
    s[0] = 0.45
    s[1] = 0.45
    c[0] = [0.225, 0.225]
    c[1] = [0.775, 0.225]
    # Small squares in 2 rows x 4 cols
    small_s = 0.18
    for k in range(8):
        col = k % 4
        row = k // 4
        cx = (col + 0.5) * 0.25
        cy = 0.55 + row * 0.22 + 0.11
        c[2+k] = [cx, cy]
        s[2+k] = small_s
    # Separation directions: 0=i left of j, 1=j left of i, 2=i below j, 3=j below i
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 2: 3 columns of squares
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 3 columns, each with 3-4 squares
    col_widths = [0.33, 0.34, 0.33]
    x_starts = [0, 0.33, 0.67]
    # Column 0: 4 squares
    for k in range(4):
        c[k] = [0.165, (k+0.5)*0.25]
        s[k] = 0.24
    # Column 1: 3 squares
    for k in range(3):
        c[4+k] = [0.5, (k+0.5)*0.333]
        s[4+k] = 0.30
    # Column 2: 3 squares
    for k in range(3):
        c[7+k] = [0.835, (k+0.5)*0.333]
        s[7+k] = 0.30
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 3: 4 large + 6 small
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 4 large in corners
    large_s = 0.35
    c[0] = [0.175, 0.175]
    c[1] = [0.825, 0.175]
    c[2] = [0.175, 0.825]
    c[3] = [0.825, 0.825]
    s[:4] = large_s
    # 6 small in center
    small_s = 0.15
    positions = [
        [0.35, 0.35], [0.5, 0.35], [0.65, 0.35],
        [0.35, 0.65], [0.5, 0.65], [0.65, 0.65]
    ]
    for k in range(6):
        c[4+k] = positions[k]
        s[4+k] = small_s
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 4: 5 large + 5 small
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 5 large
    large_s = 0.30
    c[0] = [0.15, 0.15]
    c[1] = [0.5, 0.15]
    c[2] = [0.85, 0.15]
    c[3] = [0.15, 0.5]
    c[4] = [0.85, 0.5]
    s[:5] = large_s
    # 5 small
    small_s = 0.18
    c[5] = [0.5, 0.5]
    c[6] = [0.3, 0.8]
    c[7] = [0.7, 0.8]
    c[8] = [0.5, 0.85]
    c[9] = [0.15, 0.85]
    s[5:] = small_s
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 5: 3 rows of 3-3-4
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # Row 0 (bottom): 4 squares
    for k in range(4):
        c[k] = [(k+0.5)*0.25, 0.125]
        s[k] = 0.22
    # Row 1 (middle): 3 squares
    for k in range(3):
        c[4+k] = [(k+0.5)*0.333, 0.5]
        s[4+k] = 0.30
    # Row 2 (top): 3 squares
    for k in range(3):
        c[7+k] = [(k+0.5)*0.333, 0.875]
        s[7+k] = 0.22
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 6: Mixed sizes - 2 very large, 3 medium, 5 small
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 2 very large
    s[0] = 0.40
    s[1] = 0.40
    c[0] = [0.20, 0.20]
    c[1] = [0.80, 0.20]
    # 3 medium
    s[2] = 0.25
    s[3] = 0.25
    s[4] = 0.25
    c[2] = [0.20, 0.65]
    c[3] = [0.50, 0.65]
    c[4] = [0.80, 0.65]
    # 5 small
    small_s = 0.15
    positions = [
        [0.10, 0.90], [0.30, 0.90], [0.50, 0.90], [0.70, 0.90], [0.90, 0.90]
    ]
    for k in range(5):
        c[5+k] = positions[k]
        s[5+k] = small_s
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    # Layout 7: Grid-like with varying sizes
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 2x5 grid
    for k in range(10):
        col = k % 5
        row = k // 5
        cx = (col + 0.5) * 0.20
        cy = (row + 0.5) * 0.50
        c[k] = [cx, cy]
        s[k] = 0.18
    sep_dirs = {}
    for i in range(n):
        for j in range(i+1, n):
            if abs(c[i,0] - c[j,0]) > abs(c[i,1] - c[j,1]):
                sep_dirs[(i,j)] = 0 if c[i,0] < c[j,0] else 1
            else:
                sep_dirs[(i,j)] = 2 if c[i,1] < c[j,1] else 3
    layouts.append((c.copy(), s.copy(), sep_dirs))
    
    return layouts


def _optimize_layout_slsqp(c0, s0, sep_dirs, n, tol, time_limit=25):
    """Optimize a layout with fixed separation directions using SLSQP."""
    # Variables: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = c0[i, 0]
        x0[3*i+1] = c0[i, 1]
        x0[3*i+2] = s0[i]
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 1))
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*x[3*i] - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*(1 - x[3*i]) - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*x[3*i+1] - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*(1 - x[3*i+1]) - x[3*i+2] - tol})
    
    # Separation constraints based on sep_dirs
    for (i, j), direction in sep_dirs.items():
        if direction == 0:  # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*j] - x[3*i]) - 0.5*(x[3*i+2] + x[3*j+2]) - tol})
        elif direction == 1:  # j left of i: cx_j + s_j/2 <= cx_i - s_i/2
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*i] - x[3*j]) - 0.5*(x[3*i+2] + x[3*j+2]) - tol})
        elif direction == 2:  # i below j: cy_i + s_i/2 <= cy_j - s_j/2
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*j+1] - x[3*i+1]) - 0.5*(x[3*i+2] + x[3*j+2]) - tol})
        elif direction == 3:  # j below i: cy_j + s_j/2 <= cy_i - s_i/2
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                (x[3*i+1] - x[3*j+1]) - 0.5*(x[3*i+2] + x[3*j+2]) - tol})
    
    def objective(x):
        return -np.sum(x[2::3])
    
    start_time = time.time()
    try:
        result = minimize(
            objective, x0, method='SLSQP', bounds=bounds,
            constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if time.time() - start_time > time_limit:
            return None, None
        
        if result.fun is not None and np.isfinite(result.fun):
            x_opt = result.x
            centers = np.column_stack([x_opt[0::3], x_opt[1::3]])
            sides = x_opt[2::3]
            return centers, sides
    except Exception:
        pass
    
    return None, None


def _check_valid(centers, sides, tol=1e-7):
    """Check if configuration is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_sep = (sides[i] + sides[j])/2 - tol
            if dx < min_sep and dy < min_sep:
                return False
    return True


def _heuristic_init():
    """Generate a valid heuristic initial configuration."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 2 large squares
    centers[0] = [0.25, 0.25]
    sides[0] = 0.45
    centers[1] = [0.75, 0.25]
    sides[1] = 0.45
    
    # 8 small squares in 2x4 grid
    for k in range(8):
        col = k % 4
        row = k // 4
        cx = (col + 0.5) * 0.25
        cy = 0.55 + row * 0.22 + 0.11
        centers[2+k, 0] = cx
        centers[2+k, 1] = cy
        sides[2+k] = 0.18
    
    return centers, sides


def _local_improve_aggressive(centers, sides, tol=1e-7, time_limit=60):
    """Aggressive local improvement: try to grow each square, then fine-tune positions."""
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if not _check_valid(best_centers, best_sides, tol):
        return best_centers, best_sides, best_val
    
    start_time = time.time()
    
    # Phase 1: Try to grow each square as much as possible
    for round_num in range(50):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            # Try to increase side length
            max_s = 1.0
            # Binary search for max feasible side
            lo, hi = best_sides[i], max_s
            for _ in range(20):
                mid = (lo + hi) / 2
                if _can_grow_square(best_centers, best_sides, i, mid, tol):
                    lo = mid
                else:
                    hi = mid
            
            if lo > best_sides[i] + 1e-8:
                best_sides[i] = lo
                best_val = np.sum(best_sides)
                improved = True
        
        if not improved:
            break
    
    # Phase 2: Fine-tune positions with small perturbations
    rng = np.random.default_rng(42)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step in step_sizes:
        if time.time() - start_time > time_limit:
            break
        
        for it in range(200):
            if time.time() - start_time > time_limit:
                break
            
            idx = rng.integers(0, n)
            nc = best_centers.copy()
            ns = best_sides.copy()
            
            nc[idx, 0] += rng.uniform(-step, step)
            nc[idx, 1] += rng.uniform(-step, step)
            
            # Clip to valid range
            s = ns[idx]
            nc[idx, 0] = np.clip(nc[idx, 0], s/2, 1-s/2)
            nc[idx, 1] = np.clip(nc[idx, 1], s/2, 1-s/2)
            
            # Check validity
            valid = True
            for j in range(n):
                if j == idx:
                    continue
                dx = abs(nc[idx, 0] - best_centers[j, 0])
                dy = abs(nc[idx, 1] - best_centers[j, 1])
                min_sep = (ns[idx] + best_sides[j])/2 - tol
                if dx < min_sep and dy < min_sep:
                    valid = False
                    break
            
            if valid:
                best_centers = nc
                best_val = np.sum(best_sides)
    
    return best_centers, best_sides, best_val


def _final_polish(centers, sides, tol=1e-7, time_limit=10):
    """High-precision final polish using iterative improvement."""
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if not _check_valid(best_centers, best_sides, tol):
        return best_centers, best_sides, best_val
    
    start_time = time.time()
    
    # Iteratively try to grow each square with very fine binary search
    for round_num in range(20):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            max_s = 1.0
            lo, hi = best_sides[i], max_s
            for _ in range(30):
                mid = (lo + hi) / 2
                if _can_grow_square(best_centers, best_sides, i, mid, tol):
                    lo = mid
                else:
                    hi = mid
            
            if lo > best_sides[i] + 1e-10:
                best_sides[i] = lo
                best_val = np.sum(best_sides)
                improved = True
        
        if not improved:
            break
    
    # Very fine position adjustments
    rng = np.random.default_rng(123)
    for it in range(500):
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.integers(0, n)
        nc = best_centers.copy()
        ns = best_sides.copy()
        
        step = 0.0005
        nc[idx, 0] += rng.uniform(-step, step)
        nc[idx, 1] += rng.uniform(-step, step)
        
        s = ns[idx]
        nc[idx, 0] = np.clip(nc[idx, 0], s/2, 1-s/2)
        nc[idx, 1] = np.clip(nc[idx, 1], s/2, 1-s/2)
        
        valid = True
        for j in range(n):
            if j == idx:
                continue
            dx = abs(nc[idx, 0] - best_centers[j, 0])
            dy = abs(nc[idx, 1] - best_centers[j, 1])
            min_sep = (ns[idx] + best_sides[j])/2 - tol
            if dx < min_sep and dy < min_sep:
                valid = False
                break
        
        if valid:
            # Try to grow after moving
            for _ in range(10):
                max_s = 1.0
                lo, hi = ns[idx], max_s
                for _ in range(15):
                    mid = (lo + hi) / 2
                    if _can_grow_square(nc, ns, idx, mid, tol):
                        lo = mid
                    else:
                        hi = mid
                if lo > ns[idx] + 1e-10:
                    ns[idx] = lo
                    break
            
            if _check_valid(nc, ns, tol):
                best_centers = nc
                best_sides = ns
                best_val = np.sum(best_sides)
    
    return best_centers, best_sides, best_val


def _can_grow_square(centers, sides, idx, new_s, tol):
    """Check if we can grow square idx to new_s without violating constraints."""
    n = len(sides)
    cx = centers[idx, 0]
    cy = centers[idx, 1]
    
    # Check boundary
    if cx - new_s/2 < -tol or cx + new_s/2 > 1 + tol:
        return False
    if cy - new_s/2 < -tol or cy + new_s/2 > 1 + tol:
        return False
    
    # Check overlap with others
    for j in range(n):
        if j == idx:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        min_sep = (new_s + sides[j])/2 - tol
        if dx < min_sep and dy < min_sep:
            return False
    
    return True
# EVOLVE-BLOCK-END
