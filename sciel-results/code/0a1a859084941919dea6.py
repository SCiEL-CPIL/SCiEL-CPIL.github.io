import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _check_inside(points):
    """Check if all points are inside or on the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    eps = 1e-8
    return (np.all(x >= -eps) and np.all(x <= 1.0 + eps) and
            np.all(y >= -eps) and np.all(y <= H + eps) and
            np.all(y <= SQRT3 * x + eps) and
            np.all(y <= -SQRT3 * x + SQRT3 + eps))

def _project_to_triangle(points):
    """Project points to be inside the triangle."""
    result = points.copy()
    x = result[:, 0]
    y = result[:, 1]
    
    # Clip to x range
    x = np.clip(x, 0.0, 1.0)
    # Clip y to be below both slanted edges
    y = np.minimum(y, SQRT3 * x)
    y = np.minimum(y, -SQRT3 * x + SQRT3)
    y = np.clip(y, 0.0, H)
    
    result[:, 0] = x
    result[:, 1] = y
    return result

def _generate_grid_points(res):
    """Generate a triangular grid of points inside the triangle."""
    points = []
    for i in range(res + 1):
        for j in range(res + 1 - i):
            # Barycentric coordinates
            u = i / res
            v = j / res
            w = 1.0 - u - v
            # Convert to Cartesian
            x = v * 1.0 + w * 0.5
            y = w * H
            points.append([x, y])
    return np.array(points)

def _greedy_select(grid_points, num_select=N):
    """Greedy selection of points from grid to maximize minimum area."""
    if len(grid_points) < num_select:
        return grid_points
    
    selected = [0]
    remaining = list(range(1, len(grid_points)))
    
    while len(selected) < num_select:
        best_idx = -1
        best_score = -1e10
        for idx in remaining:
            candidate = selected + [idx]
            pts = grid_points[candidate]
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return grid_points[selected]

def _local_search(points, max_iter=500, step=0.02):
    """Local search by nudging points slightly."""
    current = points.copy()
    current_score = _min_area(current)
    best = current.copy()
    best_score = current_score
    rng = np.random.default_rng(42)
    
    for _ in range(max_iter):
        # Pick a random point and nudge it
        idx = rng.integers(N)
        dx = rng.uniform(-step, step)
        dy = rng.uniform(-step, step)
        
        new_points = current.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        new_points = _project_to_triangle(new_points)
        
        if not _check_inside(new_points):
            continue
        
        new_score = _min_area(new_points)
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best

def _coarse_to_fine(deadline):
    """Multi-resolution search: start with coarse grid, then refine."""
    start_time = time.time()
    
    # Phase 1: Coarse grid search
    best_points = None
    best_score = -1e10
    
    for res in [3, 4, 5, 6]:
        if time.time() > deadline - 30:
            break
        
        grid = _generate_grid_points(res)
        if len(grid) < N:
            continue
        
        selected_idx = _greedy_select(grid)
        pts = grid[selected_idx]
        score = _min_area(pts)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # Local refinement
        refined = _local_search(pts, max_iter=200, step=0.05)
        refined_score = _min_area(refined)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined.copy()
    
    # Phase 2: Fine grid search around the best solution
    if best_points is not None and time.time() < deadline - 10:
        # Generate fine grid in a neighborhood
        for scale in [0.1, 0.05]:
            if time.time() > deadline - 5:
                break
            
            # Create fine grid around current best
            fine_points = []
            for dx in np.linspace(-scale, scale, 5):
                for dy in np.linspace(-scale, scale, 5):
                    candidate = best_points.copy()
                    candidate[0, 0] += dx
                    candidate[0, 1] += dy
                    candidate = _project_to_triangle(candidate)
                    if _check_inside(candidate):
                        fine_points.append(candidate)
            
            if len(fine_points) > N:
                for candidate in fine_points:
                    score = _min_area(candidate)
                    if score > best_score:
                        best_score = score
                        best_points = candidate.copy()
        
        # Final local refinement
        best_points = _local_search(best_points, max_iter=300, step=0.01)
    
    return best_points

def _differential_evolution_search(deadline):
    """Differential evolution search over barycentric coordinates."""
    start_time = time.time()
    
    def _bary_to_cart(bary):
        u = np.clip(bary[:, 0], 0.0, None)
        v = np.clip(bary[:, 1], 0.0, None)
        s = u + v
        mask = s > 1.0
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
        
        pts = np.empty((N, 2))
        pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
        pts[:, 1] = (1.0 - u - v) * H
        return pts
    
    def _objective(params):
        bary = params.reshape(N, 2)
        pts = _bary_to_cart(bary)
        if not _check_inside(pts):
            return 1e10
        return -_min_area(pts)
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    from scipy.optimize import differential_evolution
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=100,
        tol=1e-6,
        seed=42,
        polish=True,
        popsize=15
    )
    
    if result.success or result.fun < 1e9:
        pts = _bary_to_cart(result.x.reshape(N, 2))
        if _check_inside(pts):
            return pts
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = -1e10
    
    # Strategy 1: Coarse-to-fine grid search
    try:
        pts = _coarse_to_fine(deadline)
        if pts is not None:
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    except Exception:
        pass
    
    # Strategy 2: Differential evolution if time allows
    if time.time() < deadline - 20:
        try:
            pts = _differential_evolution_search(deadline)
            if pts is not None:
                score = _min_area(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass
    
    # Strategy 3: Multiple random restarts with local search
    if best_points is None or time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 5:
                break
            
            # Random valid points in barycentric coordinates
            bary = np.empty((N, 2))
            for m in range(N):
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v <= 1.0:
                        bary[m, 0] = u
                        bary[m, 1] = v
                        break
            
            u = bary[:, 0]
            v = bary[:, 1]
            pts = np.empty((N, 2))
            pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
            pts[:, 1] = (1.0 - u - v) * H
            
            if _check_inside(pts):
                refined = _local_search(pts, max_iter=100, step=0.03)
                score = _min_area(refined)
                if score > best_score:
                    best_score = score
                    best_points = refined
    
    # Fallback: return a valid configuration
    if best_points is None:
        rng = np.random.default_rng(42)
        while True:
            pts = np.empty((N, 2))
            for m in range(N):
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v <= 1.0:
                        x = v * 1.0 + (1.0 - u - v) * 0.5
                        y = (1.0 - u - v) * H
                        break
                pts[m, 0] = x
                pts[m, 1] = y
            
            if _check_inside(pts):
                best_points = pts
                break
    
    return best_points
