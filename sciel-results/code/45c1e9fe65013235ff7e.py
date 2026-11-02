import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute indices for all 165 triangles
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

def _min_area_and_indices(points):
    """Return (min_area, list of indices of points involved in min triangles)."""
    areas = _compute_areas(points)
    min_area = float(np.min(areas))
    threshold = min_area + 1e-10
    min_indices = np.where(areas <= threshold)[0]
    involved = set()
    for idx in min_indices:
        involved.add(int(_idx_arr[idx, 0]))
        involved.add(int(_idx_arr[idx, 1]))
        involved.add(int(_idx_arr[idx, 2]))
    return min_area, sorted(involved)

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points, ensuring validity."""
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

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    eps = 1e-10
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < -eps:
            points[m, 1] = 0.0
        if y + SQRT3 * x > SQRT3 + eps:
            d = (y + SQRT3 * x - SQRT3) / 2.0
            if d > 0:
                points[m, 0] -= SQRT3 * d
                points[m, 1] -= d
        if y > SQRT3 * x + eps:
            d = (SQRT3 * x - y) / 2.0
            if d < 0:
                points[m, 0] += SQRT3 * d
                points[m, 1] -= d
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        if y < 0:
            points[m, 1] = 0
        if y > SQRT3 * x:
            if x > 0:
                points[m, 1] = SQRT3 * x
            else:
                points[m, 1] = 0
        if y > SQRT3 * (1 - x):
            if x < 1:
                points[m, 1] = SQRT3 * (1 - x)
            else:
                points[m, 1] = 0
    return points

def _sobol_points_in_triangle(n, seed=42):
    """Generate n quasi-random points in the equilateral triangle using Sobol sequence."""
    from scipy.stats import qmc
    rng = qmc.Sobol(d=2, seed=seed)
    n_gen = n * 4
    samples = rng.random(n_gen)
    u = samples[:, 0]
    v = samples[:, 1]
    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    valid_u = u[mask]
    valid_v = v[mask]
    if len(valid_u) >= n:
        u = valid_u[:n]
        v = valid_v[:n]
    else:
        while len(valid_u) < n:
            more = rng.random(n_gen)
            u2, v2 = more[:, 0], more[:, 1]
            mask2 = (u2 >= 0) & (v2 >= 0) & (u2 + v2 <= 1)
            valid_u = np.concatenate([valid_u, u2[mask2]])
            valid_v = np.concatenate([valid_v, v2[mask2]])
        u = valid_u[:n]
        v = valid_v[:n]
    bary = np.column_stack([u, v])
    return _bary_to_cart(bary)

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta

def _objective(params, beta=50.0):
    """
    Objective function for optimization: negative of soft min area.
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points.copy())
    return -_soft_min_area(points, beta)

def _nelder_mead_optimize(points, time_limit=10.0, beta=50.0):
    """
    Use Nelder-Mead to optimize the soft min area objective.
    """
    start_t = time.time()
    x0 = points.flatten()
    
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        args=(beta,),
        options={
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        return points, _min_area(points)
    
    new_points = result.x.reshape(N, 2)
    new_points = _project_to_triangle(new_points)
    score = _min_area(new_points)
    
    if score > _min_area(points):
        return new_points, score
    else:
        return points, _min_area(points)

def _local_nudge_search(points, max_iter=500, step_sizes=None, time_limit=None):
    """
    Local search that only moves points involved in the smallest triangles.
    """
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min, _ = _min_area_and_indices(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    if time_limit is None:
        time_limit = 60.0
    start_t = time.time()
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            min_area, involved = _min_area_and_indices(current_points)
            
            if min_area > best_min:
                best_min = min_area
                best_points = current_points.copy()
            
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                
                for _ in range(20):
                    du = np.random.uniform(-step_size, step_size)
                    dv = np.random.uniform(-step_size, step_size)
                    
                    bary = _cart_to_bary(current_points)
                    new_u = bary[idx, 0] + du
                    new_v = bary[idx, 1] + dv
                    
                    new_u = max(0.0, new_u)
                    new_v = max(0.0, new_v)
                    s = new_u + new_v
                    if s > 1.0:
                        new_u /= s
                        new_v /= s
                    
                    new_x = new_v * 1.0 + (1.0 - new_u - new_v) * 0.5
                    new_y = (1.0 - new_u - new_v) * H
                    
                    if new_y < -1e-10 or new_y > SQRT3 * new_x + 1e-10 or new_y > SQRT3 * (1 - new_x) + 1e-10:
                        continue
                    
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_points, best_min

def _initial_guess_grid():
    """Generate a structured grid-like initial configuration."""
    positions = []
    for i in range(3):
        x = 0.1 + i * 0.3
        positions.append((x, 0.05))
    for i in range(4):
        x = 0.05 + i * 0.25
        y = 0.25
        if y <= SQRT3 * x and y <= SQRT3 * (1 - x):
            positions.append((x, y))
    for i in range(3):
        x = 0.15 + i * 0.25
        y = 0.45
        if y <= SQRT3 * x and y <= SQRT3 * (1 - x):
            positions.append((x, y))
    positions.append((0.5, 0.6))
    
    valid_positions = []
    for p in positions:
        x, y = p
        if y >= -1e-10 and y <= SQRT3 * x + 1e-10 and y <= SQRT3 * (1 - x) + 1e-10:
            valid_positions.append(p)
    
    if len(valid_positions) >= N:
        return np.array(valid_positions[:N])
    else:
        return _sobol_points_in_triangle(N, seed=0)

def _de_optimize(points, time_limit=20.0):
    """
    Use Differential Evolution in barycentric space to optimize the configuration.
    """
    start_t = time.time()
    
    # Convert to barycentric (u, v) where a = 1 - u - v
    bary = _cart_to_bary(points)
    x0 = bary.flatten()
    
    def _de_objective(flat_bary):
        """DE objective: minimize negative min area with constraint penalty."""
        bary = flat_bary.reshape(N, 2)
        u = bary[:, 0]
        v = bary[:, 1]
        
        # Check constraints: u >= 0, v >= 0, u + v <= 1
        penalty = 0.0
        for i in range(N):
            if u[i] < 0:
                penalty += u[i] ** 2 * 1000
            if v[i] < 0:
                penalty += v[i] ** 2 * 1000
            if u[i] + v[i] > 1:
                penalty += (u[i] + v[i] - 1) ** 2 * 1000
        
        if penalty > 1e-6:
            return penalty
        
        pts = _bary_to_cart(bary)
        pts = _project_to_triangle(pts)
        return -_min_area(pts)
    
    bounds = [(0, 1), (0, 1)] * N
    
    try:
        result = differential_evolution(
            _de_objective,
            bounds,
            maxiter=30,
            popsize=10,
            seed=42,
            tol=1e-5,
            polish=False,
            init='latinhypercube'
        )
        
        new_bary = result.x.reshape(N, 2)
        new_points = _bary_to_cart(new_bary)
        new_points = _project_to_triangle(new_points)
        score = _min_area(new_points)
        
        if score > _min_area(points):
            return new_points, score
        else:
            return points, _min_area(points)
            
    except Exception:
        return points, _min_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multiple diverse initializations
    initializations = []
    
    # Sobol-based initializations
    for seed in range(5):
        init_points = _sobol_points_in_triangle(N, seed=seed)
        init_points = _project_to_triangle(init_points)
        initializations.append(init_points)
    
    # Grid-based initialization
    grid_points = _initial_guess_grid()
    grid_points = _project_to_triangle(grid_points)
    initializations.append(grid_points)
    
    # Random initialization
    for seed in range(3):
        np.random.seed(seed + 100)
        bary = np.random.dirichlet([1, 1, 1], size=N)
        u = bary[:, 0]
        v = bary[:, 1]
        init_points = _bary_to_cart(np.column_stack([u, v]))
        init_points = _project_to_triangle(init_points)
        initializations.append(init_points)
    
    # Symmetric initial configuration
    symmetric = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, 0.0],
        [0.75, H / 2],
        [0.25, H / 2],
        [1/3, H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, H/3],
        [0.2, 0.1],
    ], dtype=float)
    symmetric = _project_to_triangle(symmetric)
    initializations.append(symmetric)
    
    # Optimize each initialization
    for i, init_points in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        # First, try Nelder-Mead with soft min
        nm_points, nm_score = _nelder_mead_optimize(
            init_points, 
            time_limit=min(12.0, deadline - time.time() - 15),
            beta=50.0
        )
        
        # Then, try Differential Evolution if time permits
        if time.time() < deadline - 20:
            de_points, de_score = _de_optimize(
                nm_points,
                time_limit=min(15.0, deadline - time.time() - 10)
            )
            if de_score > nm_score:
                nm_points, nm_score = de_points, de_score
        
        # Finally, local nudge search
        refined_points, refined_score = _local_nudge_search(
            nm_points,
            max_iter=300,
            time_limit=min(8.0, deadline - time.time() - 5)
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # Phase 2: If we have time, try more aggressive optimization on best result
    if best_points is not None and time.time() < deadline - 20:
        # Increase beta for sharper minimum approximation
        nm_points, nm_score = _nelder_mead_optimize(
            best_points,
            time_limit=min(12.0, deadline - time.time() - 12),
            beta=100.0
        )
        
        # Try DE again
        if time.time() < deadline - 15:
            de_points, de_score = _de_optimize(
                nm_points,
                time_limit=min(12.0, deadline - time.time() - 8)
            )
            if de_score > nm_score:
                nm_points, nm_score = de_points, de_score
        
        refined_points, refined_score = _local_nudge_search(
            nm_points,
            max_iter=500,
            step_sizes=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
            time_limit=min(8.0, deadline - time.time() - 3)
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # Fallback if no valid result found
    if best_points is None:
        best_points = _sobol_points_in_triangle(N, seed=0)
        best_points = _project_to_triangle(best_points)
    
    # Final projection to ensure validity
    best_points = _project_to_triangle(best_points)
    
    return best_points
