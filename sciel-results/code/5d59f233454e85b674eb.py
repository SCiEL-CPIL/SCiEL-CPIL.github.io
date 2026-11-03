import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools
from scipy.stats import qmc
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)
N_TRI = len(_indices)

# Precompute: for each point m, which triangle indices involve it
_point_tri_idx = [[] for _ in range(N)]
for tri_i, (a, b, c) in enumerate(_indices):
    _point_tri_idx[a].append(tri_i)
    _point_tri_idx[b].append(tri_i)
    _point_tri_idx[c].append(tri_i)
_point_tri_idx = [np.array(lst) for lst in _point_tri_idx]

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

def _delaunay_triangles(points):
    """Get indices of Delaunay triangles (within the point set)."""
    try:
        tri = Delaunay(points)
        return tri.simplices
    except Exception:
        return _idx_arr

def _compute_areas_involving(points, m, areas):
    """Update only the triangles involving point m. Returns updated areas array."""
    new_areas = areas.copy()
    tri_idxs = _point_tri_idx[m]
    for ti in tri_idxs:
        a, b, c = _idx_arr[ti]
        if a == m:
            det = points[a,0]*(points[b,1]-points[c,1]) + points[b,0]*(points[c,1]-points[a,1]) + points[c,0]*(points[a,1]-points[b,1])
        elif b == m:
            det = points[a,0]*(points[b,1]-points[c,1]) + points[b,0]*(points[c,1]-points[a,1]) + points[c,0]*(points[a,1]-points[b,1])
        else:
            det = points[a,0]*(points[b,1]-points[c,1]) + points[b,0]*(points[c,1]-points[a,1]) + points[c,0]*(points[a,1]-points[b,1])
        new_areas[ti] = 0.5 * abs(det)
    return new_areas

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
    return min_area, sorted(involved), areas

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

def _is_inside_triangle(x, y, tol=1e-9):
    """Check if point (x,y) is inside the equilateral triangle."""
    if y < -tol:
        return False
    if y + SQRT3 * x > SQRT3 + tol:
        return False
    if y > SQRT3 * x + tol:
        return False
    return True

def _sobol_points_in_triangle(n, seed=42):
    """Generate n quasi-random points in the equilateral triangle using Sobol sequence."""
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
    """Smooth approximation of the minimum area using log-sum-exp."""
    areas = _compute_areas(points)
    min_a = np.min(areas)
    return min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta

def _delaunay_objective(params, beta=30.0):
    """
    Objective based on Delaunay triangles only.
    The minimum area triangle in a set is always a Delaunay triangle,
    so we focus optimization on the ~10-15 Delaunay triangles.
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points.copy())
    
    # Get Delaunay triangles
    try:
        tri = Delaunay(points)
        simplices = tri.simplices
    except Exception:
        # Fallback to all triangles
        simplices = _idx_arr
    
    # Compute areas of Delaunay triangles only
    a_idx = simplices[:, 0]
    b_idx = simplices[:, 1]
    c_idx = simplices[:, 2]
    
    xa, ya = points[a_idx, 0], points[a_idx, 1]
    xb, yb = points[b_idx, 0], points[b_idx, 1]
    xc, yc = points[c_idx, 0], points[c_idx, 1]
    
    det = xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb)
    areas = 0.5 * np.abs(det)
    
    # Soft min of Delaunay triangle areas
    min_a = np.min(areas)
    soft_min = min_a - np.log(np.sum(np.exp(-beta * (areas - min_a)))) / beta
    
    # Also verify against all triangles occasionally (cheap check)
    all_areas = _compute_areas(points)
    true_min = float(np.min(all_areas))
    
    # Penalize if Delaunay min is much larger than true min (shouldn't happen often)
    penalty = max(0, true_min - soft_min) * 10.0
    
    return -(soft_min - penalty)

def _objective(params, beta=50.0):
    """Objective function for optimization: negative of soft min area."""
    points = params.reshape(N, 2)
    points = _project_to_triangle(points.copy())
    return -_soft_min_area(points, beta)

def _nelder_mead_optimize(points, time_limit=10.0, beta=50.0):
    """Use Nelder-Mead to optimize the soft min area objective."""
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

def _delaunay_nelder_mead(points, time_limit=10.0, beta=30.0):
    """
    Nelder-Mead optimization using Delaunay-based objective.
    This is faster per iteration since it only evaluates ~10-15 triangles.
    """
    start_t = time.time()
    x0 = points.flatten()
    
    result = minimize(
        _delaunay_objective,
        x0,
        method='Nelder-Mead',
        args=(beta,),
        options={
            'maxiter': 3000,
            'xatol': 1e-9,
            'fatol': 1e-11,
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

def _delaunay_local_search(points, max_iter=3000, time_limit=15.0, step_sizes=None):
    """
    Local search that only evaluates Delaunay triangles for speed.
    Periodically verifies against all 165 triangles.
    """
    if step_sizes is None:
        step_sizes = [0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    current_points = points.copy()
    current_areas = _compute_areas(current_points)
    current_min = float(np.min(current_areas))
    best_points = current_points.copy()
    best_min = current_min
    
    start_t = time.time()
    np.random.seed(99999)
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for it in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            # Find points in minimum triangles (from full computation)
            min_a = float(np.min(current_areas))
            threshold = min_a + 1e-9
            min_tris = np.where(current_areas <= threshold)[0]
            involved = set()
            for ti in min_tris:
                involved.add(int(_idx_arr[ti, 0]))
                involved.add(int(_idx_arr[ti, 1]))
                involved.add(int(_idx_arr[ti, 2]))
            
            improved = False
            for m in involved:
                if time.time() - start_t > time_limit:
                    break
                
                # Try several random perturbations of point m
                for attempt in range(20):
                    angle = np.random.uniform(0, 2 * np.pi)
                    step = np.random.uniform(0.3, 1.0) * step_size
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    
                    new_x = current_points[m, 0] + dx
                    new_y = current_points[m, 1] + dy
                    
                    if not _is_inside_triangle(new_x, new_y):
                        new_y = max(0.0, new_y)
                        if new_y > SQRT3 * new_x:
                            new_y = SQRT3 * new_x
                        if new_y > SQRT3 * (1 - new_x):
                            new_y = SQRT3 * (1 - new_x)
                        if new_y < 0 or new_y > SQRT3 * new_x + 1e-9 or new_y > SQRT3 * (1 - new_x) + 1e-9:
                            continue
                    
                    test_points = current_points.copy()
                    test_points[m, 0] = new_x
                    test_points[m, 1] = new_y
                    
                    # Fast check: only recompute triangles involving m
                    new_areas = current_areas.copy()
                    tri_idxs = _point_tri_idx[m]
                    for ti in tri_idxs:
                        a, b, c = _idx_arr[ti]
                        det = test_points[a,0]*(test_points[b,1]-test_points[c,1]) + \
                              test_points[b,0]*(test_points[c,1]-test_points[a,1]) + \
                              test_points[c,0]*(test_points[a,1]-test_points[b,1])
                        new_areas[ti] = 0.5 * abs(det)
                    
                    new_min = float(np.min(new_areas))
                    
                    if new_min > min_a + 1e-12:
                        current_points = test_points
                        current_areas = new_areas
                        improved = True
                        if new_min > best_min:
                            best_min = new_min
                            best_points = current_points.copy()
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_points, best_min

def _fast_local_search(points, max_iter=2000, time_limit=15.0, step_sizes=None):
    """
    Fast local search using incremental area updates.
    Only recomputes triangles involving the moved point.
    """
    if step_sizes is None:
        step_sizes = [0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_areas = _compute_areas(current_points)
    current_min = float(np.min(current_areas))
    best_points = current_points.copy()
    best_min = current_min
    best_areas = current_areas.copy()
    
    start_t = time.time()
    np.random.seed(12345)
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for it in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            # Find points in minimum triangles
            min_a = float(np.min(current_areas))
            threshold = min_a + 1e-9
            min_tris = np.where(current_areas <= threshold)[0]
            involved = set()
            for ti in min_tris:
                involved.add(int(_idx_arr[ti, 0]))
                involved.add(int(_idx_arr[ti, 1]))
                involved.add(int(_idx_arr[ti, 2]))
            
            improved = False
            for m in involved:
                if time.time() - start_t > time_limit:
                    break
                
                # Try several random perturbations of point m
                for attempt in range(15):
                    # Random direction in Cartesian
                    angle = np.random.uniform(0, 2 * np.pi)
                    step = np.random.uniform(0.3, 1.0) * step_size
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    
                    new_x = current_points[m, 0] + dx
                    new_y = current_points[m, 1] + dy
                    
                    if not _is_inside_triangle(new_x, new_y):
                        # Try to project
                        new_y = max(0.0, new_y)
                        if new_y > SQRT3 * new_x:
                            new_y = SQRT3 * new_x
                        if new_y > SQRT3 * (1 - new_x):
                            new_y = SQRT3 * (1 - new_x)
                        if new_y < 0 or new_y > SQRT3 * new_x + 1e-9 or new_y > SQRT3 * (1 - new_x) + 1e-9:
                            continue
                    
                    # Compute new areas for triangles involving m
                    test_points = current_points.copy()
                    test_points[m, 0] = new_x
                    test_points[m, 1] = new_y
                    
                    # Only recompute triangles involving m
                    new_areas = current_areas.copy()
                    tri_idxs = _point_tri_idx[m]
                    for ti in tri_idxs:
                        a, b, c = _idx_arr[ti]
                        det = test_points[a,0]*(test_points[b,1]-test_points[c,1]) + \
                              test_points[b,0]*(test_points[c,1]-test_points[a,1]) + \
                              test_points[c,0]*(test_points[a,1]-test_points[b,1])
                        new_areas[ti] = 0.5 * abs(det)
                    
                    new_min = float(np.min(new_areas))
                    
                    if new_min > min_a + 1e-12:
                        current_points = test_points
                        current_areas = new_areas
                        improved = True
                        if new_min > best_min:
                            best_min = new_min
                            best_points = current_points.copy()
                            best_areas = new_areas.copy()
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_points, best_min

def _annealing_search(points, time_limit=20.0):
    """
    Simulated annealing with push-apart forces on points in minimum triangles.
    Uses Delaunay triangles for fast evaluation.
    """
    current_points = points.copy()
    current_areas = _compute_areas(current_points)
    current_min = float(np.min(current_areas))
    best_points = current_points.copy()
    best_min = current_min
    
    start_t = time.time()
    np.random.seed(777)
    
    T = 0.05  # initial temperature
    cooling = 0.995
    
    while time.time() - start_t < time_limit:
        # Find points in minimum triangles
        min_a = float(np.min(current_areas))
        threshold = min_a + 1e-8
        min_tris = np.where(current_areas <= threshold)[0]
        involved = set()
        for ti in min_tris:
            involved.add(int(_idx_arr[ti, 0]))
            involved.add(int(_idx_arr[ti, 1]))
            involved.add(int(_idx_arr[ti, 2]))
        
        if len(involved) == 0:
            break
        
        # Pick a random involved point
        m = int(np.random.choice(list(involved)))
        
        # Generate perturbation: mix of random and push-away from other involved points
        if np.random.random() < 0.5:
            # Random perturbation
            angle = np.random.uniform(0, 2 * np.pi)
            step = np.random.uniform(0.005, 0.04)
            dx = step * np.cos(angle)
            dy = step * np.sin(angle)
        else:
            # Push away from centroid of other involved points
            others = [p for p in involved if p != m]
            if others:
                cx = np.mean([current_points[p, 0] for p in others])
                cy = np.mean([current_points[p, 1] for p in others])
                dx = current_points[m, 0] - cx
                dy = current_points[m, 1] - cy
                norm = np.sqrt(dx*dx + dy*dy)
                if norm > 1e-10:
                    dx /= norm
                    dy /= norm
                step = np.random.uniform(0.01, 0.05)
                dx *= step
                dy *= step
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                step = np.random.uniform(0.01, 0.04)
                dx = step * np.cos(angle)
                dy = step * np.sin(angle)
        
        new_x = current_points[m, 0] + dx
        new_y = current_points[m, 1] + dy
        
        if not _is_inside_triangle(new_x, new_y):
            new_y = max(0.0, new_y)
            if new_y > SQRT3 * new_x:
                new_y = SQRT3 * new_x
            if new_y > SQRT3 * (1 - new_x):
                new_y = SQRT3 * (1 - new_x)
            if new_y < -1e-9 or new_y > SQRT3 * new_x + 1e-9 or new_y > SQRT3 * (1 - new_x) + 1e-9:
                T *= cooling
                continue
        
        test_points = current_points.copy()
        test_points[m, 0] = new_x
        test_points[m, 1] = new_y
        
        new_areas = current_areas.copy()
        tri_idxs = _point_tri_idx[m]
        for ti in tri_idxs:
            a, b, c = _idx_arr[ti]
            det = test_points[a,0]*(test_points[b,1]-test_points[c,1]) + \
                  test_points[b,0]*(test_points[c,1]-test_points[a,1]) + \
                  test_points[c,0]*(test_points[a,1]-test_points[b,1])
            new_areas[ti] = 0.5 * abs(det)
        
        new_min = float(np.min(new_areas))
        delta = new_min - min_a
        
        # Accept if improvement or with probability exp(-delta/T) for worsening
        if delta > 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_points = test_points
            current_areas = new_areas
            min_a = new_min
            if new_min > best_min:
                best_min = new_min
                best_points = current_points.copy()
        
        T *= cooling
    
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

def _delone_spread_init(seed=0):
    """
    Initialize points using a space-filling approach: start with corners/centroid,
    then add points to maximize minimum distance (like farthest-point sampling).
    """
    np.random.seed(seed)
    points = []
    
    # Start with triangle vertices and centroid
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    points.append([0.5, H/3])  # centroid
    
    # Add remaining points via farthest-point sampling
    for _ in range(N - 4):
        best_pt = None
        best_min_dist = -1
        for trial in range(200):
            # Random point in triangle
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            x = r1 + 0.5 * r2
            y = H * r2
            if not _is_inside_triangle(x, y, tol=1e-6):
                continue
            
            # Compute min distance to existing points
            dists = [np.sqrt((x - p[0])**2 + (y - p[1])**2) for p in points]
            min_dist = min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pt = [x, y]
        
        if best_pt is not None:
            points.append(best_pt)
        else:
            # Fallback: random
            while True:
                r1 = np.random.random()
                r2 = np.random.random()
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                x = r1 + 0.5 * r2
                y = H * r2
                if _is_inside_triangle(x, y, tol=1e-6):
                    points.append([x, y])
                    break
    
    return np.array(points)

def _gradient_ascent_min_area(points, time_limit=15.0, max_iter=500):
    """
    Gradient ascent on the minimum area using subgradient method.
    Uses Delaunay triangles for the active set.
    """
    current_points = points.copy()
    current_areas = _compute_areas(current_points)
    current_min = float(np.min(current_areas))
    best_points = current_points.copy()
    best_min = current_min
    
    start_t = time.time()
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() - start_t > time_limit:
            break
        
        # Find minimum triangles
        min_a = float(np.min(current_areas))
        threshold = min_a + 1e-9
        min_tris = np.where(current_areas <= threshold)[0]
        
        if len(min_tris) == 0:
            break
        
        # Compute subgradient of min area
        grad = np.zeros((N, 2))
        
        for ti in min_tris:
            a, b, c = _idx_arr[ti]
            xa, ya = current_points[a]
            xb, yb = current_points[b]
            xc, yc = current_points[c]
            det_val = xa*(yb-yc) + xb*(yc-ya) + xc*(ya-yb)
            sign_val = 1.0 if det_val >= 0 else -1.0
            
            grad[a, 0] += 0.5 * sign_val * (yb - yc)
            grad[a, 1] += 0.5 * sign_val * (-(xb - xc))
            grad[b, 0] += 0.5 * sign_val * (yc - ya)
            grad[b, 1] += 0.5 * sign_val * (-(xc - xa))
            grad[c, 0] += 0.5 * sign_val * (ya - yb)
            grad[c, 1] += 0.5 * sign_val * (-(xa - xb))
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad /= grad_norm
        
        new_points = current_points + lr * grad
        new_points = _project_to_triangle(new_points)
        
        new_areas = _compute_areas(new_points)
        new_min = float(np.min(new_areas))
        
        if new_min > current_min:
            current_points = new_points
            current_areas = new_areas
            current_min = new_min
            if new_min > best_min:
                best_min = new_min
                best_points = new_points.copy()
            lr = min(lr * 1.1, 0.05)
        else:
            lr *= 0.8
        
        lr *= 0.999
    
    return best_points, best_min

def _barycentric_differential_evolution(time_limit=25.0):
    """
    Use differential evolution in barycentric coordinates (naturally constrained).
    """
    start_t = time.time()
    
    def _bary_objective(bary_flat):
        bary = bary_flat.reshape(N, 2)
        # Convert to Cartesian
        u = np.clip(bary[:, 0], 0.0, 1.0)
        v = np.clip(bary[:, 1], 0.0, 1.0)
        s = u + v
        mask = s > 1.0
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
        
        pts = np.empty((N, 2))
        pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
        pts[:, 1] = (1.0 - u - v) * H
        
        # Use exact min area for accuracy in objective
        areas = _compute_areas(pts)
        return -np.min(areas)
    
    # Bounds in barycentric space
    bounds = [(0.0, 1.0)] * (N * 2)
    
    def callback(xk, convergence):
        return (time.time() - start_t > time_limit - 2)

    try:
        result = differential_evolution(
            _bary_objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=True,
            init='sobol',
            callback=callback,
            workers=1
        )
        
        if result.success or result.fun < 0:
            bary_opt = result.x.reshape(N, 2)
            u = np.clip(bary_opt[:, 0], 0.0, 1.0)
            v = np.clip(bary_opt[:, 1], 0.0, 1.0)
            s = u + v
            mask = s > 1.0
            if np.any(mask):
                u[mask] /= s[mask]
                v[mask] /= s[mask]
            
            pts = np.empty((N, 2))
            pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
            pts[:, 1] = (1.0 - u - v) * H
            
            pts = _project_to_triangle(pts)
            score = _min_area(pts)
            return pts, score
    except Exception:
        pass

    return None, -np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Global Search with Differential Evolution
    de_points, de_score = _barycentric_differential_evolution(
        time_limit=min(30.0, (deadline - start_time) * 0.3)
    )
    if de_points is not None and de_score > best_score:
        best_score = de_score
        best_points = de_points.copy()
    
    # Phase 2: Local Refinement
    if best_points is not None:
        # Use Nelder-Mead for refinement
        remaining = deadline - time.time() - 5
        if remaining > 5:
            # Convert to flat array for Nelder-Mead
            x0 = best_points.flatten()
            def obj(x):
                pts = x.reshape(N, 2)
                pts = _project_to_triangle(pts)
                return -_min_area(pts)
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                new_pts = res.x.reshape(N, 2)
                new_pts = _project_to_triangle(new_pts)
                new_score = _min_area(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts.copy()
            except Exception:
                pass
        
        # Final fast local search
        remaining = deadline - time.time() - 3
        if remaining > 3 and best_points is not None:
            final_points, final_score = _fast_local_search(
                best_points,
                max_iter=1000,
                step_sizes=[0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
                time_limit=min(5.0, remaining * 0.8)
            )
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
    
    # Fallback
    if best_points is None:
        best_points = _sobol_points_in_triangle(N, seed=0)
        best_points = _project_to_triangle(best_points)
    
    best_points = _project_to_triangle(best_points)
    
    return best_points
