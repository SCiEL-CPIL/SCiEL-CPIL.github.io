import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4.0

# Precompute indices for all 165 triangles
_triplets = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _triplets])
_T_J = np.array([t[1] for t in _triplets])
_T_K = np.array([t[2] for t in _triplets])

def _cartesian_to_barycentric(points):
    """Convert Cartesian (x,y) to barycentric (a,b,c) where a+b+c=1, a,b,c>=0."""
    a = np.empty_like(points[:, 0])
    b = np.empty_like(points[:, 0])
    c = np.empty_like(points[:, 0])
    for i in range(len(points)):
        x, y = points[i, 0], points[i, 1]
        c[i] = y / S3H
        b[i] = x - 0.5 * c[i]
        a[i] = 1.0 - b[i] - c[i]
    return np.stack([a, b, c], axis=1)

def _barycentric_to_cartesian(bary):
    """Convert barycentric (a,b,c) to Cartesian (x,y)."""
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * S3H
    return np.stack([x, y], axis=1)

def _project_bary(bary):
    """Project barycentric coordinates to valid simplex (a+b+c=1, a,b,c>=0)."""
    b = bary.copy()
    for i in range(len(b)):
        a, bb, c = b[i, 0], b[i, 1], b[i, 2]
        if a < 0:
            a = 0.0
            s = bb + c
            if s > 1e-15:
                bb /= s
                c /= s
            else:
                bb = 0.5
                c = 0.5
        elif bb < 0:
            bb = 0.0
            s = a + c
            if s > 1e-15:
                a /= s
                c /= s
            else:
                a = 0.5
                c = 0.5
        elif c < 0:
            c = 0.0
            s = a + bb
            if s > 1e-15:
                a /= s
                bb /= s
            else:
                a = 0.5
                bb = 0.5
        b[i] = [a, bb, c]
    return b

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
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
        involved.add(int(_T_I[idx]))
        involved.add(int(_T_J[idx]))
        involved.add(int(_T_K[idx]))
    return min_area, sorted(involved)

def _soft_min_area(points, beta=50.0):
    """Smooth approximation of the minimum area using log-sum-exp."""
    areas = _compute_areas(points)
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - log_sum / beta

def _push_apart_gradient(points, beta=50.0):
    """
    Compute gradient of -soft_min_area w.r.t. point positions.
    This gives a 'push apart' direction for the smallest triangles.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    # Soft weights: w_i = exp(-beta*(area_i - min_a)) / sum
    exp_vals = np.exp(-beta * (areas - min_a))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of area w.r.t. each point
    # For triangle (i,j,k): area = 0.5*|cross|
    # d(area)/d(points[i]) depends on cross sign
    grad = np.zeros_like(points)
    cross = (points[_T_J, 0] - points[_T_I, 0]) * (points[_T_K, 1] - points[_T_I, 1]) - \
            (points[_T_J, 1] - points[_T_I, 1]) * (points[_T_K, 0] - points[_T_I, 0])
    signs = np.sign(cross)
    
    # For each triangle, the gradient of area w.r.t. point i is:
    # dA/dxi = 0.5 * sign(cross) * (yk - yj)
    # dA/dyi = 0.5 * sign(cross) * (xj - xk)
    # Similarly for j and k
    
    # Accumulate weighted gradients
    for t_idx in range(len(_triplets)):
        w = weights[t_idx]
        if w < 1e-10:
            continue
        i, j, k = _T_I[t_idx], _T_J[t_idx], _T_K[t_idx]
        s = 0.5 * signs[t_idx]
        
        # Gradient w.r.t. point i
        grad[i, 0] += w * s * (points[k, 1] - points[j, 1])
        grad[i, 1] += w * s * (points[j, 0] - points[k, 0])
        # Gradient w.r.t. point j
        grad[j, 0] += w * s * (points[k, 1] - points[i, 1])
        grad[j, 1] += w * s * (points[i, 0] - points[k, 0])
        # Gradient w.r.t. point k
        grad[k, 0] += w * s * (points[j, 1] - points[i, 1])
        grad[k, 1] += w * s * (points[i, 0] - points[j, 0])
    
    return grad

def _spring_based_init(rng, n_points=N):
    """
    Generate initial configuration using spring-based topology.
    Place points in a quasi-lattice pattern and relax with springs.
    """
    # Start with a regular grid pattern inside the triangle
    # Use a hexagonal lattice pattern for better packing
    points = np.zeros((n_points, 2))
    
    # Place 3 corners
    points[0] = [0.0, 0.0]
    points[1] = [1.0, 0.0]
    points[2] = [0.5, S3H]
    
    # Place remaining 8 points in a quasi-lattice
    # Use a 4x3 grid pattern mapped to the triangle
    remaining = list(range(3, n_points))
    rng.shuffle(remaining)
    
    # Define target positions for the 8 interior points
    targets = [
        [0.5, 0.0],       # mid bottom
        [0.75, S3H/2],    # mid right  
        [0.25, S3H/2],    # mid left
        [1/3, S3H/3],     # centroid-ish
        [0.333, 0.15],
        [0.667, 0.15],
        [0.5, S3H*0.6],
        [0.5, S3H*0.3],
    ]
    
    for idx, target in zip(remaining, targets):
        points[idx] = target + rng.normal(0, 0.05, size=2)
    
    # Relax with spring forces to maximize minimum area
    for _ in range(200):
        areas = _compute_areas(points)
        min_a = np.min(areas)
        
        # Compute forces: push apart points in small triangles
        forces = np.zeros_like(points)
        for t_idx in range(len(_triplets)):
            area = areas[t_idx]
            if area < min_a * 1.5:  # Only consider triangles near minimum
                i, j, k = _triplets[t_idx]
                # Force proportional to how much smaller this triangle is
                strength = max(0, min_a * 1.5 - area) / (min_a * 1.5 + 1e-10)
                
                # Compute unit vectors from each point to centroid of triangle
                centroid = (points[i] + points[j] + points[k]) / 3
                for p_idx in [i, j, k]:
                    vec = points[p_idx] - centroid
                    dist = np.linalg.norm(vec)
                    if dist > 1e-10:
                        forces[p_idx] += strength * vec / dist
        
        # Apply forces with damping
        step = 0.01
        points += step * forces
        points = _project_to_triangle(points)
    
    return points

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    for m in range(N):
        x, y = points[m, 0], points[m, 1]
        c = y / S3H
        b = x - 0.5 * c
        a = 1.0 - b - c
        
        if a < 0:
            a = 0.0
            s = b + c
            if s > 1e-15:
                b /= s
                c /= s
            else:
                b = 0.5
                c = 0.5
        elif b < 0:
            b = 0.0
            s = a + c
            if s > 1e-15:
                a /= s
                c /= s
            else:
                a = 0.5
                c = 0.5
        elif c < 0:
            c = 0.0
            s = a + b
            if s > 1e-15:
                a /= s
                b /= s
            else:
                a = 0.5
                b = 0.5
        
        points[m, 0] = b * 1.0 + c * 0.5
        points[m, 1] = c * S3H
    return points

def _objective_bary(bary_params, beta=50.0):
    """Objective in barycentric coordinates."""
    bary = _project_bary(bary_params.reshape(-1, 3))
    points = _barycentric_to_cartesian(bary)
    return -_soft_min_area(points, beta)

def _local_nudge_search(points, max_iter=500, step_sizes=None, time_limit=None):
    """Local search that only moves points involved in the smallest triangles."""
    if step_sizes is None:
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min = _min_area(current_points)
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
                    dx = np.random.uniform(-step_size, step_size)
                    dy = np.random.uniform(-step_size, step_size)
                    
                    new_x = current_points[idx, 0] + dx
                    new_y = current_points[idx, 1] + dy
                    
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    test_points = _project_to_triangle(test_points)
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

def _gradient_descent(points, beta=50.0, lr=0.01, max_iter=1000, time_limit=10.0):
    """
    Gradient descent using the push-apart gradient.
    """
    start_t = time.time()
    current = points.copy()
    best = current.copy()
    best_score = _min_area(current)
    
    for i in range(max_iter):
        if time.time() - start_t > time_limit:
            break
        
        grad = _push_apart_gradient(current, beta)
        current = current + lr * grad
        current = _project_to_triangle(current)
        
        score = _min_area(current)
        if score > best_score:
            best_score = score
            best = current.copy()
        
        # Adaptive learning rate
        if i % 100 == 0 and i > 0:
            lr *= 0.8
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = -np.inf
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (deadline - margin)
    
    def _update_best(pts):
        nonlocal best_points, best_score
        area = _min_area(pts)
        if area > best_score:
            best_score = area
            best_points = pts.copy()
    
    # Generate multiple diverse initial configurations
    bases = []
    
    # Base 1: Spring-based initialization
    bases.append(_spring_based_init(rng))
    
    # Base 2: Symmetric grid
    init2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    bases.append(_project_to_triangle(init2))
    
    # Base 3-10: Random perturbations of spring-based
    for _ in range(8):
        pts = bases[0] + rng.normal(0, 0.15, size=(N, 2))
        bases.append(_project_to_triangle(pts))
    
    # Update best with all bases
    for b in bases:
        _update_best(b)
    
    # Phase A: Gradient descent + L-BFGS-B optimization from each base
    n_starts = len(bases)
    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        
        pts = bases[s].copy()
        _update_best(pts)
        
        # Gradient descent with push-apart
        if _elapsed_ok(25):
            pts, _ = _gradient_descent(pts, beta=30.0, lr=0.02, max_iter=500, time_limit=5.0)
            _update_best(pts)
        
        # L-BFGS-B with soft min
        if _elapsed_ok(20):
            x0 = pts.flatten()
            
            def obj(params):
                p = params.reshape(N, 2)
                p = _project_to_triangle(p.copy())
                return -_soft_min_area(p, beta=50.0)
            
            try:
                result = minimize(obj, x0, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                new_pts = result.x.reshape(N, 2)
                new_pts = _project_to_triangle(new_pts)
                _update_best(new_pts)
            except Exception:
                pass
        
        # Nelder-Mead polish
        if _elapsed_ok(15):
            def nm_obj(x):
                p = x.reshape(N, 2)
                p = _project_to_triangle(p)
                return -_min_area(p)
            
            try:
                res = minimize(nm_obj, pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
                p = res.x.reshape(N, 2)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass
    
    # Phase B: Local nudge search on best
    if best_points is not None and _elapsed_ok(15):
        refined_points, refined_score = _local_nudge_search(
            best_points,
            max_iter=500,
            step_sizes=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
            time_limit=min(10.0, deadline - time.time() - 5)
        )
        _update_best(refined_points)
    
    # Phase C: Final Nelder-Mead polish
    if best_points is not None and _elapsed_ok(8):
        def nm_obj(x):
            p = x.reshape(N, 2)
            p = _project_to_triangle(p)
            return -_min_area(p)
        
        try:
            res = minimize(nm_obj, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13})
            p = res.x.reshape(N, 2)
            p = _project_to_triangle(p)
            _update_best(p)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = bases[0]
    
    # Final projection
    best_points = _project_to_triangle(best_points)
    
    return best_points
