"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    Uses convex hull facet circumradii (vectorized).
    """
    if len(points) < 3:
        return 180.0
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
    
    simplices = hull.simplices
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normals = normals / norms
    
    # Ensure outward-facing
    dots = np.sum(normals * p1, axis=1)
    normals = np.where(dots[:, None] < 0, -normals, normals)
    
    h = np.sum(normals * p1, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    max_angle = np.max(angles)
    
    return max_angle * 180.0 / np.pi


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _smooth_surrogate(params, beta=30.0):
    """
    Smooth surrogate using soft-max over angular distances to a dense test grid.
    This is truly differentiable and avoids the discontinuity of convex hull changes.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    # Use a moderately dense grid of test points
    n_theta = 15
    n_phi = 30
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(thetas, phis, indexing='ij')
    
    # Convert to Cartesian
    test_pts = np.column_stack([
        np.sin(T.ravel()) * np.cos(P.ravel()),
        np.sin(T.ravel()) * np.sin(P.ravel()),
        np.cos(T.ravel())
    ])
    
    # Compute angular distance from each test point to nearest configuration point
    # dot products
    dots = np.sum(pts[None, :, :] * test_pts[:, None, :], axis=2)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # shape: (n_test, N)
    
    # For each test point, find the minimum angle (nearest neighbor)
    min_angles = np.min(angles, axis=1)
    
    # Soft-max approximation of max(min_angles)
    # We want to minimize the maximum of min_angles
    # soft_max(x, beta) = (1/beta) * log(sum(exp(beta * x)))
    beta_scaled = beta * min_angles
    max_val = np.max(beta_scaled)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_scaled - max_val)))
    
    return logsumexp / beta


def _facet_smooth_surrogate(params, beta=50.0):
    """
    Facet-based smooth surrogate: for each convex hull facet, compute the
    angular distance from the facet normal (circumcenter direction) to the
    farthest vertex, then take soft-max over all facets.
    This is more accurate than grid-based since it uses actual hull geometry.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return 1.0  # fallback: large value
    
    simplices = hull.simplices
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normals = normals / norms
    
    # Ensure outward-facing
    dots = np.sum(normals * p1, axis=1)
    normals = np.where(dots[:, None] < 0, -normals, normals)
    
    # h = dot(normal, p1) = cos(angle) where angle is the circumradius
    h = np.sum(normals * p1, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    # Soft-max over facet circumradii
    beta_scaled = beta * angles
    max_val = np.max(beta_scaled)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_scaled - max_val)))
    
    return logsumexp / beta


def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _golden_spiral_points(offset=0.0):
    """Generate golden spiral points with a phase offset for diversity."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _perturbed_fibonacci(seed=0, jitter=0.1):
    """Fibonacci points with small random perturbations."""
    rng = np.random.RandomState(seed)
    pts = _fibonacci_points()
    noise = rng.randn(N, 3) * jitter
    pts = pts + noise
    return _normalize(pts)


def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(N, 3)
    return _normalize(points)


def _icosahedron_based_points():
    """Generate points based on icosahedron vertices with perturbations."""
    # 12 vertices of regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # Expand to 50 points by adding subdivisions
    rng = np.random.RandomState(123)
    points = [verts[i] for i in range(12)]
    
    # Add 38 more points by perturbing and adding
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        # Check minimum distance to existing points
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.1:
            points.append(new_pt)
    
    return np.array(points[:N])


def _spherical_lloyd_iteration(points, max_iter=200, time_limit=60, damping=0.5):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        new_points = points.copy()
        
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            normal_sum = np.zeros(3)
            for fi in facets:
                simplex = hull.simplices[fi]
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                if np.dot(n, p1) < 0:
                    n = -n
                
                normal_sum += n
            
            if np.linalg.norm(normal_sum) > 1e-10:
                target = normal_sum / np.linalg.norm(normal_sum)
                new_points[pi] = points[pi] + damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=20):
    """
    Local optimization using Nelder-Mead on the true covering radius.
    High-precision polish with tight tolerances.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius_radians(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 5000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _coordinate_wise_nm(points, time_limit=15, tol=1e-10):
    """
    Coordinate-wise Nelder-Mead polish: optimize each point's 3 coordinates
    one at a time with very tight tolerance. This extracts gains from
    near-converged configurations by removing numerical noise.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    
    for round_num in range(3):  # Multiple passes
        if time.time() - start_time > time_limit:
            break
        
        for idx in range(N):
            if time.time() - start_time > time_limit:
                break
            
            other_points = np.delete(points, idx, axis=0)
            current_pt = points[idx].copy()
            
            def obj_single(params):
                pt = params / np.linalg.norm(params)
                all_pts = np.vstack([other_points, pt.reshape(1, 3)])
                return _covering_radius_radians(all_pts)
            
            x0 = current_pt
            res = minimize(obj_single, x0, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': tol, 'fatol': tol, 'maxfev': 200})
            new_pt = res.x / np.linalg.norm(res.x)
            points[idx] = new_pt
    
    return _normalize(points)


def _smooth_gradient_optimize(points, time_limit=20, beta=30.0):
    """
    Gradient-based optimization using smooth surrogate with BFGS.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='BFGS', 
                      options={'maxiter': 300, 'gtol': 1e-8, 'disp': False})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _sa_optimize_covering(points, T_start=1.0, T_end=1e-4, n_iter=5000, time_limit=100):
    """
    Simulated Annealing to minimize covering radius directly.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        T = T_start * (cooling ** i)
        
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        step_size = 0.05 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        new_cost = _covering_radius(points)
        delta_C = new_cost - current_cost
        
        if delta_C < 0 or np.random.rand() < np.exp(-delta_C / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            points[idx] = original_point
            
    return best_points


def _refine_point(points, idx, time_limit=5):
    """
    Refine a single point by optimizing its position given others fixed.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    other_points = np.delete(points, idx, axis=0)
    
    def objective_single(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pt = params / np.linalg.norm(params)
        all_pts = np.vstack([other_points, pt.reshape(1, 3)])
        return _covering_radius_radians(all_pts)
    
    x0 = points[idx]
    result = minimize(objective_single, x0, method='Nelder-Mead',
                      options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
    new_pt = result.x / np.linalg.norm(result.x)
    points[idx] = new_pt
    return points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Golden spiral with different offsets
    for offset in [0.1, 0.3, 0.7]:
        initializations.append(_golden_spiral_points(offset=offset))
    
    # Perturbed Fibonacci
    for seed in [0, 1, 2]:
        initializations.append(_perturbed_fibonacci(seed=seed, jitter=0.08))
    
    # Random with different seeds
    for seed in [42, 123, 777]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Allocate time for initialization phase: ~40% of total
    init_time_budget = 0.40 * total_time_limit
    lloyd_time_each = max(5.0, (init_time_budget - 5.0) / len(initializations))
    
    for init_pts in initializations:
        if _time_left() < 55:
            break
        
        # Lloyd's relaxation
        pts = _spherical_lloyd_iteration(init_pts, max_iter=80, 
                                          time_limit=min(lloyd_time_each, _time_left() - 50), 
                                          damping=0.5)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Facet-based smooth gradient optimization (more accurate surrogate)
    if _time_left() > 45:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(20, _time_left() - 35), beta=50.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Grid-based smooth gradient with higher beta
    if _time_left() > 30:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(15, _time_left() - 25), beta=60.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Nelder-Mead on true objective (high-precision polish)
    if _time_left() > 20:
        pts = _nelder_mead_optimize(best_points, time_limit=min(12, _time_left() - 15))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Coordinate-wise Nelder-Mead polish (extracts gains from near-converged configs)
    if _time_left() > 15:
        pts = _coordinate_wise_nm(best_points, time_limit=min(10, _time_left() - 12))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final Lloyd's polish with small damping
    if _time_left() > 10:
        pts = _spherical_lloyd_iteration(best_points, max_iter=30, 
                                          time_limit=min(5, _time_left() - 8), damping=0.1)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final high-precision coordinate-wise polish
    if _time_left() > 8:
        pts = _coordinate_wise_nm(best_points, time_limit=min(5, _time_left() - 6), tol=1e-12)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
