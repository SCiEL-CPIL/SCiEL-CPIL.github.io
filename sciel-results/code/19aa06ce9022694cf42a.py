"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
    norms = np.linalg.norm(normals, axis=1)
    
    valid = norms > 1e-10
    if not np.any(valid):
        return 180.0
    
    normals = normals[valid] / norms[valid, np.newaxis]
    p1v = p1[valid]
    
    # Ensure normals point outward
    dots = np.sum(normals * p1v, axis=1)
    flip = dots < 0
    if np.any(flip):
        normals[flip] = -normals[flip]
    
    h = np.sum(normals * p1v, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    return float(np.max(angles) * 180.0 / np.pi)


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _generate_test_grid(n_theta=50, n_phi=100):
    """Generate a grid of test points on the sphere."""
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


def _smooth_surrogate(params, beta=200.0, grid=None):
    """
    Vectorized smooth differentiable surrogate for the covering radius.
    Uses a grid of test points and log-sum-exp for soft min/max.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    if grid is None:
        grid = _generate_test_grid()
        
    # Compute angular distances: grid (M,3) x pts (N,3) -> (M,N)
    dots = np.clip(grid @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min over points for each grid point
    neg_beta_angles = -beta * angles
    max_neg = np.max(neg_beta_angles, axis=1, keepdims=True)
    exp_terms = np.exp(neg_beta_angles - max_neg)
    sum_exp = np.sum(exp_terms, axis=1)
    logsumexp_neg = max_neg.squeeze() + np.log(sum_exp)
    soft_min_angles = -logsumexp_neg / beta  # (M,)
    
    # Soft-max over grid points
    beta_soft_min = beta * soft_min_angles
    max_val = np.max(beta_soft_min)
    logsumexp_max = max_val + np.log(np.sum(np.exp(beta_soft_min - max_val)))
    soft_max = logsumexp_max / beta
    
    return soft_max


def _facet_smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate based on convex hull facet circumradii.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
    
    facet_angles = []
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        facet_angles.append(angle)
    
    if not facet_angles:
        return np.pi
    
    facet_angles = np.array(facet_angles)
    
    # Soft-max over facet angles
    beta_facet = beta * facet_angles
    max_val = np.max(beta_facet)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_facet - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _smooth_gradient_optimize(points, time_limit=30, beta=200.0, use_facet=False, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    if use_facet:
        grid = None
    elif grid is None:
        grid = _generate_test_grid()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        if use_facet:
            return _facet_smooth_surrogate(params, beta=beta)
        else:
            return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 12000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _jittered_fibonacci_points(seed=123, jitter=0.05):
    """Generate Fibonacci points with small random jitter to break symmetries."""
    pts = _fibonacci_points()
    rng = np.random.RandomState(seed)
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
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    rng = np.random.RandomState(123)
    points = [verts[i] for i in range(12)]
    
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.1:
            points.append(new_pt)
    
    return np.array(points[:N])


def _spherical_lloyd_iteration(points, max_iter=100, time_limit=30, damping=0.3):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) with adaptive damping.
    Vectorized facet processing for speed.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    current_damping = damping
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        simplices = hull.simplices
        n_facets = len(simplices)
        
        # Vectorized facet normal computation
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            break
        
        normals = normals[valid] / norms[valid, np.newaxis]
        p1v = p1[valid]
        
        # Ensure normals point outward
        dots = np.sum(normals * p1v, axis=1)
        flip = dots < 0
        if np.any(flip):
            normals[flip] = -normals[flip]
        
        # Accumulate weighted normals per point
        normal_sums = np.zeros((N, 3))
        weight_sums = np.zeros(N)
        
        # For each valid facet, add its weighted normal to all three vertices
        valid_indices = np.where(valid)[0]
        for fi in valid_indices:
            simplex = simplices[fi]
            w = norms[fi]
            n = normals[fi]
            for pi in simplex:
                normal_sums[pi] += w * n
                weight_sums[pi] += w
        
        new_points = points.copy()
        for pi in range(N):
            if weight_sums[pi] > 1e-10:
                target = normal_sums[pi] / weight_sums[pi]
                target_norm = np.linalg.norm(target)
                if target_norm > 1e-10:
                    target = target / target_norm
                    new_points[pi] = points[pi] + current_damping * (target - points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
            current_damping *= 0.95
        else:
            current_damping *= 0.5
            if current_damping < 0.01:
                break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=20, maxiter=2000):
    """
    Local optimization using Nelder-Mead on the true covering radius.
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
                      options={'maxiter': maxiter, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True, 'maxfev': 8000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _worst_facet_relaxation(points, time_limit=10.0, max_iter=50):
    """
    Targeted relaxation: move points adjacent to the worst facet towards
    reduce its circumradius.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        simplices = hull.simplices
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            break
        
        normals = normals[valid] / norms[valid, np.newaxis]
        p1v = p1[valid]
        
        dots = np.sum(normals * p1v, axis=1)
        flip = dots < 0
        if np.any(flip):
            normals[flip] = -normals[flip]
        
        h = np.sum(normals * p1v, axis=1)
        h = np.clip(h, -1.0, 1.0)
        angles = np.arccos(h)
        
        # Find the worst facet
        worst_idx = np.argmax(angles)
        worst_angle = angles[worst_idx]
        
        if worst_angle < 0.1:  # ~5.7 degrees, already good
            break
        
        # Get the three points of the worst facet
        valid_indices = np.where(valid)[0]
        worst_simplex = simplices[valid_indices[worst_idx]]
        
        # Compute circumcenter of the worst facet
        wp1 = points[worst_simplex[0]]
        wp2 = points[worst_simplex[1]]
        wp3 = points[worst_simplex[2]]
        
        wv1 = wp2 - wp1
        wv2 = wp3 - wp1
        wn = np.cross(wv1, wv2)
        wn_norm = np.linalg.norm(wn)
        if wn_norm < 1e-10:
            break
        wn = wn / wn_norm
        
        if np.dot(wn, wp1) < 0:
            wn = -wn
        
        # Move the three points slightly towards the circumcenter direction
        # which reduces the facet's circumradius
        damping = 0.1
        for pi in worst_simplex:
            target = wn
            target = target / np.linalg.norm(target)
            points[pi] = points[pi] + damping * (target - points[pi])
        
        points = _normalize(points)
        new_cost = _covering_radius(points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = points.copy()
    
    return best_points


def _tangent_polish(points, time_limit=15.0, max_iter=200):
    """
    High-precision polish using tangent space optimization.
    Projects gradients onto the tangent space of the sphere.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        simplices = hull.simplices
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            break
        
        normals = normals[valid] / norms[valid, np.newaxis]
        p1v = p1[valid]
        
        dots = np.sum(normals * p1v, axis=1)
        flip = dots < 0
        if np.any(flip):
            normals[flip] = -normals[flip]
        
        h = np.sum(normals * p1v, axis=1)
        h = np.clip(h, -1.0, 1.0)
        angles = np.arccos(h)
        
        # Find the worst facet
        worst_idx = np.argmax(angles)
        worst_angle = angles[worst_idx]
        
        if worst_angle < 0.01:  # Very small, likely converged
            break
        
        # Get the three points of the worst facet
        valid_indices = np.where(valid)[0]
        worst_simplex = simplices[valid_indices[worst_idx]]
        
        # Compute gradient directions for the three points
        # We want to move them to reduce the circumradius
        wp1 = points[worst_simplex[0]]
        wp2 = points[worst_simplex[1]]
        wp3 = points[worst_simplex[2]]
        
        # The optimal direction is towards the facet normal (circumcenter direction)
        wv1 = wp2 - wp1
        wv2 = wp3 - wp1
        wn = np.cross(wv1, wv2)
        wn_norm = np.linalg.norm(wn)
        if wn_norm < 1e-10:
            break
        wn = wn / wn_norm
        
        if np.dot(wn, wp1) < 0:
            wn = -wn
        
        # Adaptive step size
        step_size = min(0.05, worst_angle * 0.5)
        
        for pi in worst_simplex:
            # Project movement onto tangent space
            current = points[pi]
            direction = wn - np.dot(wn, current) * current
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-10:
                direction = direction / dir_norm
                points[pi] = current + step_size * direction
        
        points = _normalize(points)
        new_cost = _covering_radius(points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = points.copy()
        else:
            # Reduce step size
            step_size *= 0.8
    
    return best_points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0  # Safety margin for 150s limit
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Jittered Fibonacci (breaks symmetry)
    for seed in [123, 456, 789, 246, 357]:
        initializations.append(_jittered_fibonacci_points(seed=seed, jitter=0.05))
    
    # Random with different seeds
    for seed in [42, 123, 777, 999, 555, 333, 888]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Lloyd's relaxation from multiple starts
    lloyd_time_each = 3.0
    for init_pts in initializations:
        if _time_left() < 50:
            break
        
        pts = _spherical_lloyd_iteration(init_pts, max_iter=60, time_limit=min(lloyd_time_each, _time_left() - 45), damping=0.3)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Worst facet relaxation
    if _time_left() > 45:
        pts = _worst_facet_relaxation(best_points, time_limit=min(8, _time_left() - 40), max_iter=30)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 3: Low-beta grid-based optimization (broad basin)
    if _time_left() > 40:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(10, _time_left() - 34), beta=15.0, use_facet=False)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Medium-beta grid-based optimization
    if _time_left() > 30:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(10, _time_left() - 24), beta=50.0, use_facet=False)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5: Facet-based smooth optimization (directly targets covering radius)
    if _time_left() > 22:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 17), beta=150.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: Worst facet relaxation
    if _time_left() > 15:
        pts = _worst_facet_relaxation(best_points, time_limit=min(5, _time_left() - 12), max_iter=25)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 7: High-Beta grid-based polish (sharpens focus on worst facet)
    if _time_left() > 10:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(6, _time_left() - 7), beta=300.0, use_facet=False)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 8: Very high-Beta facet-based polish
    if _time_left() > 6:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(4, _time_left() - 4), beta=500.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 9: Tangent space polish for high precision
    if _time_left() > 4:
        pts = _tangent_polish(best_points, time_limit=min(3, _time_left() - 3), max_iter=200)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
