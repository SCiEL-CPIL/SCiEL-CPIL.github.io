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
    """Generate a uniform test grid on the sphere using Fibonacci spiral."""
    n_points = n_theta * n_phi
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    grid = np.stack([x, y, z], axis=-1)
    return grid


def _facet_smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate based on convex hull facet circumradii using log-sum-exp.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
    
    simplices = hull.simplices
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1)
    
    valid = norms > 1e-10
    if not np.any(valid):
        return np.pi
    
    normals = normals[valid] / norms[valid, np.newaxis]
    p1v = p1[valid]
    
    dots = np.sum(normals * p1v, axis=1)
    flip = dots < 0
    if np.any(flip):
        normals[flip] = -normals[flip]
    
    h = np.sum(normals * p1v, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    # Soft-max over facet angles using log-sum-exp
    if len(angles) == 0:
        return np.pi
    
    beta_angles = beta * angles
    max_val = np.max(beta_angles)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_angles - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _repulsion_energy_and_grad(points, k=10.0):
    """
    Compute repulsion energy and gradient to spread points evenly.
    Uses inverse-distance repulsion.
    """
    pts = _normalize(points)
    n = len(pts)
    
    # Compute pairwise distances
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)
    np.fill_diagonal(angles, np.pi)
    
    # Energy: sum of -1/d for each pair (repulsion)
    # Use a soft version to avoid singularities
    soft_angles = np.where(angles < 1e-10, 1e-10, angles)
    energy = -np.sum(1.0 / soft_angles)
    
    # Gradient: for each point i, gradient is sum over j of d(1/d_ij)/d(theta_ij)
    # d(1/d)/d(theta) = -1/d^2 * sign
    # But we need gradient w.r.t. position, not angle
    # Approximate: gradient direction is away from nearest neighbors
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = angles[i, j]
            if d < 1e-10:
                continue
            # Force direction: from j to i (away from j)
            force_dir = pts[i] - pts[j]
            force_norm = np.linalg.norm(force_dir)
            if force_norm < 1e-10:
                continue
            force_dir = force_dir / force_norm
            # Project onto tangent space at pts[i]
            tangent = force_dir - np.dot(force_dir, pts[i]) * pts[i]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-10:
                continue
            tangent = tangent / tangent_norm
            # Gradient contribution: -1/d^2 (negative because we want to increase distance)
            grad[i] += k * tangent / (d * d)
    
    return energy, grad


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


def _smooth_gradient_optimize(points, time_limit=30, beta=200.0, use_facet=True, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _facet_smooth_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxfun': 15000})
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


def _repulsion_refine(points, time_limit=10.0, max_iter=50):
    """
    Refine using repulsion energy to spread points more evenly.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _repulsion_energy_and_grad(points, k=5.0)
        
        # Move in negative gradient direction (repulsion pushes points apart)
        step_size = 0.05
        new_points = points - step_size * grad
        new_points = _normalize(new_points)
        
        new_cost = _covering_radius(new_points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points.copy()
            points = new_points
            step_size *= 0.9
        else:
            step_size *= 0.5
            if step_size < 0.001:
                break
    
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
    lloyd_time_each = 2.5
    for init_pts in initializations:
        if _time_left() < 55:
            break
        
        pts = _spherical_lloyd_iteration(init_pts, max_iter=80, time_limit=min(lloyd_time_each, _time_left() - 50), damping=0.3)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Repulsion refinement to improve spread
    if _time_left() > 50:
        pts = _repulsion_refine(best_points, time_limit=min(5, _time_left() - 45), max_iter=30)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 3: Worst facet relaxation
    if _time_left() > 45:
        pts = _worst_facet_relaxation(best_points, time_limit=min(6, _time_left() - 40), max_iter=30)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Facet-based smooth optimization (broad basin, low beta)
    if _time_left() > 38:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 32), beta=50.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5: Medium-beta facet-based optimization
    if _time_left() > 28:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 22), beta=100.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: High-beta facet-based optimization
    if _time_left() > 18:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(6, _time_left() - 13), beta=200.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 7: Worst facet relaxation
    if _time_left() > 12:
        pts = _worst_facet_relaxation(best_points, time_limit=min(4, _time_left() - 9), max_iter=25)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 8: Very high-beta facet-based polish
    if _time_left() > 8:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(4, _time_left() - 5), beta=500.0, use_facet=True)
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
