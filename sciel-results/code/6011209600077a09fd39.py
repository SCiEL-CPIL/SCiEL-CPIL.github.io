"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Global deadline management
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin for 120s limit

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

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
    n_facets = len(simplices)
    
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
    """Generate a grid of test points on the sphere for surrogate."""
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

def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

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

def _force_based_relaxation(points, time_limit=10.0, target_radius=None):
    """
    Force-based relaxation:
    1. Calculate the "worst" point on the sphere (farthest from any config point).
    2. Apply forces to move config points towards this worst point (gap filling).
    3. Apply repulsive forces between config points to maintain spacing.
    4. Update positions on the sphere.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    best_points = points.copy()
    best_cost = _covering_radius(points)
    
    if target_radius is None:
        target_radius = best_cost * np.pi / 180.0
    
    # Adaptive parameters
    step_size = 0.05
    repulsion_strength = 1.0
    attraction_strength = 2.0
    
    for i in range(200):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10.0:
            break
        
        # 1. Find the worst point on the sphere
        # Use a dense grid to approximate the maximum distance
        grid = _generate_test_grid(n_theta=30, n_phi=60)
        dots = np.clip(grid @ points.T, -1.0, 1.0)
        angles = np.arccos(dots)
        min_angles = np.min(angles, axis=1)
        worst_idx = np.argmax(min_angles)
        worst_point = grid[worst_idx]
        current_max_angle = min_angles[worst_idx]
        
        # If current max angle is close to target, we are near optimal
        if current_max_angle <= target_radius:
            break
            
        # 2. Compute forces
        forces = np.zeros_like(points)
        
        # Attraction towards worst point
        # Move points that are "responsible" for the gap towards the worst point
        # Simple heuristic: move all points slightly, but weight by distance to worst point
        dist_to_worst = np.arccos(np.clip(points @ worst_point, -1.0, 1.0))
        # Points far from worst point should move towards it? 
        # Actually, the point *closest* to the worst point is the one defining the gap.
        # We want to move that specific point towards the worst point.
        # But moving all points slightly towards the "center of mass" of the gap region is also good.
        
        # Let's identify the k-nearest neighbors to the worst point
        k = 3
        k_indices = np.argsort(dist_to_worst)[:k]
        
        for idx in k_indices:
            p = points[idx]
            # Direction from p to worst_point (tangent to sphere)
            v = worst_point - p
            v = v - np.dot(v, p) * p # Project to tangent plane
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                v = v / norm_v
                forces[idx] += attraction_strength * v
        
        # Repulsion between points
        # For each pair, if too close, repel
        # Vectorized repulsion is expensive for N=50, but O(N^2) is fine (2500 ops)
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                p1 = points[i_idx]
                p2 = points[j_idx]
                dot = np.dot(p1, p2)
                dist = np.arccos(np.clip(dot, -1.0, 1.0))
                
                # Repel if closer than some threshold (e.g., ideal spacing)
                ideal_spacing = 2 * np.arcsin(np.sqrt(N / (4 * np.pi))) # Rough estimate
                if dist < ideal_spacing:
                    # Force direction: away from each other
                    # Tangent component of p2 - p1 for p1
                    v1 = p2 - p1
                    v1 = v1 - np.dot(v1, p1) * p1
                    norm_v1 = np.linalg.norm(v1)
                    if norm_v1 > 1e-10:
                        v1 = v1 / norm_v1
                        forces[i_idx] -= repulsion_strength * v1
                        forces[j_idx] += repulsion_strength * v1
        
        # 3. Update positions
        new_points = points.copy()
        for i_idx in range(N):
            f = forces[i_idx]
            norm_f = np.linalg.norm(f)
            if norm_f > 1e-10:
                f = f / norm_f
                new_points[i_idx] = points[i_idx] + step_size * f
                new_points[i_idx] = new_points[i_idx] / np.linalg.norm(new_points[i_idx])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points.copy()
            points = new_points
            # Adaptive step size
            if i % 10 == 0:
                step_size *= 0.95
        else:
            # Reduce step size if no improvement
            step_size *= 0.8
            if step_size < 0.001:
                break
                
    return best_points

def _smooth_gradient_optimize(points, time_limit=20, beta=200.0):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 12000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)

def _nelder_mead_optimize(points, time_limit=10, maxiter=2000):
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
                      options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True, 'maxfev': 5000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    global _START_TIME
    _START_TIME = time.time()
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Random with different seeds
    for seed in [42, 123, 777, 999, 555]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Force-Based Relaxation from multiple starts
    # This is the main structural improvement
    for init_pts in initializations:
        if _time_left() < 30:
            break
        
        # Allocate time for FBR
        fbr_time = min(15.0, _time_left() - 25.0)
        if fbr_time < 5.0:
            break
            
        pts = _force_based_relaxation(init_pts, time_limit=fbr_time)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Smooth gradient optimization (polish)
    if _time_left() > 20:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(15, _time_left() - 10), beta=200.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 3: Nelder-Mead final polish
    if _time_left() > 10:
        pts = _nelder_mead_optimize(best_points, time_limit=min(8, _time_left() - 5), maxiter=2000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: One more FBR pass if time allows (refinement)
    if _time_left() > 5:
        pts = _force_based_relaxation(best_points, time_limit=min(3, _time_left() - 2))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
