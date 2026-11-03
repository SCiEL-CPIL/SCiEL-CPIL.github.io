"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# Global deadline management
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def _generate_test_grid(n_theta=60, n_phi=120):
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

def _facet_smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate based on facet circumradii using log-sum-exp.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi / 2
    
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
        return np.pi / 2
    
    normals = normals[valid] / norms[valid, np.newaxis]
    p1v = p1[valid]
    
    dots = np.sum(normals * p1v, axis=1)
    flip = dots < 0
    if np.any(flip):
        normals[flip] = -normals[flip]
    
    h = np.sum(normals * p1v, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    # Soft max
    beta_angles = beta * angles
    max_val = np.max(beta_angles)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_angles - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max

def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _jittered_fibonacci_points(seed=42):
    """Generate jittered Fibonacci spiral points."""
    pts = _fibonacci_points()
    rng = np.random.RandomState(seed)
    noise = rng.randn(N, 3) * 0.1
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

def _spherical_lloyd_iteration(points, max_iter=50, time_limit=10.0):
    """
    Spherical Lloyd's algorithm: iteratively move each point to the
    centroid of its Voronoi cell on the sphere.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 5.0:
            break
        
        new_points = np.zeros_like(points)
        
        for i in range(N):
            # Compute angular distance from each point to point i
            dots = np.clip(points @ points[i], -1.0, 1.0)
            angles = np.arccos(dots)
            
            # Find the Voronoi cell: points closer to i than to any other point
            # Use a grid to approximate the cell
            grid = _generate_test_grid(n_theta=20, n_phi=40)
            grid_dots = np.clip(grid @ points.T, -1.0, 1.0)
            grid_angles = np.arccos(grid_dots)
            
            # Cell of point i: grid points where angle to i is minimum
            min_idx = np.argmin(grid_angles, axis=1)
            cell_mask = (min_idx == i)
            
            if np.any(cell_mask):
                cell_pts = grid[cell_mask]
                centroid = np.sum(cell_pts, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-10:
                    new_points[i] = centroid / centroid_norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        new_points = _normalize(new_points)
        
        # Check convergence
        diff = np.linalg.norm(new_points - points)
        points = new_points
        
        if diff < 1e-6:
            break
    
    return points

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
        dist_to_worst = np.arccos(np.clip(points @ worst_point, -1.0, 1.0))
        k = 3
        k_indices = np.argsort(dist_to_worst)[:k]
        
        for idx in k_indices:
            p = points[idx]
            v = worst_point - p
            v = v - np.dot(v, p) * p
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                v = v / norm_v
                forces[idx] += attraction_strength * v
        
        # Repulsion between points
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                p1 = points[i_idx]
                p2 = points[j_idx]
                dot = np.dot(p1, p2)
                dist = np.arccos(np.clip(dot, -1.0, 1.0))
                
                ideal_spacing = 2 * np.arcsin(np.sqrt(N / (4 * np.pi)))
                if dist < ideal_spacing:
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
            if i % 10 == 0:
                step_size *= 0.95
        else:
            step_size *= 0.8
            if step_size < 0.001:
                break
                
    return best_points

def _smooth_gradient_optimize(points, time_limit=20, beta=200.0, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    if grid is None:
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

def _facet_gradient_optimize(points, time_limit=20, beta=100.0):
    """
    Gradient-based optimization using facet-based smooth surrogate.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _facet_smooth_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 8000})
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

def _riesz_energy_optimize(points, s=20.0, time_limit=15.0):
    """
    Maximize Riesz s-energy (equivalent to minimizing -energy) to spread points.
    This indirectly minimizes covering radius via packing-covering duality.
    """
    start_time = time.time()
    
    def negative_energy(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        
        # Compute pairwise distances
        dots = np.clip(pts @ pts.T, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(angles, 0.0)
        
        # Riesz energy: sum of 1/d^s
        # Use log-sum-exp for stability
        neg_s_angles = -s * angles
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        neg_s_angles_masked = np.where(mask, neg_s_angles, -np.inf)
        
        max_val = np.max(neg_s_angles_masked)
        logsumexp = max_val + np.log(np.sum(np.exp(neg_s_angles_masked - max_val)))
        energy = np.exp(logsumexp)
        
        return -energy
    
    x0 = points.flatten()
    result = minimize(negative_energy, x0, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 8000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    global _START_TIME
    _START_TIME = time.time()
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Jittered Fibonacci
    for seed in [1, 2, 3]:
        initializations.append(_jittered_fibonacci_points(seed=seed))
    
    # Random with different seeds
    for seed in [42, 123, 777, 999, 555]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Force-Based Relaxation from multiple starts
    for init_pts in initializations:
        if _time_left() < 30:
            break
        
        fbr_time = min(12.0, _time_left() - 25.0)
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
    
    # Phase 2: Spherical Lloyd's iteration for structural improvement
    if _time_left() > 25:
        pts = _spherical_lloyd_iteration(best_points, max_iter=30, time_limit=min(10, _time_left() - 15))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 3: Multi-stage beta cascade for smooth gradient optimization
    if _time_left() > 30:
        grid = _generate_test_grid(n_theta=50, n_phi=100)
        
        for beta in [15, 50, 150, 300, 500]:
            if _time_left() < 15:
                break
            tl = min(8, _time_left() - 10)
            if tl < 3:
                break
            pts = _smooth_gradient_optimize(best_points, time_limit=tl, beta=beta, grid=grid)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Phase 4: Facet-based gradient optimization
    if _time_left() > 20:
        pts = _facet_gradient_optimize(best_points, time_limit=min(10, _time_left() - 10), beta=100.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5: Riesz energy optimization for packing
    if _time_left() > 15:
        pts = _riesz_energy_optimize(best_points, s=20.0, time_limit=min(8, _time_left() - 10))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: Nelder-Mead final polish
    if _time_left() > 10:
        pts = _nelder_mead_optimize(best_points, time_limit=min(6, _time_left() - 5), maxiter=2000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 7: One more FBR pass if time allows
    if _time_left() > 5:
        pts = _force_based_relaxation(best_points, time_limit=min(3, _time_left() - 2))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
