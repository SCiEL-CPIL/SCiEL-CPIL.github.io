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


def _generate_test_grid(n_theta=40, n_phi=80):
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
        grid = _generate_test_grid(n_theta=40, n_phi=80)
    
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


def _riesz_energy(points, s=20.0):
    """
    Compute Riesz s-energy: sum over pairs of 1/|pi-pj|^s.
    Minimizing this pushes points apart, maximizing minimum separation.
    """
    pts = _normalize(points)
    n = len(pts)
    # Compute pairwise distances
    # dist[i,j] = |pi - pj|
    # Use dot product: |pi-pj|^2 = 2 - 2*pi.pj
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy = sum_{i!=j} 1/dist(i,j)^s
    # For large s, this is dominated by the closest pairs
    energy = np.sum(1.0 / (dist ** s)) - n  # subtract diagonal terms (1/1^s = 1)
    return energy


def _riesz_energy_grad(points, s=20.0):
    """
    Compute gradient of Riesz s-energy with respect to points on the sphere.
    We project the gradient to the tangent space.
    """
    pts = _normalize(points).copy()
    n = len(pts)
    
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Gradient of energy w.r.t. points
    # dE/dpi = sum_{j!=i} s * (pi - pj) / |pi - pj|^(s+2)
    # But we need to be on the sphere, so we project to tangent space.
    
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dij = dist[i, j]
            if dij < 1e-10:
                continue
            # d(1/dij^s)/d(pi) = -s * (pi - pj) / dij^(s+2)
            # But energy = sum 1/dij^s, so dE/dpi = sum_j -s*(pi-pj)/dij^(s+2)
            # Wait, let me re-derive.
            # E = sum_{i<j} 2/dij^s (symmetric)
            # Or E = sum_{i!=j} 1/dij^s
            # dE/dpi = sum_{j!=i} d(1/|pi-pj|^s)/d(pi)
            # = sum_{j!=i} -s * (pi - pj) / |pi-pj|^(s+2)
            
            coeff = -s / (dij ** (s + 2))
            grad[i] += coeff * (pts[i] - pts[j])
    
    # Project gradient to tangent space: grad_tangent = grad - (grad.p) * p
    for i in range(n):
        proj = np.dot(grad[i], pts[i])
        grad[i] -= proj * pts[i]
    
    return grad


def _riesz_optimize(points, s=20.0, time_limit=20, maxiter=500):
    """
    Optimize points to minimize Riesz s-energy using L-BFGS-B.
    This maximizes minimum pairwise separation.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10
        pts = params.reshape(N, 3)
        return _riesz_energy(pts, s=s)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


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
            p1v[flip] = p1v[flip]  # p1v stays the same, normal flips
        
        h = np.sum(normals * p1v, axis=1)
        h = np.clip(h, -1.0, 1.0)
        angles = np.arccos(h)
        
        # Map facets to points
        # Each facet has 3 points; accumulate weighted normals for each point
        new_points = points.copy()
        
        # For each valid facet, get its 3 point indices
        valid_indices = np.where(valid)[0]
        facet_pts = simplices[valid_indices]  # (n_valid, 3)
        
        # Accumulate for each point
        point_normal_sum = np.zeros((N, 3))
        point_weight_sum = np.zeros(N)
        
        for k in range(len(valid_indices)):
            fi = valid_indices[k]
            w = angles[fi]
            n_vec = normals[fi]
            for pi_idx in facet_pts[fi]:
                point_normal_sum[pi_idx] += w * n_vec
                point_weight_sum[pi_idx] += w
        
        for pi in range(N):
            if point_weight_sum[pi] > 1e-10:
                target = point_normal_sum[pi]
                target_norm = np.linalg.norm(target)
                if target_norm > 1e-10:
                    target = target / target_norm
                    new_points[pi] = points[pi] + current_damping * (target - points[pi])
                    new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
            current_damping = min(0.8, current_damping * 1.1)
        else:
            current_damping *= 0.5
            if current_damping < 0.01:
                break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=20, maxiter=1000):
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
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10})
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
                      options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9})
    new_pt = result.x / np.linalg.norm(result.x)
    points[idx] = new_pt
    return points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Precompute grid for smooth surrogate
    grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    best_points = None
    best_cost = np.inf
    
    # Stage 1: Riesz energy optimization to get well-spread initial points
    # This is the structural change: use packing/repulsion to find good basins
    initializations = []
    
    # Fibonacci spiral as baseline
    initializations.append(_fibonacci_points())
    
    # Riesz energy optimized from multiple starts
    for seed in [42, 123, 777, 999, 555]:
        if _time_left() < 80:
            break
        init = _random_points(seed=seed)
        # Optimize with Riesz energy (high s for strong repulsion)
        pts_riesz = _riesz_optimize(init, s=20.0, time_limit=min(15, _time_left() - 60))
        initializations.append(pts_riesz)
    
    # Also try Riesz from Fibonacci
    if _time_left() > 70:
        pts_riesz_fib = _riesz_optimize(_fibonacci_points(), s=20.0, time_limit=min(15, _time_left() - 55))
        initializations.append(pts_riesz_fib)
    
    # Jittered Fibonacci
    initializations.append(_jittered_fibonacci_points(seed=123, jitter=0.05))
    initializations.append(_jittered_fibonacci_points(seed=456, jitter=0.1))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    # Stage 2: For each initialization, do Lloyd's + smooth gradient + Nelder-Mead
    for init_pts in initializations:
        if _time_left() < 30:
            break
        
        # Lloyd's relaxation
        pts = _spherical_lloyd_iteration(init_pts, max_iter=40, time_limit=min(12, _time_left() - 25), damping=0.3)
        
        # Smooth gradient optimization
        if _time_left() > 15:
            pts = _smooth_gradient_optimize(pts, time_limit=min(10, _time_left() - 12), beta=100.0, use_facet=True)
        
        # Nelder-Mead polish
        if _time_left() > 8:
            pts = _nelder_mead_optimize(pts, time_limit=min(6, _time_left() - 5), maxiter=500)
        
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Stage 3: High-precision polish on the best
    if _time_left() > 20:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(12, _time_left() - 15), beta=500.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Stage 4: Final Nelder-Mead
    if _time_left() > 10:
        pts = _nelder_mead_optimize(best_points, time_limit=min(8, _time_left() - 5), maxiter=800)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Stage 5: Simulated Annealing if time permits
    if _time_left() > 25:
        pts = _sa_optimize_covering(best_points, T_start=0.2, T_end=1e-4, n_iter=3000, time_limit=min(15, _time_left() - 10))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
            
        # Quick polish after SA
        if _time_left() > 8 and cost < best_cost:
            pts = _nelder_mead_optimize(best_points, time_limit=min(5, _time_left() - 3), maxiter=300)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
