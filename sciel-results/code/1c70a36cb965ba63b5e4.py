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
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    mask = norm_n[:, 0] > 1e-10
    if not np.any(mask):
        return 180.0
    
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    # Ensure normals point outward
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    h = np.sum(n * p1m, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    return float(np.max(angles)) * 180.0 / np.pi


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _generate_test_grid(n_theta=50, n_phi=100):
    """Generate a dense grid of test points on the sphere."""
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


def _smooth_surrogate(params, beta=300.0, grid=None):
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


def _facet_smooth_surrogate(params, beta=200.0):
    """
    Smooth surrogate based on convex hull facet circumradii (vectorized).
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
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    mask = norm_n[:, 0] > 1e-10
    if not np.any(mask):
        return np.pi
    
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    h = np.sum(n * p1m, axis=1)
    h = np.clip(h, -1.0, 1.0)
    facet_angles = np.arccos(h)
    
    if len(facet_angles) == 0:
        return np.pi
    
    # Soft-max over facet angles
    beta_facet = beta * facet_angles
    max_val = np.max(beta_facet)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_facet - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _epigraph_objective(params, grid=None, beta=200.0):
    """
    Epigraph formulation: minimize t subject to soft_max <= t.
    We use a penalty approach: t + penalty * max(0, soft_max - t)^2
    """
    pts = params[:N*3].reshape(N, 3)
    t = params[N*3]
    
    if grid is None:
        grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    pts = _normalize(pts)
    
    # Compute angular distances
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
    
    # Penalty for constraint violation
    violation = np.maximum(0.0, soft_max - t)
    penalty = 1000.0 * violation ** 2
    
    return t + penalty


def _smooth_gradient_optimize(points, time_limit=25, beta=300.0, use_facet=False, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    if use_facet:
        grid = None
    elif grid is None:
        grid = _generate_test_grid(n_theta=50, n_phi=100)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        if use_facet:
            return _facet_smooth_surrogate(params, beta=beta)
        else:
            return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 10000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _epigraph_optimize(points, time_limit=20, grid=None, beta=200.0):
    """
    Optimize using epigraph formulation with auxiliary variable t.
    """
    start_time = time.time()
    
    if grid is None:
        grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    # Initial t estimate
    t0 = _covering_radius_radians(points) + 0.05
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        return _epigraph_objective(params, grid=grid, beta=beta)
    
    x0 = np.concatenate([points.flatten(), [t0]])
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000})
    best_points = result.x[:N*3].reshape(N, 3)
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


def _golden_spiral_points(offset=0.0):
    """Generate golden spiral points with a phase offset for diversity."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _perturbed_fibonacci(seed=0, jitter=0.08):
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


def _spherical_lloyd_iteration(points, max_iter=150, time_limit=30, damping=0.4):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) - vectorized version.
    For each point, compute the average of outward normals of adjacent facets,
    then move the point toward that direction.
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
        
        simplices = hull.simplices
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1, keepdims=True)
        mask = norm_n[:, 0] > 1e-10
        if not np.any(mask):
            break
        
        n_valid = n[mask] / norm_n[mask]
        p1_valid = p1[mask]
        simplices_valid = simplices[mask]
        
        # Ensure normals point outward
        dots = np.sum(n_valid * p1_valid, axis=1)
        n_valid[dots < 0] *= -1
        
        # Accumulate normal contributions for each point
        normal_sum = np.zeros((N, 3))
        
        # For each facet, add its normal to all three vertices
        for fi in range(len(simplices_valid)):
            fi_idx = fi  # index into valid arrays
            s = simplices_valid[fi_idx]
            nv = n_valid[fi_idx]
            normal_sum[s[0]] += nv
            normal_sum[s[1]] += nv
            normal_sum[s[2]] += nv
        
        # Compute new positions
        norms_ns = np.linalg.norm(normal_sum, axis=1, keepdims=True)
        valid_mask = norms_ns[:, 0] > 1e-10
        
        new_points = points.copy()
        for pi in range(N):
            if valid_mask[pi]:
                target = normal_sum[pi] / norms_ns[pi, 0]
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
            # Try smaller damping
            if i > 20:
                damping *= 0.9
            else:
                break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=15):
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
                      options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13, 'maxfev': 8000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _coordinate_wise_nm(points, time_limit=15, tol=1e-11):
    """
    Coordinate-wise Nelder-Mead polish: optimize each point's 3 coordinates
    one at a time with very tight tolerance. This extracts gains from
    near-converged configurations by removing numerical noise.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    
    for round_num in range(4):  # Multiple passes
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
                          options={'maxiter': 80, 'xatol': tol, 'fatol': tol, 'maxfev': 300})
            new_pt = res.x / np.linalg.norm(res.x)
            points[idx] = new_pt
    
    return _normalize(points)


def _facet_shrink_optimize(points, time_limit=15, beta=500.0):
    """
    Focus specifically on the worst facet and try to shrink it.
    Identify the facet with maximum circumradius, then optimize the three
    vertices of that facet while keeping others fixed.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    
    for iteration in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Find worst facet
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
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1, keepdims=True)
        mask = norm_n[:, 0] > 1e-10
        if not np.any(mask):
            break
        
        n_valid = n[mask] / norm_n[mask]
        p1_valid = p1[mask]
        
        dots = np.sum(n_valid * p1_valid, axis=1)
        n_valid[dots < 0] *= -1
        
        h = np.sum(n_valid * p1_valid, axis=1)
        h = np.clip(h, -1.0, 1.0)
        facet_angles = np.arccos(h)
        
        worst_idx = np.argmax(facet_angles)
        worst_facet = simplices[mask][worst_idx]
        worst_indices = worst_facet.tolist()
        
        # Optimize the three vertices of the worst facet
        other_points = np.delete(points, worst_indices, axis=0)
        facet_points = points[worst_indices].copy()
        
        def obj_facet(params):
            fp = params.reshape(3, 3)
            fp = _normalize(fp)
            all_pts = np.vstack([other_points, fp])
            return _covering_radius_radians(all_pts)
        
        x0 = facet_points.flatten()
        res = minimize(obj_facet, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 500})
        
        new_facet_points = _normalize(res.x.reshape(3, 3))
        points[worst_indices] = new_facet_points
    
    return _normalize(points)


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
    
    # Jittered Fibonacci - breaks symmetry
    for seed in [123, 456, 789]:
        initializations.append(_jittered_fibonacci_points(seed=seed, jitter=0.05))
    
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
    
    # Allocate time for initialization phase: ~45% of total
    init_time_budget = 0.45 * total_time_limit
    lloyd_time_each = max(4.0, (init_time_budget - 5.0) / len(initializations))
    
    for init_pts in initializations:
        if _time_left() < 55:
            break
        
        # Lloyd's relaxation
        pts = _spherical_lloyd_iteration(init_pts, max_iter=100, 
                                          time_limit=min(lloyd_time_each, _time_left() - 50), 
                                          damping=0.4)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # High-beta facet-based smooth gradient optimization
    if _time_left() > 45:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(20, _time_left() - 35), 
                                         beta=300.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Grid-based smooth gradient with high beta
    if _time_left() > 35:
        grid = _generate_test_grid(n_theta=50, n_phi=100)
        pts = _smooth_gradient_optimize(best_points, time_limit=min(15, _time_left() - 30), 
                                         beta=400.0, use_facet=False, grid=grid)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Epigraph optimization
    if _time_left() > 25:
        pts = _epigraph_optimize(best_points, time_limit=min(12, _time_left() - 20), 
                                  beta=200.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Nelder-Mead on true objective (high-precision polish)
    if _time_left() > 18:
        pts = _nelder_mead_optimize(best_points, time_limit=min(10, _time_left() - 15))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Facet shrink optimization
    if _time_left() > 12:
        pts = _facet_shrink_optimize(best_points, time_limit=min(8, _time_left() - 10), 
                                      beta=500.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Coordinate-wise Nelder-Mead polish (extracts gains from near-converged configs)
    if _time_left() > 10:
        pts = _coordinate_wise_nm(best_points, time_limit=min(7, _time_left() - 8), 
                                   tol=1e-12)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final Lloyd's polish with small damping
    if _time_left() > 7:
        pts = _spherical_lloyd_iteration(best_points, max_iter=40, 
                                          time_limit=min(4, _time_left() - 5), damping=0.1)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final high-precision coordinate-wise polish
    if _time_left() > 5:
        pts = _coordinate_wise_nm(best_points, time_limit=min(3, _time_left() - 3), 
                                   tol=1e-14)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
