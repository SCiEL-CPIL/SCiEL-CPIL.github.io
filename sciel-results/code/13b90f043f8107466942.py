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
    Uses the convex hull facets: the covering radius is the max over facets of arccos(h)
    where h is the distance from origin to the facet plane.
    """
    if len(points) < 3:
        return np.pi
    try:
        hull = ConvexHull(points)
    except:
        return np.pi
    
    max_angle = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Normal to the plane
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # Ensure n points away from origin
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        if angle > max_angle:
            max_angle = angle
            
    return max_angle * 180.0 / np.pi


def _smooth_surrogate(points, temperature=0.05):
    """
    Smooth surrogate for the covering radius using log-sum-exp over facet distances.
    The covering radius is max_facet arccos(h_f). We approximate this with a soft-max.
    Since arccos is decreasing in h, maximizing min h is equivalent to minimizing max arccos(h).
    We use: soft_max(arccos(h_f)) ≈ -log(sum exp(-arccos(h_f)/T)) * T
    But for optimization, it's easier to work with h directly.
    We want to maximize the minimum h. Use soft-min: soft_min(h_f) = T * log(sum exp(-h_f/T))
    Then covering_radius ≈ arccos(soft_min(h_f)).
    """
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
    
    h_values = []
    for simplex in hull.simplices:
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
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        h_values.append(h)
    
    if not h_values:
        return 180.0
    
    h_arr = np.array(h_values)
    # Soft-min of h values: T * log(sum(exp(-h/T)))
    neg_h_over_T = -h_arr / temperature
    # For numerical stability
    max_val = np.max(neg_h_over_T)
    log_sum = max_val + np.log(np.sum(np.exp(neg_h_over_T - max_val)))
    soft_min_h = temperature * log_sum
    
    # Clip to valid range
    soft_min_h = np.clip(soft_min_h, -1.0, 1.0)
    
    return np.arccos(soft_min_h) * 180.0 / np.pi


def _fibonacci_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points():
    """Generate random points on the sphere."""
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.random.uniform(0, np.pi, N)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _icosahedron_based_points():
    """
    Generate points based on icosahedron vertices expanded to 50 points.
    Start with 12 icosahedron vertices and add 38 more by subdividing.
    """
    # Icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    vertices = _normalize(vertices)
    
    # Add more points by finding midpoints of edges and projecting to sphere
    points = [vertices[i] for i in range(12)]
    
    # Find edges (pairs of vertices that are close)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist < 2.0:  # Heuristic for edges
                edges.append((i, j))
    
    # Add midpoints of some edges
    for i, j in edges:
        if len(points) >= N:
            break
        mid = (vertices[i] + vertices[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        # Check if this midpoint is not too close to existing points
        too_close = False
        for p in points:
            if np.linalg.norm(mid - p) < 0.3:
                too_close = True
                break
        if not too_close:
            points.append(mid)
    
    # If we still need more points, add random ones
    while len(points) < N:
        p = _random_points()[0]
        points.append(p)
    
    return np.array(points[:N])


def _spherical_lloyd_iteration(points, max_iter=200, time_limit=60, damping=0.5):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Points are moved toward the centroids of their spherical Voronoi cells.
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
        
        # Build a mapping from point index to list of facet indices
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            # Compute the average of facet normals for this point
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
                
                # Ensure outward normal
                if np.dot(n, p1) < 0:
                    n = -n
                
                normal_sum += n
            
            if np.linalg.norm(normal_sum) > 1e-10:
                # Move point toward the average normal direction
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
            # If cost increased, reduce damping and try again with smaller step
            break
    
    return best_points


def _smooth_gradient_optimize(points, time_limit=30, temperature=0.05):
    """
    Optimize using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _smooth_surrogate(pts, temperature)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _nelder_mead_optimize(points, time_limit=20):
    """
    Nelder-Mead optimization on the true covering radius.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
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
    Refine a single point by moving it to minimize the covering radius.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    p = points[idx].copy()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        new_p = params.reshape(3)
        new_p = new_p / np.linalg.norm(new_p)
        new_points = points.copy()
        new_points[idx] = new_p
        return _covering_radius(new_points)
    
    result = minimize(objective, p, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
    new_p = result.x.reshape(3)
    new_p = new_p / np.linalg.norm(new_p)
    
    new_points = points.copy()
    new_points[idx] = new_p
    return _normalize(new_points)


def _differential_evolution_optimize(time_limit=60):
    """
    Use differential evolution for global optimization.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius(pts)
    
    # Bounds: each coordinate in [-1, 1], but we normalize so it doesn't matter much
    bounds = [(-1.0, 1.0)] * (N * 3)
    
    result = differential_evolution(
        objective, bounds, maxiter=50, popsize=15, tol=1e-8,
        mutation=(0.5, 1.0), recombination=0.7, seed=42,
        polish=False, init='latinhypercube'
    )
    
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0  # Safety margin before 150s kill
    start_time = time.time()
    
    # Try multiple diverse initializations
    initializations = [
        _fibonacci_points(),
        _random_points(),
        _icosahedron_based_points()
    ]
    
    best_points = None
    best_cost = np.inf
    
    for init in initializations:
        if time.time() - start_time > total_time_limit - 30:
            break
        
        # 1. Spherical Lloyd's algorithm
        time_spent = time.time() - start_time
        remaining = total_time_limit - time_spent
        if remaining > 15:
            points_lloyd = _spherical_lloyd_iteration(init, max_iter=100, time_limit=min(remaining - 10, 30), damping=0.5)
            cost_lloyd = _covering_radius(points_lloyd)
            if cost_lloyd < best_cost:
                best_cost = cost_lloyd
                best_points = points_lloyd.copy()
        
        # 2. Smooth gradient optimization
        time_spent = time.time() - start_time
        remaining = total_time_limit - time_spent
        if remaining > 15:
            points_smooth = _smooth_gradient_optimize(points_lloyd, time_limit=min(remaining - 10, 25), temperature=0.05)
            cost_smooth = _covering_radius(points_smooth)
            if cost_smooth < best_cost:
                best_cost = cost_smooth
                best_points = points_smooth.copy()
        
        # 3. Nelder-Mead on true objective
        time_spent = time.time() - start_time
        remaining = total_time_limit - time_spent
        if remaining > 10:
            points_nm = _nelder_mead_optimize(points_smooth, time_limit=min(remaining - 5, 15))
            cost_nm = _covering_radius(points_nm)
            if cost_nm < best_cost:
                best_cost = cost_nm
                best_points = points_nm.copy()
    
    # 4. Simulated Annealing for further improvement
    time_spent = time.time() - start_time
    remaining = total_time_limit - time_spent
    if remaining > 20:
        points_sa = _sa_optimize_covering(best_points, T_start=0.5, T_end=1e-4, n_iter=10000, time_limit=min(remaining - 10, 40))
        cost_sa = _covering_radius(points_sa)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_points = points_sa.copy()
    
    # 5. Per-point refinement
    time_spent = time.time() - start_time
    remaining = total_time_limit - time_spent
    if remaining > 10:
        # Refine each point in turn
        for idx in range(N):
            if time.time() - start_time > total_time_limit - 5:
                break
            points_refined = _refine_point(best_points, idx, time_limit=min(remaining - 5, 5))
            cost_refined = _covering_radius(points_refined)
            if cost_refined < best_cost:
                best_cost = cost_refined
                best_points = points_refined.copy()
    
    # 6. Final Lloyd's polish
    time_spent = time.time() - start_time
    remaining = total_time_limit - time_spent
    if remaining > 5:
        points_final = _spherical_lloyd_iteration(best_points, max_iter=50, time_limit=min(remaining - 2, 10), damping=0.3)
        cost_final = _covering_radius(points_final)
        if cost_final < best_cost:
            best_cost = cost_final
            best_points = points_final.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
