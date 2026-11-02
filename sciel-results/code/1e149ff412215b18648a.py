import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
TIME_LIMIT = 143.0  # safety margin from 150s

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array), worst_idx, vertex_facet_list
    """
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None, None
    
    n_facets = len(hull.simplices)
    if n_facets == 0:
        return hull, np.zeros((0,3)), np.zeros(0), None, [[] for _ in range(N)]
    
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    # Vectorized circumcenter computation
    a = points[hull.simplices[:, 0]]
    b = points[hull.simplices[:, 1]]
    c = points[hull.simplices[:, 2]]
    
    v1 = a - b
    v2 = b - c
    n_vec = np.cross(v1, v2)
    norm_n = np.linalg.norm(n_vec, axis=1)
    
    valid = norm_n > 1e-12
    
    if np.any(valid):
        cc = n_vec[valid] / norm_n[valid, np.newaxis]
        dots = np.sum(cc * a[valid], axis=1)
        cc[dots < 0] *= -1
        centers[valid] = cc
        cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
        radii[valid] = np.arccos(cos_r)
    
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
    
    # Build vertex-facet incidence
    vertex_facet_list = [[] for _ in range(N)]
    for f_idx in range(n_facets):
        for v_idx in hull.simplices[f_idx]:
            vertex_facet_list[v_idx].append(f_idx)
    
    return hull, centers, radii, worst_idx, vertex_facet_list

def _compute_covering_radius(points):
    """
    Compute the actual covering radius (max facet circumradius).
    """
    hull, centers, radii, worst_idx, _ = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    """
    hull, centers, radii, worst_idx, vertex_facet_list = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    shifted = beta * (radii - max_r)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    
    grads = np.zeros_like(points)
    n_facets = len(hull.simplices)
    
    for i in range(n_facets):
        w = weights[i]
        if w < 1e-15:
            continue
        r = radii[i]
        if r < 1e-12:
            continue
        
        C = centers[i]
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            grad_contrib = -C / sin_r
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _grid_covering_radius(points, grid_points=None):
    """
    Compute covering radius using a dense grid of test points on the sphere.
    This provides a smooth, differentiable signal.
    """
    if grid_points is None:
        # Create a reasonably dense grid
        n_theta = 100
        n_phi = 200
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        THETA, PHI = np.meshgrid(theta, phi)
        grid_points = np.column_stack([
            np.sin(THETA) * np.cos(PHI).flatten(),
            np.sin(THETA) * np.sin(PHI).flatten(),
            np.cos(THETA).flatten()
        ])
    
    # Compute angular distance from each grid point to nearest site
    dots = grid_points @ points.T  # (M, N)
    min_dots = np.max(dots, axis=1)  # max dot = min angle
    angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
    return np.max(angles)

def _grid_soft_max_and_grad(points, grid_points, beta=50.0):
    """
    Compute soft-max of angular distances on grid and gradient.
    """
    dots = grid_points @ points.T  # (M, N)
    min_dots = np.max(dots, axis=1)
    angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
    
    max_angle = np.max(angles)
    shifted = beta * (angles - max_angle)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    soft_radius = max_angle + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient: for each grid point, the nearest site gets a contribution
    # d(angle)/d(point_j) where angle = arccos(dot(grid_i, point_j))
    # = -grid_i / sin(angle) * indicator(j is nearest)
    grads = np.zeros_like(points)
    
    # For each grid point, find nearest site
    nearest_idx = np.argmax(dots, axis=1)
    
    for i in range(len(grid_points)):
        w = weights[i]
        if w < 1e-15:
            continue
        j = nearest_idx[i]
        angle = angles[i]
        sin_a = np.sin(angle)
        if sin_a < 1e-10:
            continue
        gp = grid_points[i]
        p = points[j]
        grad_contrib = -gp / sin_a
        radial = np.dot(grad_contrib, p) * p
        tangential = grad_contrib - radial
        grads[j] += w * tangential
    
    return soft_radius, grads

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _facet_targeted_move(points, time_budget, max_iters=2000):
    """
    Directly move vertices of worst facets toward their circumcenters.
    """
    current = points.copy()
    best_radius = _compute_covering_radius(current)
    best_points = current.copy()
    
    start_time = time.time()
    step_size = 0.05
    min_step = 1e-12
    decay = 0.8
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        hull, centers, radii, worst_idx, _ = _compute_facet_data(current)
        if hull is None or worst_idx is None:
            break
        
        current_radius = radii[worst_idx]
        if current_radius < best_radius - 1e-14:
            best_radius = current_radius
            best_points = current.copy()
        
        simplex = hull.simplices[worst_idx]
        C = centers[worst_idx]
        r = radii[worst_idx]
        
        if r < 1e-10:
            break
        
        improved = False
        for k in range(3):
            p_idx = simplex[k]
            p = current[p_idx]
            
            d = C - p
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-12:
                continue
            d = d / d_norm
            
            best_new_radius = current_radius
            best_new_p = p
            
            for scale in [step_size, step_size * 0.5, step_size * 0.25]:
                new_p = p + scale * d
                new_p = new_p / np.linalg.norm(new_p)
                
                test_current = current.copy()
                test_current[p_idx] = new_p
                
                new_radius = _compute_covering_radius(test_current)
                if new_radius < best_new_radius - 1e-14:
                    best_new_radius = new_radius
                    best_new_p = new_p
            
            if best_new_p is not p:
                current[p_idx] = best_new_p
                improved = True
                step_size = min(step_size * 1.1, 0.08)
            else:
                step_size *= decay
        
        if not improved:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _grid_lloyd_iteration(points, grid_points, time_budget, max_iters=300):
    """
    Grid-based Lloyd iteration: for each point, move it toward the centroid
    of grid points for which it is the nearest site, weighted by distance.
    """
    current = points.copy()
    best_radius = _compute_covering_radius(current)
    best_points = current.copy()
    
    start_time = time.time()
    step_size = 0.03
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        dots = grid_points @ current.T
        nearest_idx = np.argmax(dots, axis=1)
        min_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
        
        updates = np.zeros_like(current)
        counts = np.zeros(N)
        
        for i in range(len(grid_points)):
            j = nearest_idx[i]
            weight = angles[i] ** 2  # weight by distance squared
            updates[j] += weight * grid_points[i]
            counts[j] += weight
        
        mask = counts > 1e-10
        if np.any(mask):
            centroids = updates[mask] / counts[mask, np.newaxis]
            # Move toward centroid
            current[mask] += step_size * (centroids - current[mask])
            current = _project_to_sphere(current)
        
        new_radius = _compute_covering_radius(current)
        if new_radius < best_radius - 1e-14:
            best_radius = new_radius
            best_points = current.copy()
        else:
            step_size *= 0.9
            if step_size < 1e-12:
                break
    
    return best_points, best_radius

def _grid_slqp_polish(points, grid_points, time_budget, beta=100.0):
    """
    Use L-BFGS-B to minimize the grid-based soft-max covering radius.
    This provides a smooth surrogate for the true covering radius.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for beta_val in [beta, beta * 2, beta * 4]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
        def objective_and_grad(x):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, grads = _grid_soft_max_and_grad(pts, grid_points, beta=beta_val)
            grad_flat = grads.reshape(-1)
            return soft_r, grad_flat
        
        x0 = best_pts.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 400, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 30}
            )
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            new_radius = _compute_covering_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                best_pts = new_pts
                best_radius = new_radius
        except Exception:
            pass
    
    return best_pts, best_radius

def _repulsion_relax(points, time_budget, max_iters=2000):
    """
    Simple repulsion-based relaxation: each pair of points repels each other,
    and each point is attracted toward the center of its Voronoi cell.
    """
    current = points.copy()
    best_radius = _compute_covering_radius(current)
    best_points = current.copy()
    
    start_time = time.time()
    step_size = 0.01
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        forces = np.zeros_like(current)
        
        for i in range(N):
            p_i = current[i]
            dots = current @ p_i
            dots[i] = -2.0
            nn_j = np.argmin(dots)
            
            nn = current[nn_j]
            away = p_i - nn
            away_norm = np.linalg.norm(away)
            if away_norm > 1e-12:
                away = away / away_norm
                forces[i] += 0.5 * away
            
            dots2 = dots.copy()
            dots2[nn_j] = -2.0
            nn2_j = np.argmin(dots2)
            nn2 = current[nn2_j]
            away2 = p_i - nn2
            away2_norm = np.linalg.norm(away2)
            if away2_norm > 1e-12:
                away2 = away2 / away2_norm
                forces[i] += 0.3 * away2
        
        new_current = current + step_size * forces
        new_current = _project_to_sphere(new_current)
        
        new_radius = _compute_covering_radius(new_current)
        if new_radius < best_radius - 1e-14:
            current = new_current
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = current.copy()
        else:
            step_size *= 0.9
    
    return best_points, best_radius

def _lbfgs_stage(points, time_budget):
    """
    Use L-BFGS-B to minimize the soft-max covering radius (facet-based).
    """
    start_time = time.time()
    best_pts = points.copy()
    best_soft_r = _compute_covering_radius(points)
    
    for beta in [80.0, 150.0, 300.0]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
        def objective_and_grad(x):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _project_to_sphere(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=beta)
            grad_flat = grads.reshape(-1)
            return soft_r, grad_flat
        
        x0 = best_pts.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 600, 'maxfun': 4000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
            )
            if result.fun < best_soft_r - 1e-12:
                best_pts = _project_to_sphere(result.x.reshape(N, 3))
                best_soft_r = _compute_covering_radius(best_pts)
        except Exception:
            pass
    
    return best_pts

def _local_polish(points, time_budget, beta=100.0):
    """
    Final local polish using projected gradient descent on tangent space.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-14
    decay = 0.7
    growth = 1.15
    stagnant = 0
    max_stagnant = 150
    momentum = 0.4
    velocity = np.zeros_like(points)
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-14:
            best_radius = true_radius
            best_points = current_points.copy()
            stagnant = 0
        else:
            stagnant += 1
        
        if stagnant > max_stagnant:
            break
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-16:
            break
        
        velocity = momentum * velocity - (1 - momentum) * grads / grad_norm
        
        best_new_radius = true_radius
        best_new_points = current_points
        
        for trial_scale in [step_size, step_size * 0.5, step_size * 0.25]:
            new_points = current_points + trial_scale * velocity
            new_points = _project_to_sphere(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_new_radius - 1e-16:
                best_new_radius = new_radius
                best_new_points = new_points
        
        if best_new_points is not current_points:
            current_points = best_new_points
            step_size = min(step_size * growth, 0.06)
        else:
            step_size *= decay
            velocity *= 0.5
            if step_size < min_step:
                break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_radius = np.inf
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts with different seeds
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 2024, 31415]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # 3. Icosahedron-based
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(50 - len(ico_verts), 3))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    ico_seed = np.vstack([ico_verts, extra])
    seeds.append(ico_seed)
    
    # Create a grid for grid-based methods
    n_theta = 80
    n_phi = 160
    theta = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi_grid)
    grid_points = np.column_stack([
        np.sin(THETA) * np.cos(PHI).flatten(),
        np.sin(THETA) * np.sin(PHI).flatten(),
        np.cos(THETA).flatten()
    ])
    
    n_seeds = len(seeds)
    
    # Stage 1: Multi-stage optimization for each seed
    stage1_deadline = deadline - 45.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        
        if remaining < 5:
            break
        
        if i < n_seeds - 1:
            time_budget = min(5.0, remaining / (n_seeds - i))
        else:
            time_budget = min(6.0, remaining)
        
        if time_budget < 3.0:
            break
        
        # Apply L-BFGS stage (facet-based)
        pts = _lbfgs_stage(seed_pts, time_budget * 0.6)
        
        # Facet-targeted move
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts, rad = _facet_targeted_move(pts, min(2.0, remaining - 2))
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Additional optimization on best candidates
    if best_points is not None:
        # Grid-based Lloyd iteration
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 30:
            budget = min(12.0, remaining - 25)
            if budget > 4:
                pts, rad = _grid_lloyd_iteration(best_points, grid_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Grid-based SLQP polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 25:
            budget = min(10.0, remaining - 20)
            if budget > 4:
                pts, rad = _grid_slqp_polish(best_points, grid_points, budget, beta=100.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Repulsion relax on best
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 20:
            budget = min(8.0, remaining - 15)
            if budget > 3:
                pts, rad = _repulsion_relax(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Fine local polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            budget = min(10.0, remaining - 12)
            if budget > 4:
                pts, rad = _local_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(7.0, remaining - 8)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 7:
            budget = min(5.0, remaining - 5)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=1000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Final high-precision polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 4:
            budget = min(3.0, remaining - 3)
            if budget > 1.5:
                pts, rad = _local_polish(best_points, budget, beta=2000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized and valid
    best_points = _normalize(best_points)
    
    # Final sanity check
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10):
        best_points = seeds[0]
    
    return best_points
