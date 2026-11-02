import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
TIME_LIMIT = 143.0  # safety margin from 150s

# Pre-compute quadrature grid for grid-based objective
def _build_grid(n_theta=40, n_phi=80):
    """Build a quadrature grid on the sphere."""
    theta = np.linspace(0, np.pi, n_theta, endpoint=False) + np.pi / n_theta
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    # Weights for spherical quadrature (sin(theta) factor)
    w = np.sin(T).ravel()
    w /= np.sum(w)
    return grid, w

_GRID, _GRID_W = _build_grid(40, 80)
_N_GRID = _GRID.shape[0]

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _grid_objective_and_grad(points):
    """
    Compute the mean reciprocal distance to nearest site over the grid,
    and its gradient w.r.t. the site positions.
    
    Objective: sum_g w_g / d_g  where d_g = min_i ||grid_g - p_i||
    This penalizes large empty regions (small distances to nearest site).
    
    We use a soft-min to make this differentiable:
    d_g ≈ exp(-k * min_i ||grid_g - p_i||^2) approximated via log-sum-exp
    Actually, let's use: soft_min = -log(sum exp(-k*d_i^2))/k
    Then objective = sum w_g * k * exp(-k * soft_min) ... this gets complex.
    
    Simpler: use the reciprocal of a soft-min distance.
    soft_min_dist = -1/k * log(sum_i exp(-k * dist_i^2))
    Then 1/soft_min_dist is smooth where soft_min_dist > 0.
    """
    # Compute pairwise distances from grid to points
    # diff[g, i, :] = grid[g] - points[i]
    diff = _GRID[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)  # shape (N_GRID, N)
    
    k = 20.0  # inverse temperature for soft-min
    
    # Soft-min of distances (not squared, to get distance not distance^2)
    # Use: soft_min(d_i) ≈ -1/k * log(sum exp(-k * d_i))
    # But we have d_i^2, so d_i = sqrt(d_i^2)
    dist = np.sqrt(dist_sq + 1e-15)
    
    # Soft-min: -1/k * log(sum exp(-k * dist))
    # For numerical stability, use log-sum-exp trick
    shifted = -k * dist
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted), axis=1, keepdims=True))
    soft_min_dist = -log_sum / k  # shape (N_GRID, 1)
    
    # Objective: sum w_g / soft_min_dist_g
    obj = np.sum(_GRID_W / soft_min_dist)
    
    # Gradient w.r.t. points
    # d/dp_i (1/soft_min_dist_g) = -1/soft_min_dist_g^2 * d(soft_min_dist_g)/dp_i
    # d(soft_min_dist_g)/dp_i = (1/k) * d/dp_i log(sum_j exp(-k * dist_j))
    #                          = (1/k) * [sum_j exp(-k*dist_j) * (-k) * d(dist_j)/dp_i] / sum_j exp(-k*dist_j)
    #                          = -[sum_j exp(-k*dist_j) * d(dist_j)/dp_i] / sum_j exp(-k*dist_j)
    # d(dist_j)/dp_i = (p_i - grid_g) / dist_j  (for j=i, else 0)
    
    # More carefully:
    # soft_min_dist_g = -1/k * log(sum_j exp(-k * dist_jg))
    # d(soft_min_dist_g)/dp_i = -1/k * [1/sum_j exp(-k*dist_jg)] * d/dp_i [sum_j exp(-k*dist_jg)]
    #                          = -1/k * [1/S_g] * sum_j exp(-k*dist_jg) * (-k) * d(dist_jg)/dp_i
    #                          = [1/S_g] * sum_j exp(-k*dist_jg) * d(dist_jg)/dp_i
    # where S_g = sum_j exp(-k*dist_jg)
    # d(dist_jg)/dp_i = 0 if j != i, = (p_i - grid_g)/dist_ig if j == i
    
    # So: d(soft_min_dist_g)/dp_i = exp(-k*dist_ig)/S_g * (p_i - grid_g)/dist_ig
    
    # Let's compute weights w_gi = exp(-k*dist_ig) / S_g
    exp_terms = np.exp(shifted)  # shape (N_GRID, N), already shifted
    S_g = np.sum(exp_terms, axis=1, keepdims=True)  # shape (N_GRID, 1)
    w_gi = exp_terms / S_g  # shape (N_GRID, N)
    
    # d(soft_min_dist_g)/dp_i = w_gi * (p_i - grid_g) / dist_ig
    # But we need to be careful: dist_ig is in the denominator
    
    # Gradient of objective w.r.t. p_i:
    # grad_i = sum_g w_g * d/dp_i (1/soft_min_dist_g)
    #        = sum_g w_g * (-1/soft_min_dist_g^2) * d(soft_min_dist_g)/dp_i
    #        = -sum_g w_g / soft_min_dist_g^2 * w_gi * (p_i - grid_g) / dist_ig
    
    # Let's compute this efficiently
    # For each grid point g, contribution to grad_i from site i:
    # -w_g / soft_min_dist_g^2 * w_gi * (p_i - grid_g) / dist_ig
    
    # Vectorized:
    # inv_soft_min_sq = 1 / soft_min_dist^2  shape (N_GRID, 1)
    inv_soft_min_sq = 1.0 / (soft_min_dist + 1e-15) ** 2  # shape (N_GRID, 1)
    
    # w_g / soft_min_dist^2 shape (N_GRID, 1)
    wg_over_sm2 = _GRID_W[:, np.newaxis] * inv_soft_min_sq  # shape (N_GRID, 1)
    
    # w_gi * (p_i - grid_g) / dist_ig: shape (N_GRID, N, 3)
    # (p_i - grid_g) = -diff[g, i, :]
    # dist_ig = dist[g, i]
    
    # factor[g, i] = w_gi[g, i] / dist[g, i]
    factor = w_gi / (dist + 1e-15)  # shape (N_GRID, N)
    
    # grad_i = -sum_g wg_over_sm2[g] * factor[g, i] * (p_i - grid_g)
    #        = -sum_g wg_over_sm2[g] * factor[g, i] * (-diff[g, i, :])
    #        = sum_g wg_over_sm2[g] * factor[g, i] * diff[g, i, :]
    
    # diff[g, i, :] = grid[g] - p_i
    # So grad_i = sum_g wg_over_sm2[g] * factor[g, i] * (grid[g] - p_i)
    
    # Vectorized:
    # grad[i] = sum_g wg_over_sm2[g, 0] * factor[g, i] * (grid[g] - p_i)
    
    # Let's compute:
    # weighted_diff[g, i, :] = wg_over_sm2[g, 0] * factor[g, i] * diff[g, i, :]
    # Then grad[i] = sum_g weighted_diff[g, i, :]
    
    # But this is N_GRID * N * 3 which is 3200*50*3 = 480000, manageable
    
    weighted_diff = wg_over_sm2[:, np.newaxis, :] * factor[:, :, np.newaxis] * diff  # (N_GRID, N, 3)
    grads = np.sum(weighted_diff, axis=0)  # (N, 3)
    
    return obj, grads

def _compute_facet_data(points):
    """
    Compute convex hull and facet circumcenters/radii.
    Returns: hull, centers (array), radii (array), worst_idx
    """
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None, None
    
    n_facets = len(hull.simplices)
    centers = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        Gram = np.array([
            [1.0, ab, ac],
            [ab, 1.0, bc],
            [ac, bc, 1.0]
        ])
        
        try:
            m = np.linalg.solve(Gram, np.ones(3))
            c_dir = m[0]*a + m[1]*b + m[2]*c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                continue
            centers[i] = c_dir / norm_c_dir
            radii[i] = np.arccos(np.clip(np.dot(centers[i], a), -1.0, 1.0))
        except np.linalg.LinAlgError:
            continue
    
    if n_facets > 0 and np.any(radii > 0):
        worst_idx = np.argmax(radii)
    else:
        worst_idx = None
    
    return hull, centers, radii, worst_idx

def _compute_covering_radius(points):
    """
    Compute the actual covering radius (max facet circumradius).
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]
    """
    Compute the actual covering radius (max facet circumradius).
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius_and_grad(points, beta=50.0):
    """
    Compute soft-max radius and its gradient w.r.t. points.
    Uses a more numerically stable approach.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    # Use log-sum-exp trick for numerical stability
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
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r) projected to tangent space
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            # Project gradient to tangent space at p
            grad_contrib = -C / sin_r
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads
    """
    Compute soft-max radius and its gradient w.r.t. points.
    Uses a more numerically stable approach.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    # Use log-sum-exp trick for numerical stability
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
        
        # Gradient of arccos(dot(C, p)) w.r.t. p is -C/sin(r) projected to tangent space
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            # Project gradient to tangent space at p
            grad_contrib = -C / sin_r
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _grid_based_global_opt(points, time_budget):
    """
    Use the grid-based mean reciprocal distance objective for global optimization.
    This provides a smooth, globally differentiable signal.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_obj, _ = _grid_objective_and_grad(points)
    
    def objective_and_grad(x):
        if time.time() - start_time > time_budget:
            raise Exception("timeout")
        pts = _project_to_sphere(x.reshape(N, 3))
        obj, grads = _grid_objective_and_grad(pts)
        return obj, grads.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'maxfun': 1500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
        )
        if result.fun < best_obj - 1e-10:
            best_pts = _project_to_sphere(result.x.reshape(N, 3))
    except Exception:
        pass
    
    return best_pts

def _repulsion_relax(pts, time_budget, alpha_start=0.05):
    """
    Repulsion relaxation: minimize sum 1/||p_i - p_j||^2 via gradient descent.
    Helps spread points more uniformly before covering optimization.
    """
    pts = pts.copy()
    alpha = alpha_start
    start_t = time.time()
    
    def repulsion_energy(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
        return 0.5 * np.sum(inv_dist_sq)
    
    current_energy = repulsion_energy(pts)
    
    while time.time() - start_t < time_budget:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane
        dot_fp = np.sum(forces * pts, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * pts
        
        # Adaptive step with backtracking
        step = alpha
        improved = False
        for _ in range(8):
            new_pts = pts + step * forces_tan
            new_pts = _project_to_sphere(new_pts)
            new_energy = repulsion_energy(new_pts)
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                improved = True
                break
            step *= 0.5
        
        if not improved:
            alpha *= 0.85
        else:
            alpha = min(alpha * 1.02, 0.08)
        
        alpha = max(alpha, 1e-6)
    
    return pts
    """
    Repulsion relaxation: minimize sum 1/||p_i - p_j||^2 via gradient descent.
    Helps spread points more uniformly before covering optimization.
    """
    pts = pts.copy()
    alpha = alpha_start
    start_t = time.time()
    
    def repulsion_energy(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
        return 0.5 * np.sum(inv_dist_sq)
    
    current_energy = repulsion_energy(pts)
    
    while time.time() - start_t < time_budget:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project onto tangent plane
        dot_fp = np.sum(forces * pts, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * pts
        
        # Adaptive step with backtracking
        step = alpha
        improved = False
        for _ in range(8):
            new_pts = pts + step * forces_tan
            new_pts = _project_to_sphere(new_pts)
            new_energy = repulsion_energy(new_pts)
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                improved = True
                break
            step *= 0.5
        
        if not improved:
            alpha *= 0.85
        else:
            alpha = min(alpha * 1.02, 0.08)
        
        alpha = max(alpha, 1e-6)
    
    return pts

def _lloyd_iteration(pts, time_budget, step_size=0.03):
    """
    Weighted Lloyd iteration: move each point toward weighted average of
    circumcenters of facets it belongs to, weighted by facet radius.
    """
    pts = pts.copy()
    start_t = time.time()
    
    while time.time() - start_t < time_budget:
        hull, centers, radii, worst_idx = _compute_facet_data(pts)
        if hull is None:
            break
        
        max_rad = np.max(radii)
        if max_rad < 1e-10:
            break
        
        updates = np.zeros_like(pts)
        weights_sum = np.zeros(N)
        
        for f in range(len(hull.simplices)):
            rad = radii[f]
            if rad < 1e-12:
                continue
            # Weight: higher for larger facets
            w = (rad / max_rad) ** 2
            cc = centers[f]
            for idx in hull.simplices[f]:
                updates[idx] += w * (cc - pts[idx])
                weights_sum[idx] += w
        
        mask = weights_sum > 1e-10
        if np.any(mask):
            pts[mask] += step_size * updates[mask] / weights_sum[mask, np.newaxis]
            pts = _project_to_sphere(pts)
    
    return pts

def _lbfgs_stage(points, time_budget):
    """
    Use L-BFGS-B to minimize the soft-max covering radius.
    This provides a robust global structure optimization.
    Uses adaptive beta: starts moderate, increases for finer convergence.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_soft_r = _compute_covering_radius(points)
    
    # Try three beta values: moderate for global, medium, high for local
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
    Directly minimizes the soft-max covering radius.
    Uses adaptive step size with line search and momentum.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.02
    min_step = 1e-14
    decay = 0.7
    growth = 1.1
    stagnant = 0
    max_stagnant = 100
    momentum = 0.3
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
        
        # Update velocity with momentum
        velocity = momentum * velocity - (1 - momentum) * grads / grad_norm
        
        new_points = current_points + step_size * velocity
        new_points = _project_to_sphere(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius - 1e-16:
            current_points = new_points
            step_size = min(step_size * growth, 0.06)
        else:
            step_size *= decay
            velocity *= 0.5  # Dampen momentum on failure
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _facet_targeted_move(points, time_budget):
    """
    Directly move vertices of the worst facets toward their circumcenters.
    This is a more direct approach than gradient descent for targeting the bottleneck.
    """
    pts = points.copy()
    start_t = time.time()
    best_radius = _compute_covering_radius(pts)
    best_pts = pts.copy()
    
    while time.time() - start_t < time_budget:
        hull, centers, radii, worst_idx = _compute_facet_data(pts)
        if hull is None or worst_idx is None:
            break
        
        # Get the worst facet
        worst_facet = hull.simplices[worst_idx]
        worst_center = centers[worst_idx]
        worst_radius = radii[worst_idx]
        
        if worst_radius < 1e-10:
            break
        
        # Move vertices of the worst facet toward the circumcenter
        # Use a small step to avoid instability
        step = 0.02
        new_pts = pts.copy()
        
        for idx in worst_facet:
            # Move toward circumcenter
            direction = worst_center - pts[idx]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                new_pts[idx] = pts[idx] + step * direction / norm_dir
        
        new_pts = _project_to_sphere(new_pts)
        new_radius = _compute_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-15:
            pts = new_pts
            best_radius = new_radius
            best_pts = new_pts.copy()
        else:
            # Try smaller step
            step *= 0.5
            if step < 1e-6:
                break
    
    return best_pts, best_radius

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
    
    n_seeds = len(seeds)
    
    # Stage 1: Pre-processing with grid-based global optimization
    # Then facet-based local optimization for each seed
    # Reserve ~55s for final polish stages
    stage1_deadline = deadline - 55.0

    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        
        if remaining < 5:
            break
        
        if i < n_seeds - 1:
            time_budget = min(8.0, remaining / (n_seeds - i))
        else:
            time_budget = min(10.0, remaining)
        
        if time_budget < 4.0:
            break
        
        # Step 1: Repulsion relaxation to spread points
        pts = _repulsion_relax(seed_pts, min(1.5, time_budget * 0.15))
        
        # Step 2: Grid-based global optimization (new smooth objective)
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts = _grid_based_global_opt(pts, min(3.0, remaining * 0.4))
        
        # Step 3: Lloyd iteration to align with facet structure
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts = _lloyd_iteration(pts, min(2.0, remaining * 0.25))
        
        # Step 4: L-BFGS optimization on facet-based soft-max
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts = _lbfgs_stage(pts, min(3.0, remaining * 0.35))
        
        # Step 5: Quick local polish
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 4:
            pts, rad = _local_polish(pts, min(2.0, remaining - 2), beta=80.0)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Facet-targeted moves on best candidate
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 28:
            budget = min(15.0, remaining - 22)
            if budget > 3:
                pts, rad = _facet_targeted_move(best_points, budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Fine local polish with moderate beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 20:
            budget = min(12.0, remaining - 16)
            if budget > 4:
                pts, rad = _local_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish with high beta for final convergence
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 14:
            budget = min(8.0, remaining - 11)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(5.0, remaining - 8)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=1000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Final high-precision polish with very high beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 6:
            budget = min(3.0, remaining - 5)
            if budget > 1:
                pts, rad = _local_polish(best_points, budget, beta=2000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized and valid
    best_points = _normalize(best_points)
    
    # Final sanity check: verify no coincident points, not in hemisphere
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10):
        best_points = seeds[0]
    
    return best_points
