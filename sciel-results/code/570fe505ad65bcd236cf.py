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

def _soft_max_radius(points, beta=50.0):
    """
    Compute soft-max of facet radii for smooth optimization.
    """
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi, None
    max_r = np.max(radii)
    exp_terms = np.exp(beta * (radii - max_r))
    log_sum = np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)
    return soft_radius, weights

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

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

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
        
        # Update velocity with momentum
        velocity = momentum * velocity - (1 - momentum) * grads / grad_norm
        
        # Try multiple step sizes for better line search
        best_new_radius = true_radius
        best_new_points = current_points
        for trial_scale in [1.0, 0.5, 0.25, 0.1]:
            trial_step = step_size * trial_scale
            new_points = current_points + trial_step * velocity
            new_points = _project_to_sphere(new_points)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_new_radius:
                best_new_radius = new_radius
                best_new_points = new_points
        
        if best_new_radius < true_radius - 1e-16:
            current_points = best_new_points
            step_size = min(step_size * growth, 0.08)
        else:
            step_size *= decay
            velocity *= 0.5  # Dampen momentum on failure
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _repulsion_relax(pts, time_budget, alpha_start=0.05):
    """
    Repulsion relaxation: minimize sum 1/||p_i - p_j||^2 via gradient descent.
    This helps spread points more evenly and escape poor initial configurations.
    """
    start_time = time.time()
    current = pts.copy()
    best = pts.copy()
    best_score = np.sum(1.0 / np.maximum(np.sum((current[:, None, :] - current[None, :, :])**2, axis=-1), 1e-10))
    
    alpha = alpha_start
    stagnant = 0
    
    while time.time() - start_time < time_budget:
        # Compute pairwise distances
        diff = current[:, None, :] - current[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-10)
        
        # Gradient of sum 1/dist_sq w.r.t. each point
        # d(1/||p_i - p_j||^2)/d(p_i) = 2*(p_i - p_j) / ||p_i - p_j||^4
        inv_dist4 = 1.0 / (dist_sq**2)
        grad = np.zeros_like(current)
        for i in range(N):
            for j in range(N):
                if i != j:
                    grad[i] += 2.0 * (current[i] - current[j]) * inv_dist4[i, j]
        
        # Project to tangent space
        for i in range(N):
            radial = np.dot(grad[i], current[i]) * current[i]
            grad[i] -= radial
        
        # Update
        new_pts = current + alpha * grad
        new_pts = _normalize(new_pts)
        
        # Check improvement
        new_score = np.sum(1.0 / np.maximum(np.sum((new_pts[:, None, :] - new_pts[None, :, :])**2, axis=-1), 1e-10))
        if new_score < best_score:
            best = new_pts.copy()
            best_score = new_score
            current = new_pts
            stagnant = 0
            alpha = min(alpha * 1.1, 0.1)
        else:
            stagnant += 1
            alpha *= 0.7
            if stagnant > 50 or alpha < 1e-6:
                break
    
    return best

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
    
    # 3. Structured seeds: golden spiral variations
    for offset in [0.0, 0.3, 0.7]:
        j = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * j / N
        z = np.clip(z, -1.0, 1.0)
        phi = np.pi * (1.0 + np.sqrt(5.0)) * j
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        spiral = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        seeds.append(spiral)
    
    # 4. Cube-based seeds: distribute on cube faces
    rng_cube = np.random.default_rng(42)
    cube_pts = []
    for face in range(6):
        n_per_face = N // 6 + (1 if face < N % 6 else 0)
        for _ in range(n_per_face):
            u = rng_cube.uniform(-1, 1)
            v = rng_cube.uniform(-1, 1)
            if face == 0:
                p = np.array([1.0, u, v])
            elif face == 1:
                p = np.array([-1.0, u, v])
            elif face == 2:
                p = np.array([u, 1.0, v])
            elif face == 3:
                p = np.array([u, -1.0, v])
            elif face == 4:
                p = np.array([u, v, 1.0])
            else:
                p = np.array([u, v, -1.0])
            cube_pts.append(p)
    cube_pts = np.array(cube_pts[:N])
    cube_pts /= np.linalg.norm(cube_pts, axis=1, keepdims=True)
    seeds.append(cube_pts)
    
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
    
    # Stage 1: L-BFGS optimization for each seed
    # Reserve ~35s for final polish stages
    stage1_deadline = deadline - 35.0
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        
        if remaining < 4:
            break
        
        if i < n_seeds - 1:
            time_budget = min(5.0, remaining / (n_seeds - i))
        else:
            time_budget = min(7.0, remaining)
        
        if time_budget < 2.5:
            break
        
        # Apply L-BFGS stage
        pts = _lbfgs_stage(seed_pts, time_budget)
        
        # Quick local polish
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 3:
            pts, rad = _local_polish(pts, min(2.5, remaining - 2), beta=80.0)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Repulsion relaxation to improve spread before final polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 25:
            budget = min(8.0, remaining - 20)
            if budget > 3:
                relaxed = _repulsion_relax(best_points, budget, alpha_start=0.02)
                rad_relaxed = _compute_covering_radius(relaxed)
                if rad_relaxed < best_radius:
                    best_radius = rad_relaxed
                    best_points = relaxed.copy()
        
        # Fine local polish with moderate beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 20:
            budget = min(12.0, remaining - 15)
            if budget > 4:
                pts, rad = _local_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish with high beta for final convergence
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 14:
            budget = min(8.0, remaining - 10)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(6.0, remaining - 7)
            if budget > 2:
                pts, rad = _local_polish(best_points, budget, beta=1000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Final high-precision polish with very high beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 7:
            budget = min(4.0, remaining - 5)
            if budget > 2:
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
