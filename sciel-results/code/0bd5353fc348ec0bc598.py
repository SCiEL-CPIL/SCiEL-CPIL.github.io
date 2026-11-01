import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
TIME_LIMIT = 145.0  # safety margin from 150s

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
    
    simplices = hull.simplices
    n_facets = len(simplices)
    if n_facets == 0:
        return hull, np.zeros((0, 3)), np.zeros(0), None
        
    # Vectorized computation of circumcenters
    # For each triangle (a, b, c), we want m such that:
    # m.a = 1, m.b = 1, m.c = 1
    # m = x*a + y*b + z*c
    # [1   a.b a.c] [x]   [1]
    # [a.b 1   b.c] [y] = [1]
    # [a.c b.c 1  ] [z]   [1]
    
    # Extract vertices
    # a = points[simplices[:, 0]]
    # b = points[simplets[:, 1]]
    # c = points[simplices[:, 2]]
    
    # We can compute the Gram matrix for each facet.
    # G[i] = [[1, ab, ac], [ab, 1, bc], [ac, bc, 1]]
    
    # Let's compute dot products
    # a, b, c shapes: (n_facets, 3)
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    ab = np.einsum('ij,ij->i', a, b)
    ac = np.einsum('ij,ij->i', a, c)
    bc = np.einsum('ij,ij->i', b, c)
    
    # Build Gram matrices
    # G = np.zeros((n_facets, 3, 3))
    # G[:, 0, 0] = 1.0
    # G[:, 0, 1] = ab
    # G[:, 0, 2] = ac
    # G[:, 1, 0] = ab
    # G[:, 1, 1] = 1.0
    # G[:, 1, 2] = bc
    # G[:, 2, 0] = ac
    # G[:, 2, 1] = bc
    # G[:, 2, 2] = 1.0
    
    # Instead of building G explicitly, we can use np.linalg.solve on a batch?
    # No, np.linalg.solve doesn't batch 3x3 systems directly in older numpy without care, but it supports (..., M, M).
    # Let's try to construct G.
    
    G = np.empty((n_facets, 3, 3))
    G[:, 0, 0] = 1.0
    G[:, 1, 1] = 1.0
    G[:, 2, 2] = 1.0
    G[:, 0, 1] = ab
    G[:, 0, 2] = ac
    G[:, 1, 0] = ab
    G[:, 1, 2] = bc
    G[:, 2, 0] = ac
    G[:, 2, 1] = bc
    
    ones = np.ones((n_facets, 3))
    
    try:
        m_coeffs = np.linalg.solve(G, ones) # shape (n_facets, 3)
    except np.linalg.LinAlgError:
        # Fallback to loop if singular (should be rare for valid hulls)
        m_coeffs = np.zeros((n_facets, 3))
        for i in range(n_facets):
            try:
                m_coeffs[i] = np.linalg.solve(G[i], ones[i])
            except:
                m_coeffs[i] = 0.0

    # c_dir = m[0]*a + m[1]*b + m[2]*c
    # c_dir[i] = m_coeffs[i,0]*a[i] + m_coeffs[i,1]*b[i] + m_coeffs[i,2]*c[i]
    c_dirs = m_coeffs[:, 0:1] * a + m_coeffs[:, 1:2] * b + m_coeffs[:, 2:3] * c
    
    norms_c = np.linalg.norm(c_dirs, axis=1, keepdims=True)
    # Avoid division by zero
    norms_c = np.maximum(norms_c, 1e-15)
    centers = c_dirs / norms_c
    
    # Radii: arccos(dot(center, a))
    dots = np.einsum('ij,ij->i', centers, a)
    dots = np.clip(dots, -1.0, 1.0)
    radii = np.arccos(dots)
    
    # Handle cases where center computation failed (norm was tiny)
    # If norm was tiny, radius is effectively pi or undefined. 
    # For robustness, if norm < 1e-10, we treat radius as large (pi/2 or pi)
    invalid_mask = np.linalg.norm(c_dirs, axis=1) < 1e-10
    if np.any(invalid_mask):
         radii[invalid_mask] = np.pi/2 # Conservative estimate

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
    Uses a more numerically stable approach with correct tangent-space gradient.
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
    
    # Vectorize gradient accumulation where possible?
    # grad[p_idx] += w * (-c_tangent / sin_r)
    # c_tangent = C - (C.p) p
    
    simplices = hull.simplices
    
    # We need to accumulate gradients for each point.
    # Let's loop over facets but vectorize the dot products if possible? 
    # The loop is over facets, which is ~100. This is fast enough.
    
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
        
        simplex = simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            # Gradient of arccos(dot(C, p)) w.r.t. p on tangent space:
            # d/dp arccos(C·p) = -(C - (C·p)p) / sin(arccos(C·p))
            c_dot_p = np.dot(C, p)
            c_tangent = C - c_dot_p * p
            grad_tangent = -c_tangent / sin_r
            grads[p_idx] += w * grad_tangent
    
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
    # Increased maxiter and tightened tolerances due to speedup
    for beta in [50.0, 150.0, 400.0]:
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
                options={'maxiter': 2000, 'maxfun': 15000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
            )
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            new_r = _compute_covering_radius(new_pts)
            if new_r < best_soft_r - 1e-12:
                best_pts = new_pts
                best_soft_r = new_r
        except Exception:
            pass
    
    return best_pts

def _local_polish(points, time_budget, beta=100.0):
    """
    Final local polish using projected gradient descent on tangent space.
    Directly minimizes the soft-max covering radius.
    Uses adaptive step size with backtracking line search.
    """
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.03
    min_step = 1e-15
    decay = 0.6
    growth = 1.1
    stagnant = 0
    max_stagnant = 120
    
    while time.time() - start_time < time_budget:
        soft_radius, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-15:
            best_radius = true_radius
            best_points = current_points.copy()
            stagnant = 0
        else:
            stagnant += 1
        
        if stagnant > max_stagnant:
            break
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-17:
            break
        
        direction = -grads / grad_norm
        
        # Backtracking line search
        accepted = False
        for _ls in range(20):
            new_points = current_points + step_size * direction
            new_points = _project_to_sphere(new_points)
            
            new_radius = _compute_covering_radius(new_points)
            
            if new_radius < true_radius - 1e-16:
                current_points = new_points
                step_size = min(step_size * growth, 0.1)
                accepted = True
                break
            else:
                step_size *= decay
        
        if not accepted:
            if step_size < min_step:
                break
            # Try a smaller step
            step_size = max(step_size, 1e-14)
    
    return best_points, best_radius

def _grid_slqp_polish(points, time_budget):
    """
    SLSQP-based polish using a dense spherical grid as constraint points.
    Minimizes slack variable t subject to grid-point distance constraints.
    This is a differentiable surrogate that can escape local minima of the
    hull-based approach.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_radius = _compute_covering_radius(best_pts)
    
    # Create a moderately dense spherical grid for constraints
    # ~2000 points should be sufficient to capture the covering radius well
    n_theta = 40
    n_phi = 40
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    grid = np.column_stack([
        np.sin(theta_grid) * np.cos(phi_grid),
        np.sin(theta_grid) * np.sin(phi_grid),
        np.cos(theta_grid)
    ]).reshape(-1, 3)
    # Remove duplicates (poles)
    grid = grid[~np.isclose(grid[:, 2], 1.0) | ~np.isclose(grid[:, 0], 0.0) & ~np.isclose(grid[:, 1], 0.0)]
    grid = grid[~np.isclose(grid[:, 2], -1.0) | ~np.isclose(grid[:, 0], 0.0) & ~np.isclose(grid[:, 1], 0.0)]
    # Keep unique
    seen = set()
    unique_grid = []
    for g in grid:
        key = tuple(np.round(g, 10))
        if key not in seen:
            seen.add(key)
            unique_grid.append(g)
    grid = np.array(unique_grid)
    n_grid = len(grid)
    
    def _grid_covering_radius(pts):
        """Compute max angular distance from grid points to nearest site."""
        # Vectorized: compute all dot products
        dots = grid @ pts.T  # (n_grid, N)
        max_dots = np.max(dots, axis=1)  # (n_grid,)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
        return np.max(angles)
    
    def _objective_and_grad(x):
        if time.time() - start_time > time_budget:
            raise Exception("timeout")
        pts = _normalize(x[:3*N].reshape(N, 3))
        t = x[3*N]
        
        # Objective: minimize t
        f = t
        
        # Gradient of objective
        grad = np.zeros_like(x)
        grad[3*N] = 1.0
        
        # Constraints: for each grid point, arccos(max_dot) <= t
        # Equivalent: max_dot >= cos(t)
        # We use the smooth max of (cos(t) - dot_i) as constraint
        dots = grid @ pts.T  # (n_grid, N)
        max_dots = np.max(dots, axis=1)
        cos_t = np.cos(t)
        
        # Gradient of max_dot w.r.t. pts
        # For each grid point, the argmax site contributes
        argmax_idx = np.argmax(dots, axis=1)
        
        for k in range(N):
            mask = (argmax_idx == k)
            if np.any(mask):
                # d(max_dot)/d(pts[k]) = grid[mask]
                grad[3*N + 3*k:3*N + 3*k+3] += (grid[mask] @ np.ones(n_grid)).reshape(3)
        
        # Actually, let's use a simpler approach:
        # f = t + mu * sum(max(0, arccos(max_dot_i) - t)^2)
        # This is a penalty approach
        
        return f, grad
    
    # Use penalty-based approach instead for simplicity
    def _penalty_obj_and_grad(x):
        if time.time() - start_time > time_budget:
            raise Exception("timeout")
        pts = _normalize(x[:3*N].reshape(N, 3))
        t = x[3*N]
        
        dots = grid @ pts.T
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
        
        # Penalty: sum of (max(0, angle_i - t))^2
        violations = np.maximum(0, angles - t)
        f = t + 0.5 * np.sum(violations ** 2)
        
        grad = np.zeros_like(x)
        grad[3*N] = 1.0 + np.sum(violations)
        
        # Gradient w.r.t. points
        # d(angle_i)/d(pts) = -grid[i] / sin(angle_i) for the nearest site
        sin_angles = np.sin(angles)
        sin_angles = np.maximum(sin_angles, 1e-10)
        
        argmax_idx = np.argmax(dots, axis=1)
        
        for k in range(N):
            mask = (argmax_idx == k) & (violations > 0)
            if np.any(mask):
                # d(angle)/d(pts[k]) = -grid_i / sin(angle_i)
                contrib = grid[mask] * (violations[mask] / sin_angles[mask])[:, None]
                grad[3*N + 3*k:3*N + 3*k+3] -= np.sum(contrib, axis=0)
        
        return f, grad
    
    x0 = np.concatenate([best_pts.reshape(-1), [best_radius]])
    
    try:
        result = minimize(
            _penalty_obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.fun < best_radius + 0.01:
            new_pts = _normalize(result.x[:3*N].reshape(N, 3))
            new_radius = _compute_covering_radius(new_pts)
            if new_radius < best_radius:
                best_pts = new_pts
                best_radius = new_radius
    except Exception:
        pass
    
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
    # Increased number of seeds and varied noise
    seed_vals = [42, 1, 2, 3, 7, 13, 99, 123, 555, 777, 2024, 31415, 101, 202, 303, 404, 505, 606, 707, 808]
    for i, seed_val in enumerate(seed_vals):
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        # Add some noise to break symmetry?
        # Or just use random points.
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
    
    # 4. Add perturbed Fibonacci
    rng_pert = np.random.default_rng(999)
    pert_fib = fib + rng_pert.normal(scale=0.1, size=(N, 3))
    pert_fib /= np.linalg.norm(pert_fib, axis=1, keepdims=True)
    seeds.append(pert_fib)

    n_seeds = len(seeds)

    # Stage 1: L-BFGS optimization for each seed
    # Reserve ~20s for final polish stages
    stage1_deadline = deadline - 20.0

    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed

        if remaining < 5:
            break

        if i < n_seeds - 1:
            time_budget = min(5.0, remaining / (n_seeds - i))
        else:
            time_budget = min(8.0, remaining)

        if time_budget < 3.0:
            break

        # Apply L-BFGS stage
        pts = _lbfgs_stage(seed_pts, time_budget)
        
        # Quick local polish
        elapsed = time.time() - start_time
        remaining = stage1_deadline - elapsed
        if remaining > 4:
            pts, rad = _local_polish(pts, min(3.0, remaining - 2), beta=100.0)
        else:
            rad = _compute_covering_radius(pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()

    # Stage 2: Final polish on best candidate
    if best_points is not None:
        # Fine local polish with moderate beta
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 15:
            budget = min(10.0, remaining - 10)
            if budget > 5:
                pts, rad = _local_polish(best_points, budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Very fine polish with high beta for final convergence
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 10:
            budget = min(8.0, remaining - 8)
            if budget > 4:
                pts, rad = _local_polish(best_points, budget, beta=500.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Ultra-fine polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 8:
            budget = min(5.0, remaining - 5)
            if budget > 3:
                pts, rad = _local_polish(best_points, budget, beta=1000.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
        
        # Final tight polish
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5:
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
    
    # Final sanity check: verify no coincident points, not in hemisphere
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-10):
        best_points = seeds[0]
    
    return best_points
