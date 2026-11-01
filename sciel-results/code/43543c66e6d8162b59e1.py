import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 143.0

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _compute_facet_data(points):
    points = _normalize(points)
    try:
        hull = scipy.spatial.ConvexHull(points)
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
    hull, centers, radii, worst_idx = _compute_facet_data(points)
    if hull is None or worst_idx is None:
        return np.pi
    return radii[worst_idx]

def _soft_max_radius_and_grad(points, beta=50.0):
    hull, centers, radii, worst_idx = _compute_facet_data(points)
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

def _grid_soft_max_radius_and_grad(points, grid, beta=50.0):
    """
    Soft-max of angular distances from each grid point to nearest configuration point.
    """
    # Compute angular distances: grid (M,3) vs points (N,3)
    # dot products
    dots = grid @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    ang_dists = np.arccos(dots)  # (M, N)
    
    # For each grid point, the distance to nearest config point
    min_dists = np.min(ang_dists, axis=1)  # (M,)
    
    # Soft-max of min_dists
    max_d = np.max(min_dists)
    shifted = beta * (min_dists - max_d)
    max_shifted = np.max(shifted)
    exp_terms = np.exp(shifted - max_shifted)
    log_sum = max_shifted + np.log(np.sum(exp_terms))
    soft_radius = max_d + log_sum / beta
    weights = exp_terms / np.sum(exp_terms)  # (M,)
    
    # Gradient: for each grid point m, the gradient of arccos(dot(g, p_j)) w.r.t. p_j
    # is -g / sin(dist) projected to tangent space at p_j
    # But we only take the min over j, so we need to know which j is the nearest
    # For soft-max, we weight by the softmax weights and also by which j is the argmin
    # Actually, the gradient of min_j f_j is the gradient of f_{j*} where j* is the argmin
    # For soft-min, it's a weighted average
    
    # For each grid point m, find the nearest config point index
    min_indices = np.argmin(ang_dists, axis=1)  # (M,)
    
    grads = np.zeros_like(points)
    for m in range(len(grid)):
        w = weights[m]
        if w < 1e-12:
            continue
        j = min_indices[m]
        g = grid[m]
        d = min_dists[m]
        sin_d = np.sin(d)
        if sin_d < 1e-10:
            continue
        p = points[j]
        # Gradient of arccos(dot(g, p)) w.r.t. p is -g/sin(d)
        grad_contrib = -g / sin_d
        # Project to tangent space at p
        radial = np.dot(grad_contrib, p) * p
        tangential = grad_contrib - radial
        grads[j] += w * tangential
    
    return soft_radius, grads

def _make_grid(n_theta=30, n_phi=60):
    """
    Create a dense spherical grid of sample points.
    """
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid

def _repulsion_relax(pts, time_budget, alpha_start=0.05):
    pts = pts.copy()
    alpha = alpha_start
    start_t = time.time()
    
    while time.time() - start_t < time_budget:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dot_fp = np.sum(forces * pts, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * pts
        
        step = alpha
        improved = False
        for _ in range(8):
            new_pts = pts + step * forces_tan
            new_pts = _normalize(new_pts)
            # Check for improvement in repulsion energy
            diff_new = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
            dist_sq_new = np.sum(diff_new * diff_new, axis=2)
            inv_dist_sq_new = np.zeros_like(dist_sq_new)
            np.divide(1.0, dist_sq_new, out=inv_dist_sq_new, where=~mask)
            new_energy = 0.5 * np.sum(inv_dist_sq_new)
            energy = 0.5 * np.sum(inv_dist_sq)
            if new_energy < energy:
                pts = new_pts
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
            w = (rad / max_rad) ** 2
            cc = centers[f]
            for idx in hull.simplices[f]:
                updates[idx] += w * (cc - pts[idx])
                weights_sum[idx] += w
        
        mask = weights_sum > 1e-10
        if np.any(mask):
            pts[mask] += step_size * updates[mask] / weights_sum[mask, np.newaxis]
            pts = _normalize(pts)
    
    return pts

def _lbfgs_facet(points, time_budget):
    """
    L-BFGS-B optimization using soft-max of facet circumradii.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_r = _compute_covering_radius(points)
    
    for beta in [80.0, 150.0, 300.0]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
        def objective_and_grad(x):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _normalize(x.reshape(N, 3))
            soft_r, grads = _soft_max_radius_and_grad(pts, beta=beta)
            return soft_r, grads.reshape(-1)
        
        x0 = best_pts.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if result.fun < best_r:
                best_pts = _normalize(result.x.reshape(N, 3))
                best_r = _compute_covering_radius(best_pts)
        except Exception:
            pass
    
    return best_pts, best_r

def _lbfgs_grid(points, grid, time_budget):
    """
    L-BFGS-B optimization using soft-max of grid-based covering radius.
    """
    start_time = time.time()
    best_pts = points.copy()
    best_r = _compute_covering_radius(points)
    
    for beta in [50.0, 100.0, 200.0]:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 3.0:
            break
        
        def objective_and_grad(x):
            if time.time() - start_time > time_budget:
                raise Exception("timeout")
            pts = _normalize(x.reshape(N, 3))
            soft_r, grads = _grid_soft_max_radius_and_grad(pts, grid, beta=beta)
            return soft_r, grads.reshape(-1)
        
        x0 = best_pts.reshape(-1)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if result.fun < best_r:
                best_pts = _normalize(result.x.reshape(N, 3))
                best_r = _compute_covering_radius(best_pts)
        except Exception:
            pass
    
    return best_pts, best_r

def _local_search_gradient(points, time_budget, beta=100.0):
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    step_size = 0.03
    min_step = 1e-7
    decay = 0.95
    growth = 1.05
    
    while time.time() - start_time < time_budget:
        soft_r, grads = _soft_max_radius_and_grad(current_points, beta=beta)
        
        if soft_r < best_radius - 1e-12:
            best_radius = soft_r
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-12:
            break
        
        grads = grads / grad_norm
        new_points = current_points + step_size * grads
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < soft_r:
            current_points = new_points
            step_size = min(step_size * growth, 0.1)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _simulated_annealing(points, time_budget, T_start=0.05, T_end=0.001):
    current_points = points.copy()
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    start_time = time.time()
    total_time = time_budget
    
    while time.time() - start_time < time_budget:
        elapsed = time.time() - start_time
        progress = min(elapsed / total_time, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        rng = np.random.default_rng()
        noise = rng.normal(size=(N, 3)) * T
        new_points = current_points + noise
        new_points = _normalize(new_points)
        
        new_radius = _compute_covering_radius(new_points)
        delta = new_radius - best_radius
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            current_points = new_points
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
    
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
    
    # 2. Random starts
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
    
    # 3. Icosahedron-based seed
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    if len(ico_verts) < N:
        rng = np.random.default_rng(0)
        extra = rng.normal(size=(N - len(ico_verts), 3))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        ico_verts = np.vstack([ico_verts, extra])
    seeds.append(ico_verts[:N])
    
    # Create grid for grid-based optimization
    grid = _make_grid(n_theta=25, n_phi=50)
    
    n_seeds = len(seeds)
    
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 20:
            break
        
        if i < n_seeds - 1:
            time_budget = min(12.0, remaining / (n_seeds - i))
        else:
            time_budget = min(20.0, remaining)
        
        if time_budget < 5.0:
            break
        
        # Phase 1: Repulsion relaxation to spread points
        if time_budget > 8.0:
            seed_pts = _repulsion_relax(seed_pts, time_budget=min(5.0, time_budget * 0.3))
        
        # Phase 2: Alternate between facet-based and grid-based L-BFGS
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 10:
            break
        
        if i % 2 == 0:
            # Facet-based L-BFGS
            t_budget = min(10.0, remaining * 0.5)
            if t_budget > 3:
                pts, rad = _lbfgs_facet(seed_pts, time_budget=t_budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
            
            # Follow up with Lloyd iteration
            elapsed = time.time() - start_time
            remaining = deadline - elapsed
            if remaining > 5 and best_points is not None:
                t_budget = min(5.0, remaining * 0.4)
                if t_budget > 2:
                    pts = _lloyd_iteration(best_points, time_budget=t_budget)
                    rad = _compute_covering_radius(pts)
                    if rad < best_radius:
                        best_radius = rad
                        best_points = pts.copy()
        else:
            # Grid-based L-BFGS
            t_budget = min(10.0, remaining * 0.5)
            if t_budget > 3:
                pts, rad = _lbfgs_grid(seed_pts, grid, time_budget=t_budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
            
            # Follow up with local gradient search
            elapsed = time.time() - start_time
            remaining = deadline - elapsed
            if remaining > 5 and best_points is not None:
                t_budget = min(5.0, remaining * 0.4)
                if t_budget > 2:
                    pts, rad = _local_search_gradient(best_points, time_budget=t_budget, beta=150.0)
                    if rad < best_radius:
                        best_radius = rad
                        best_points = pts.copy()
    
    # Final polish: simulated annealing
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 15:
            budget = min(15.0, remaining - 5)
            if budget > 5:
                pts, rad = _simulated_annealing(best_points, time_budget=budget, T_start=0.03, T_end=0.0005)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    # Final high-precision polish with facet-based L-BFGS
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 10:
            budget = min(12.0, remaining - 3)
            if budget > 5:
                pts, rad = _lbfgs_facet(best_points, time_budget=budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    # Final grid-based polish
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 8:
            budget = min(10.0, remaining - 2)
            if budget > 4:
                pts, rad = _lbfgs_grid(best_points, grid, time_budget=budget)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    best_points = _normalize(best_points)
    
    return best_points
