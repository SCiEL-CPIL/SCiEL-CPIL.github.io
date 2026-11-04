"""
Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius.
Combines BCC lattice initialization with DE optimization, L-BFGS-B violation-energy
minimization, SGLD for escape, and high-precision contact-graph polish.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _energy_and_grad(x, target_r):
    """
    Energy for a fixed target radius: sum of squared violations of
    pairwise separation (2*r) and wall distance (r).
    Returns (energy, gradient_flat).
    """
    c = x.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False
    
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    energy_pair = 0.5 * np.sum(violations ** 2)
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    wall_dists = HALF - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    energy_wall = np.sum(wall_violations ** 2)
    
    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _soft_min_energy(x, target_r, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for the min objective.
    We want to maximize min(d_pair/2, wall_dist).
    Equivalently, minimize -min(...) = max(-d_pair/2, -wall_dist).
    Using soft-max: soft_max(a_i) ≈ (1/beta) * log(sum(exp(beta*a_i)))
    So we minimize soft_max(-d_pair/2, -wall_dist) over all constraints.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    # Only consider pairs that are close (to avoid numerical issues)
    # Use a threshold: only pairs within 2.5*target_r
    close_mask = dists < 2.5 * target_r
    close_mask[np.diag_indices_from(close_mask)] = False
    
    # Negative pairwise half-distances
    neg_pair = -dists / 2.0
    
    # Wall distances
    wall_dists = HALF - np.max(np.abs(c), axis=1)
    neg_wall = -wall_dists
    
    # Collect all negative values
    # For pairs: only those that are "active" (close)
    pair_vals = neg_pair[close_mask]
    wall_vals = neg_wall
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft max with log-sum-exp for numerical stability
    max_val = np.max(all_vals)
    if max_val > 1e10:
        return max_val, np.zeros_like(x)
    
    exp_vals = np.exp(beta * (all_vals - max_val))
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        soft_max_val = max_val
    else:
        soft_max_val = max_val + np.log(sum_exp) / beta
    
    # Gradient: d(soft_max)/dx = sum_i (exp(beta*a_i)/sum_exp) * (beta * da_i/dx)
    # = sum_i (p_i) * da_i/dx where p_i = exp(beta*a_i)/sum_exp
    weights = exp_vals / sum_exp
    
    # Gradient for pairs
    grad_pair = np.zeros((N, 3))
    # For each active pair (i,j), da/dx_i = -0.5 * (c_i - c_j)/d
    # We need to accumulate weighted contributions
    # This is expensive, so we approximate by only considering the top few
    # For simplicity, use the violation gradient scaled
    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Gradient for walls
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    signs = np.sign(c)
    grad_wall = 2.0 * wall_violations * signs
    
    total_grad = (grad_pair + grad_wall).flatten()
    
    return soft_max_val, total_grad

def _local_optimize_violation(init_pts, target_r, deadline, max_iter=3000):
    """
    L-BFGS-B minimization of the violation energy for a fixed target radius.
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 50}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def _sgld_refine(init_pts, target_r, deadline, n_steps=5000):
    """
    Stochastic Gradient Langevin Dynamics to escape local minima.
    """
    t_start = time.time()
    x = init_pts.flatten().copy()
    best_x = x.copy()
    best_r = _compute_radius(x.reshape(N, 3))
    
    step_size = 0.002
    temp = 0.005
    step_decay = 0.99995
    temp_decay = 0.99995
    
    for i in range(n_steps):
        if time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(x, target_r)
        
        noise = np.random.randn(len(x))
        x = x - step_size * grad + np.sqrt(2 * step_size * temp) * noise
        
        x = np.clip(x, -0.495, 0.495)
        
        if i % 200 == 0 or i == n_steps - 1:
            pts = x.reshape(N, 3)
            if np.all(np.isfinite(pts)):
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_x = x.copy()
        
        step_size *= step_decay
        temp *= temp_decay
        
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return best_x.reshape(N, 3)

def _generate_bcc_points(n_max=4):
    """Generate a set of BCC lattice points centered at the origin."""
    points = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max):
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    center = points.mean(axis=0)
    points = points - center
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _generate_fcc():
    """Generate FCC lattice points in the cube."""
    points = []
    a = 0.2
    basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]])
    
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                for b in basis:
                    pt = a * np.array([ix, iy, iz]) + a * b
                    points.append(pt)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    points *= 0.5 / 0.6
    
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) > N:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), N, replace=False)
        points = points[idx]
    
    if len(points) < N:
        rng = np.random.default_rng(43)
        extra = rng.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
        
    return points

def _generate_grid():
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _bcc_objective(params, V):
    """
    Objective function for BCC lattice optimization.
    params: [scale, tx, ty, tz, rx, ry, rz]
    V: base BCC points (N, 3)
    Returns negative score.
    """
    s, tx, ty, tz, rx, ry, rz = params
    
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    C = s * (V @ R.T) + np.array([tx, ty, tz])
    
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    
    if r_bound < 0:
        return -r_bound + 10.0
    
    D = cdist(C, C)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r_pair = min_dist / 2.0
    
    r = min(r_bound, r_pair)
    return -r

def _lattice_optimize_bcc(V, time_limit=25.0):
    """Optimize BCC lattice parameters using differential evolution."""
    t_start = time.time()
    
    bounds = [(0.05, 0.25), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
              (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    def objective(params):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _bcc_objective(params, V)
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            args=(V,),
            maxiter=80,
            popsize=20,
            tol=1e-7,
            seed=42,
            polish=True
        )
        return result.x
    except TimeoutError:
        return None
    except Exception:
        return None

def _contact_graph_refine(centers, radius, deadline):
    """
    Identify near-contact pairs and refine only the involved spheres.
    """
    if time.time() > deadline - 5.0:
        return centers, _compute_radius(centers)
    
    centers = centers.copy()
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.05 * radius)
    
    if len(pairs) == 0:
        return centers, _compute_radius(centers)
    
    involved = set()
    for i, j in pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)
    
    if len(involved) < 2:
        return centers, _compute_radius(centers)
    
    idx_array = np.array(involved)
    n_involved = len(involved)
    
    # Build a sub-problem: maximize min distance among involved pairs and wall
    # Use a simple approach: iterative pairwise push
    
    for iteration in range(50):
        if time.time() > deadline - 3.0:
            break
        
        improved = False
        for i, j in pairs:
            c_i = centers[i].copy()
            c_j = centers[j].copy()
            d = c_i - c_j
            dist = np.linalg.norm(d)
            
            if dist < 2.0 * radius:
                # Push apart
                push = (2.0 * radius - dist) / 2.0
                if dist > 1e-12:
                    direction = d / dist
                    new_ci = c_i + push * direction
                    new_cj = c_j - push * direction
                    
                    # Check walls
                    if np.all(np.abs(new_ci) < HALF - 1e-9) and np.all(np.abs(new_cj) < HALF - 1e-9):
                        centers[i] = new_ci
                        centers[j] = new_cj
                        improved = True
        
        if not improved:
            break
    
    # Also push against walls
    for i in range(N):
        for k in range(3):
            if centers[i, k] > 0 and centers[i, k] + radius > HALF:
                centers[i, k] = HALF - radius - 1e-9
            elif centers[i, k] < 0 and centers[i, k] - radius < -HALF:
                centers[i, k] = -HALF + radius + 1e-9
    
    centers = np.clip(centers, -HALF + 1e-9, HALF - 1e-9)
    return centers, _compute_radius(centers)

def _high_precision_polish(centers, deadline):
    """
    Final high-precision polish: use very tight tolerances and more iterations.
    """
    if time.time() > deadline - 10.0:
        return centers, _compute_radius(centers)
    
    current_r = _compute_radius(centers)
    
    # Try to push the radius up slightly using violation minimization
    target_r = current_r + 1e-6
    
    opt_pts = _local_optimize_violation(
        centers + np.random.default_rng(99).normal(0, 1e-6, size=(N, 3)),
        target_r,
        deadline,
        max_iter=5000
    )
    
    r_opt = _compute_radius(opt_pts)
    if r_opt > current_r:
        centers = opt_pts
        current_r = r_opt
    
    # Contact graph refinement
    centers, current_r = _contact_graph_refine(centers, current_r, deadline)
    
    return centers, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_centers = None
    best_radius = 0.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate initializations
    initializations = []
    
    # BCC lattice with DE optimization (primary)
    if time.time() < deadline - 60.0:
        V_bcc = _generate_bcc_points(4)
        params = _lattice_optimize_bcc(V_bcc, time_limit=25.0)
        
        if params is not None:
            s, tx, ty, tz, rx, ry, rz = params
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            
            pts_bcc = s * (V_bcc @ R.T) + np.array([tx, ty, tz])
            pts_bcc = np.clip(pts_bcc, -0.49, 0.49)
            initializations.append(pts_bcc)
        
        # Also add unoptimized BCC
        V_bcc_simple = _generate_bcc_points(4)
        V_bcc_simple -= V_bcc_simple.mean(axis=0)
        scale = 0.5 / np.max(np.abs(V_bcc_simple)) * 0.9
        pts_bcc_simple = scale * V_bcc_simple
        pts_bcc_simple = np.clip(pts_bcc_simple, -0.49, 0.49)
        initializations.append(pts_bcc_simple)
    
    # FCC lattice
    fcc_centers = _generate_fcc()
    initializations.append(fcc_centers)
    
    # Grid
    grid_centers = _generate_grid()
    initializations.append(grid_centers)
    
    # Perturbed versions
    for i in range(4):
        if time.time() > deadline - 50.0:
            break
        base = _generate_bcc_points(4)
        base -= base.mean(axis=0)
        scale = 0.5 / np.max(np.abs(base)) * 0.9
        base = scale * base
        noise_scale = 0.02 + 0.01 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)
    
    for i in range(3):
        if time.time() > deadline - 45.0:
            break
        base = _generate_fcc()
        noise_scale = 0.02 + 0.01 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)
    
    # Random
    for i in range(2):
        if time.time() > deadline - 40.0:
            break
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)
    
    # Phase 2: Optimize each initialization with multiple target radii
    target_radii = [0.105, 0.108, 0.110, 0.111, 0.112, 0.113]
    
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 35.0:
            break
        
        # Try each target radius from this initialization
        for target_r in target_radii:
            if time.time() > deadline - 20.0:
                break
            
            # Add small noise to break symmetry
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)
            
            opt_pts = _local_optimize_violation(init_pts, target_r, deadline)
            
            if np.all(np.isfinite(opt_pts)):
                r_opt = _compute_radius(opt_pts)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_pts.copy()
    
    # Phase 3: SGLD refinement on best solution
    if best_centers is not None and time.time() < deadline - 20.0:
        target_r = best_radius + 1e-5
        sgld_pts = _sgld_refine(best_centers, target_r, deadline, n_steps=3000)
        r_sgld = _compute_radius(sgld_pts)
        if r_sgld > best_radius:
            best_radius = r_sgld
            best_centers = sgld_pts
    
    # Phase 4: High-precision polish
    if best_centers is not None and time.time() < deadline - 15.0:
        best_centers, best_radius = _high_precision_polish(best_centers, deadline)
    
    # Phase 5: Final contact graph refinement
    if best_centers is not None and time.time() < deadline - 5.0:
        for _ in range(2):
            if time.time() > deadline - 3.0:
                break
            best_centers, best_radius = _contact_graph_refine(best_centers, best_radius, deadline)
    
    # Final validation
    if best_centers is None:
        best_centers = _generate_grid()
    
    best_centers = np.clip(best_centers, -HALF + 1e-9, HALF - 1e-9)
    
    return best_centers
