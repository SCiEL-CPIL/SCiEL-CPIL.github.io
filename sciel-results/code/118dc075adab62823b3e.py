import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # 15s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check if inside unit circle
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0
    
    # Calculate min distance to boundary
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    # Calculate min pairwise distance
    if N < 2:
        return r_cont
        
    # Efficient pairwise distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # We only need upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation with log-sum-exp trick for numerical stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms: d/dpts_i of (1 - |c_i|) = -pts_i / |c_i|
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms: d/dpts_i of (|c_i - c_j|/2) = 0.5 * (c_i - c_j)/|c_i - c_j|
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0
    
    for spacing in [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
        
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _ring_init():
    """Generate a ring-based initialization matching known optimal structure."""
    best_pts = None
    best_score = -1.0
    
    # Known optimal structure: 1 center + 4 rings: 1, 6, 12, 18, 13 = 50
    rings = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.95]
    
    for seed in range(20):
        rng = np.random.default_rng(42 + seed)
        pts_list = []
        
        for ring_idx, count in enumerate(rings):
            if ring_idx == 0:
                pts_list.append([0.0, 0.0])
            else:
                radius = ring_radii[ring_idx]
                angle_offset = rng.uniform(0, 2 * np.pi)
                for k in range(count):
                    angle = angle_offset + 2 * np.pi * k / count
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    r_jitter = rng.normal(0, 0.003)
                    x += r_jitter * np.cos(angle)
                    y += r_jitter * np.sin(angle)
                    pts_list.append([x, y])
        
        pts = np.array(pts_list[:N])
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    if best_pts is not None:
        return best_pts
    return _hex_lattice_init()

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _perturb(pts, sigma=0.01):
    """Add small random perturbation to escape local minima."""
    noise = np.random.normal(0, sigma, pts.shape)
    return pts + noise

def _project_to_disk(pts):
    """Project points back into the unit disk."""
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        pts = pts.copy()
        pts[mask] = pts[mask] / dists[mask, np.newaxis]
    return pts

def _force_directed_relax(centers, max_steps=3000):
    """
    Force-directed relaxation using repulsive springs and boundary constraint.
    Acts as a local search / escape mechanism from local minima.
    """
    current = centers.copy()
    n = len(current)
    best_score = _score(current)
    best_positions = current.copy()
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 15.0:
            break
        
        forces = np.zeros_like(current)
        
        # Repulsive forces between all pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        diff = current[i_idx, :] - current[j_idx, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        
        # Strong repulsion when close, using 1/d^2 potential
        repulsion_strength = 1.0 / (safe_dists ** 2 + 1e-6)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        weighted_forces = repulsion_strength[:, np.newaxis] * unit_vecs
        
        np.add.at(forces, i_idx, weighted_forces)
        np.add.at(forces, j_idx, -weighted_forces)
        
        # Boundary constraint: push inward if too close to edge
        norms = np.linalg.norm(current, axis=1)
        # Soft boundary: quadratic penalty outside radius 0.9
        boundary_force = np.where(norms > 0.9, (norms - 0.9) ** 2 * 10.0, 0.0)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        forces -= boundary_force[:, np.newaxis] * current / safe_norms[:, np.newaxis]
        
        # Adaptive time step based on max force
        max_force = np.max(np.linalg.norm(forces, axis=1))
        dt = 0.0005 / (1.0 + max_force * 5.0)
        
        current += dt * forces
        current = _project_to_disk(current)
        
        # Check improvement
        s = _score(current)
        if s > best_score:
            best_score = s
            best_positions = current.copy()
    
    return best_positions

def _optimize_lbfgs(pts, beta, maxiter=500, gtol=1e-8):
    """Run L-BFGS-B optimization with given beta."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            pts.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            jac=True,
            bounds=[(-1.0, 1.0)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol, 'disp': False}
        )
        new_pts = res.x.reshape(N, 2)
        new_pts = _project_to_disk(new_pts)
        return new_pts
    except Exception:
        return pts

def _newton_active_set(pts, max_iter=50, tol=1e-12):
    """
    Newton-Raphson active-set solver for the exact contact-constraint system.
    
    Given an initial packing, identify the active contact graph (pairs at minimum
    distance and points on boundary), then solve the nonlinear system where all
    active constraints are tight: |c_i - c_j| = 2r for active pairs, |c_i| + r = 1
    for boundary points. Uses adaptive active-set updates.
    """
    current = pts.copy().astype(float)
    n = len(current)
    
    for outer_iter in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Compute current distances
        norms = np.linalg.norm(current, axis=1)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Current radius
        min_pair = 0.5 * np.min(dist_mat[np.triu_indices(n, k=1)])
        min_bound = 1.0 - np.max(norms)
        r = min(min_pair, min_bound)
        
        if r <= 0:
            break
        
        # Identify active constraints with tolerance
        active_tol = 1e-6
        iu = np.triu_indices(n, k=1)
        pair_dists = dist_mat[iu]
        active_pairs = np.where(pair_dists < 2 * r + active_tol)[0]
        active_boundary = np.where(norms > 1.0 - r - active_tol)[0]
        
        # Need enough constraints to determine the system (2N unknowns)
        # Each active pair gives 1 constraint, each boundary point gives 1 constraint
        # We need at least 2N constraints for a determined system
        if len(active_pairs) + len(active_boundary) < 2 * n:
            # Relax tolerance to get more active constraints
            active_tol *= 2
            active_pairs = np.where(pair_dists < 2 * r + active_tol)[0]
            active_boundary = np.where(norms > 1.0 - r - active_tol)[0]
            if len(active_pairs) + len(active_boundary) < 2 * n:
                break
        
        # Build the nonlinear system F(x) = 0
        # Unknowns: 2N coordinates + 1 radius = 2N+1 variables
        # But we can eliminate r by using it as a parameter
        
        # For Newton, we solve for perturbations to points and r
        # F_k = 0 for each active constraint
        
        # Build Jacobian and residual
        # Variables: x_1...x_N, y_1...y_N, r (2N+1 variables)
        # Constraints:
        #   For each active pair (i,j): |c_i - c_j|^2 - (2r)^2 = 0
        #   For each boundary point i: |c_i|^2 + 2r|c_i| + r^2 - 1 = 0  (i.e., |c_i| + r - 1 = 0)
        
        n_pairs = len(active_pairs)
        n_bound = len(active_boundary)
        n_constraints = n_pairs + n_bound
        n_vars = 2 * n + 1  # 2N coordinates + r
        
        if n_constraints < n_vars:
            # Underdetermined; use least-squares approach
            pass
        
        # Residual and Jacobian
        def _residual_jacobian(var_vec):
            """Compute residual and Jacobian for the active constraint system."""
            pts_var = var_vec[:2 * n].reshape(n, 2)
            r_var = var_vec[2 * n]
            
            F = np.zeros(n_constraints)
            J = np.zeros((n_constraints, n_vars))
            
            # Pair constraints: |c_i - c_j|^2 - 4r^2 = 0
            for k, pidx in enumerate(active_pairs):
                i, j = iu[0][pidx], iu[1][pidx]
                d = pts_var[i] - pts_var[j]
                F[k] = np.dot(d, d) - 4 * r_var ** 2
                # dF/d(pts_i) = 2*d, dF/d(pts_j) = -2*d, dF/dr = -8r
                J[k, 2 * i:2 * i + 2] = 2 * d
                J[k, 2 * j:2 * j + 2] = -2 * d
                J[k, 2 * n] = -8 * r_var
            
            # Boundary constraints: |c_i|^2 + 2r|c_i| + r^2 - 1 = 0
            # Simpler: |c_i| + r - 1 = 0
            for k, bidx in enumerate(active_boundary):
                i = bidx
                norm_i = np.linalg.norm(pts_var[i])
                F[n_pairs + k] = norm_i + r_var - 1.0
                if norm_i > 1e-10:
                    grad_pts = pts_var[i] / norm_i
                    J[n_pairs + k, 2 * i:2 * i + 2] = grad_pts
                J[n_pairs + k, 2 * n] = 1.0
            
            return F, J
        
        # Initial variable vector
        var_vec = np.zeros(n_vars)
        var_vec[:2 * n] = current.flatten()
        var_vec[2 * n] = r
        
        # Newton iterations
        newton_converged = False
        for newton_iter in range(20):
            F, J = _residual_jacobian(var_vec)
            res_norm = np.linalg.norm(F)
            if res_norm < tol:
                newton_converged = True
                break
            
            # Solve J * delta = -F (least squares if underdetermined)
            try:
                if n_constraints >= n_vars:
                    delta, _, _, _ = np.linalg.lstsq(J, -F, rcond=None)
                else:
                    # Underdetermined: find minimum-norm solution
                    delta, _, _, _ = np.linalg.lstsq(J.T, -F, rcond=None)
                    # Actually for underdetermined, we want min-norm solution of J*delta = -F
                    # Use SVD or just regularize
                    reg = 1e-8 * np.eye(n_vars)
                    delta = np.linalg.solve(J.T @ J + reg, J.T @ (-F))
            except np.linalg.LinAlgError:
                break
            
            # Line search: find alpha such that ||F(var + alpha*delta)|| < ||F(var)||
            alpha = 1.0
            for ls_iter in range(30):
                var_trial = var_vec + alpha * delta
                # Project points to disk
                pts_trial = var_trial[:2 * n].reshape(n, 2)
                norms_trial = np.linalg.norm(pts_trial, axis=1)
                mask = norms_trial > 1.0
                if np.any(mask):
                    pts_trial[mask] = pts_trial[mask] / norms_trial[mask, np.newaxis]
                    var_trial[:2 * n] = pts_trial.flatten()
                
                F_trial, _ = _residual_jacobian(var_trial)
                if np.linalg.norm(F_trial) < res_norm * (1 - 1e-4 * alpha):
                    break
                alpha *= 0.5
            
            var_vec = var_vec + alpha * delta
            
            # Project points to disk
            pts_check = var_vec[:2 * n].reshape(n, 2)
            norms_check = np.linalg.norm(pts_check, axis=1)
            mask = norms_check > 1.0
            if np.any(mask):
                pts_check[mask] = pts_check[mask] / norms_check[mask, np.newaxis]
                var_vec[:2 * n] = pts_check.flatten()
        
        if newton_converged:
            current = var_vec[:2 * n].reshape(n, 2)
            current = _project_to_disk(current)
            new_score = _score(current)
            old_score = _score(pts)
            if new_score > old_score:
                current = _project_to_disk(current)
                return current
        else:
            # Newton didn't converge; check if we made progress anyway
            current_trial = var_vec[:2 * n].reshape(n, 2)
            current_trial = _project_to_disk(current_trial)
            if _score(current_trial) > _score(pts):
                return current_trial
    
    return pts

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(8):
        inits.append(_random_init())
        
    best_pts = inits[0]
    best_score = _score(best_pts)
    
    # Phase 1: For each init, do force-directed relaxation then L-BFGS-B with adaptive beta
    betas = [10, 50, 200, 1000]
    
    for init_pts in inits:
        if time.time() > _DEADLINE - 25.0:
            break
            
        current_pts = init_pts.copy()
        
        # Force-directed relaxation to get a better starting point
        current_pts = _force_directed_relax(current_pts, max_steps=500)
        s = _score(current_pts)
        if s > best_score:
            best_score = s
            best_pts = current_pts.copy()
            
        # L-BFGS-B with increasing beta
        for beta in betas:
            if time.time() > _DEADLINE - 20.0:
                break
                
            new_pts = _optimize_lbfgs(current_pts, beta, maxiter=300, gtol=1e-8)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
                
            # If we got a good improvement, do a short force-directed pass to escape
            if new_score > best_score - 1e-6 and time.time() < _DEADLINE - 25.0:
                escaped = _force_directed_relax(current_pts, max_steps=200)
                esc_score = _score(escaped)
                if esc_score > best_score:
                    best_score = esc_score
                    best_pts = escaped.copy()
                    current_pts = escaped.copy()
        
        # Check time
        if time.time() > _DEADLINE - 20.0:
            break
            
    # Phase 2: Multiple final polish passes with very high beta on the best found
    polish_betas = [2000, 5000, 10000, 20000, 50000, 100000]
    for beta in polish_betas:
        if time.time() > _DEADLINE - 5.0:
            break
        new_pts = _optimize_lbfgs(best_pts, beta, maxiter=500, gtol=1e-12)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    
    # Phase 2b: Newton active-set refinement on the best solution
    if time.time() < _DEADLINE - 5.0:
        new_pts = _newton_active_set(best_pts, max_iter=30, tol=1e-13)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    
    # Phase 3: Perturbation-based local search near the best solution
    if time.time() < _DEADLINE - 10.0:
        for sigma in [0.02, 0.01, 0.005, 0.002, 0.001]:
            if time.time() > _DEADLINE - 5.0:
                break
            for attempt in range(6):
                if time.time() > _DEADLINE - 5.0:
                    break
                perturbed = _perturb(best_pts, sigma)
                perturbed = _project_to_disk(perturbed)
                
                # Force-directed escape
                perturbed = _force_directed_relax(perturbed, max_steps=300)
                
                # Quick polish with moderate beta
                new_pts = _optimize_lbfgs(perturbed, 500, maxiter=300, gtol=1e-10)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
    
    # Phase 4: Final Newton polish on the very best
    if time.time() < _DEADLINE - 3.0:
        final_pts = _newton_active_set(best_pts, max_iter=50, tol=1e-14)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    return best_pts
