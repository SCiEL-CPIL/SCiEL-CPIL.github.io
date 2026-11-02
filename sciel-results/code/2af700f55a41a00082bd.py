"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_lattice_init():
    """Generate hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    return points

def _ring_init():
    """Generate ring-based initialization with multiple concentric rings."""
    points = np.array([[0.0, 0.0]])
    remaining = N - 1
    r = 0.0
    ring_idx = 0
    while remaining > 0 and r < 0.95:
        # Estimate how many points fit on this ring
        circumference = 2 * np.pi * r
        # Points need to be at least 2r apart (rough estimate)
        est_spacing = 0.06
        n_on_ring = max(1, int(circumference / est_spacing)) if r > 0.01 else 6
        n_on_ring = min(n_on_ring, remaining)
        
        for i in range(n_on_ring):
            angle = 2 * np.pi * i / n_on_ring + ring_idx * 0.1
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
            remaining -= 1
            if remaining == 0:
                break
        r += 0.05
        ring_idx += 1
    
    # Pad or trim to exactly N points
    if len(points) < N:
        extra = np.random.randn(N - len(points), 2) * 0.1
        points = np.vstack([points, extra])
    elif len(points) > N:
        points = points[:N]
    
    return points

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _objective_and_grad(x_flat, beta):
    """
    Objective: -soft_min(pairwise_dists/2, boundary_dists)
    Gradient: computed analytically.
    """
    x = x_flat.reshape(-1, 2)
    n = len(x)
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # We only need i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(x, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min
    y_min = np.min(all_dists)
    exps = np.exp(-beta * (all_dists - y_min))
    sum_exps = np.sum(exps)
    soft_min_val = y_min - (1.0 / beta) * np.log(sum_exps)
    
    # Objective is -soft_min_val
    obj = -soft_min_val
    
    # Gradient
    probs = exps / sum_exps
    
    grad_flat = np.zeros_like(x_flat)
    
    # Gradient of pairwise distances
    safe_dists = np.where(dists < 1e-8, 1e-8, dists)
    
    # We need to map pair index back to (i, j) to build the gradient matrix.
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    A = np.zeros((n, n))
    p_pairs = probs[:len(pair_dists)]
    d_pairs = dists[i_idx, j_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1e-8, d_pairs)
    weights = p_pairs / (2.0 * safe_d_pairs)
    
    A[i_idx, j_idx] = weights
    A[j_idx, i_idx] = weights
    
    row_sums = A.sum(axis=1)
    Ax = A @ x
    
    grad_pairwise = x * row_sums[:, np.newaxis] - Ax
    
    # Gradient of boundary distances
    p_bound = probs[len(pair_dists):]
    safe_norms = np.where(norms < 1e-8, 1e-8, norms)
    grad_bound = -p_bound[:, np.newaxis] * x / safe_norms[:, np.newaxis]
    
    total_grad = grad_pairwise + grad_bound
    
    # Objective is -soft_min, so gradient is -total_grad
    return obj, -total_grad.flatten()

def _force_directed_relax(points, max_iters=200, dt=0.001):
    """
    Force-directed relaxation to push overlapping circles apart and keep them inside.
    This is a local polish step after gradient optimization.
    """
    n = len(points)
    points = points.copy()
    current_score = _compute_score(points)
    
    for it in range(max_iters):
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion for overlapping or close circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    # Random small push
                    forces[i] += np.random.randn(2) * 1e-6
                    forces[j] -= np.random.randn(2) * 1e-6
                    continue
                # If circles are too close, push apart
                if dist < 0.11:  # threshold based on expected radius ~0.055
                    strength = (0.11 - dist) / 0.11
                    force_dir = diff / dist
                    forces[i] += strength * force_dir
                    forces[j] -= strength * force_dir
        
        # Boundary constraint: push inward if too close to edge
        norms = np.linalg.norm(points, axis=1)
        for i in range(n):
            if norms[i] > 0.95:
                force_dir = -points[i] / (norms[i] + 1e-8)
                strength = (norms[i] - 0.95) / 0.05
                forces[i] += strength * force_dir
        
        # Update positions with adaptive step size
        forces_norm = np.linalg.norm(forces, axis=1)
        max_force = np.max(forces_norm) if len(forces_norm) > 0 else 1.0
        if max_force > 1e-8:
            step = dt / max_force
            points += forces * step
        
        # Project inside unit circle
        norms = np.linalg.norm(points, axis=1)
        mask_out = norms > 0.999
        if np.any(mask_out):
            points[mask_out] *= 0.999 / norms[mask_out, np.newaxis]
        
        # Check if improved
        new_score = _compute_score(points)
        if new_score > current_score:
            current_score = new_score
        elif current_score - new_score > 1e-10:
            # Revert if significantly worse
            points -= forces * step
            norms = np.linalg.norm(points, axis=1)
            mask_out = norms > 0.999
            if np.any(mask_out):
                points[mask_out] *= 0.999 / norms[mask_out, np.newaxis]
            break
        
        # Convergence check
        if np.max(forces_norm) < 1e-6:
            break
    
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Multiple initializations
    np.random.seed(42)
    inits = [_hex_lattice_init()]
    for _ in range(2):
        inits.append(_ring_init())
    
    best_points = None
    best_score = 0.0
    
    for init_points in inits:
        if time.time() > deadline - 30.0:
            break
        
        # 2. L-BFGS Optimization with Soft Min
        betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0, 10000.0]
        
        x_current = init_points.flatten()
        
        for beta in betas:
            if time.time() > deadline - 15.0:
                break
            
            def obj_grad(x):
                return _objective_and_grad(x, beta)
            
            try:
                result = minimize(
                    lambda x: obj_grad(x)[0],
                    x_current,
                    jac=lambda x: obj_grad(x)[1],
                    method='L-BFGS-B',
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                x_current = result.x
            except Exception:
                pass
        
        # 3. Force-directed polish
        points_fd, score_fd = _force_directed_relax(x_current.reshape(-1, 2), max_iters=300, dt=0.0005)
        
        # 4. Final L-BFGS with very high beta for precision
        if time.time() < deadline - 5.0:
            x_final = points_fd.flatten()
            for beta in [50000.0, 100000.0]:
                if time.time() > deadline - 3.0:
                    break
                def obj_grad_final(x):
                    return _objective_and_grad(x, beta)
                try:
                    result = minimize(
                        lambda x: obj_grad_final(x)[0],
                        x_final,
                        jac=lambda x: obj_grad_final(x)[1],
                        method='L-BFGS-B',
                        bounds=[(-1.0, 1.0)] * (2 * N),
                        options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-12}
                    )
                    x_final = result.x
                except Exception:
                    pass
            
            points_final = x_final.reshape(-1, 2)
            score_final = _compute_score(points_final)
            
            if score_final > best_score:
                best_score = score_final
                best_points = points_final
        else:
            if score_fd > best_score:
                best_score = score_fd
                best_points = points_fd
    
    # 5. Try random perturbations from best solution if time permits
    if best_points is not None and time.time() < deadline - 8.0:
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            perturb = np.random.randn(2 * N) * 0.005
            x_pert = best_points.flatten() + perturb
            try:
                def obj_grad_pert(x):
                    return _objective_and_grad(x, 50000.0)
                res = minimize(
                    lambda x: obj_grad_pert(x)[0],
                    x_pert,
                    jac=lambda x: obj_grad_pert(x)[1],
                    method='L-BFGS-B',
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 80, 'ftol': 1e-16, 'gtol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                score = _compute_score(cand)
                if score > best_score:
                    best_score = score
                    best_points = cand
            except Exception:
                pass
    
    return best_points if best_points is not None else _hex_lattice_init()
# EVOLVE-BLOCK-END
