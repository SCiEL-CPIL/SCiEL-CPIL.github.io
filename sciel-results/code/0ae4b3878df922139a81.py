"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the exact packing radius for given centers."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    
    # Radial distances
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists > 1.0):
        return -1.0
        
    # Pairwise distances using upper triangle
    i_idx, j_idx = np.triu_indices(N, k=1)
    diffs = points[i_idx] - points[j_idx]
    pair_dists = np.linalg.norm(diffs, axis=1)
    min_pair_dist = np.min(pair_dists)
    min_radial_dist = 1.0 - np.max(dists)
    r = min(min_pair_dist / 2.0, min_radial_dist)
    return r

def _generate_hex_initialization():
    """Generate a hexagonal lattice clipped to the unit circle."""
    best_pts = None
    best_score = -1.0

    for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
        spacing = 2.0 / (np.sqrt(N) * scale)
        nx = int(2.0 / spacing) + 3
        ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3

        x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0

        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                    pts_list.append([px, py])

        if len(pts_list) < N:
            continue

        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:N]]

        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts

    if best_pts is None:
        rng = np.random.RandomState(42)
        best_pts = rng.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
        best_score = _compute_score(best_pts)

    return best_pts, best_score

def _generate_ring_initialization():
    """Generate points in concentric rings."""
    best_pts = None
    best_score = -1.0
    
    for num_rings in range(10, 25):
        ring_radii = np.linspace(0.05, 0.95, num_rings)
        ring_points = []
        
        # Estimate radius to distribute points
        r_est = 1.0 / np.sqrt(N)
        
        for r_ring in ring_radii:
            n_circ = max(1, int(2 * np.pi * r_ring / (2 * r_est)))
            angles = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            for a in angles:
                x = r_ring * np.cos(a)
                y = r_ring * np.sin(a)
                ring_points.append([x, y])
        
        ring_points.append([0.0, 0.0])
        ring_points = np.array(ring_points)
        
        if len(ring_points) > N:
            dists = np.linalg.norm(ring_points, axis=1)
            idx = np.argsort(dists)[:N]
            ring_points = ring_points[idx]
        elif len(ring_points) < N:
            continue
            
        score = _compute_score(ring_points)
        if score > best_score:
            best_score = score
            best_pts = ring_points.copy()
            
    if best_pts is None:
        # Fallback to hex
        best_pts, best_score = _generate_hex_initialization()
        
    return best_pts, best_score

def _soft_min_obj_grad(x, beta):
    """
    Compute objective (negative soft-min) and gradient.
    x: shape (2*N,)
    beta: inverse temperature
    """
    p = x.reshape(-1, 2)
    n = len(p)
    
    # Pairwise distances
    iu = np.triu_indices(n, k=1)
    diff = p[iu[0]] - p[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    pair_vals = dists / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    
    all_vals = np.concatenate([pair_vals, boundary_dists])
    
    # Soft-min calculation: -1/beta * log(sum(exp(-beta * vals)))
    # Objective is -soft_min = 1/beta * log(sum(exp(-beta * vals)))
    z = -beta * all_vals
    max_z = np.max(z)
    exp_vals = np.exp(z - max_z)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(x)
        
    log_sum_exp = max_z + np.log(sum_exp)
    objective = log_sum_exp / beta
    
    # Weights for gradient
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(p)
    
    # Pair contributions
    # d/dp_i (dist/2) = 0.5 * (p_i - p_j) / dist
    # grad_obj = - sum w_k * grad(d_k)
    # For pair ij:
    # grad_{p_i} obj += - w_ij * 0.5 * (p_i - p_j) / dist
    # grad_{p_j} obj += - w_ij * 0.5 * (p_j - p_i) / dist
    
    w_pairs = weights[:len(pair_vals)]
    i_idx, j_idx = iu
    
    # Safe division
    safe_dists = np.where(pair_vals > 1e-12, pair_vals, 1.0)
    coeff = w_pairs * 0.5 / safe_dists
    vecs = diff # p[i] - p[j]
    
    # grad p_i += - coeff * vecs
    # grad p_j += - coeff * (-vecs) = + coeff * vecs
    
    np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
    
    # Boundary contributions
    # d/dp_i (1 - |p_i|) = - p_i / |p_i|
    # grad_obj += - w_i * (- p_i / |p_i|) = w_i * p_i / |p_i|
    
    w_bounds = weights[len(pair_vals):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    return objective, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Initializations
    hex_pts, hex_score = _generate_hex_initialization()
    ring_pts, ring_score = _generate_ring_initialization()
    
    candidates = []
    if hex_score > 0:
        candidates.append((hex_pts, hex_score))
    if ring_score > 0:
        candidates.append((ring_pts, ring_score))
        
    if not candidates:
        # Emergency fallback
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        pts = pts * (0.9 / norms)[:, None]
        return pts
        
    # Sort candidates by score, best first
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_points = candidates[0][0].copy()
    best_score = candidates[0][1]
    
    # 2. Optimization Loop
    # Beta schedule: start low for global exploration, increase for local precision
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    # Try optimizing the best candidate first
    current_pts = best_points.copy()
    current_score = best_score
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        x0 = current_pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            # L-BFGS-B with analytical gradient
            res = minimize(
                _soft_min_obj_grad, x0, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.status in [1, 2, 3, 4]: # Common statuses
                new_points = res.x.reshape(-1, 2)
                new_score = _compute_score(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    current_pts = new_points.copy()
                    current_score = new_score
                    
        except Exception:
            pass
            
    # 3. Try the second best candidate if time remains
    if len(candidates) > 1 and time.time() < deadline - 30:
        second_pts, second_score = candidates[1]
        current_pts = second_pts.copy()
        
        # Use higher betas for polishing
        betas_polish = [50.0, 100.0, 200.0, 500.0]
        
        for beta in betas_polish:
            if time.time() > deadline - 5:
                break
                
            x0 = current_pts.flatten()
            bounds = [(-1.0, 1.0)] * (2 * N)
            
            try:
                res = minimize(
                    _soft_min_obj_grad, x0, 
                    args=(beta,), 
                    method='L-BFGS-B', 
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.status in [1, 2, 3, 4]:
                    new_points = res.x.reshape(-1, 2)
                    new_score = _compute_score(new_points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        current_pts = new_points.copy()
                        
            except Exception:
                pass

    return best_points
# EVOLVE-BLOCK-END
