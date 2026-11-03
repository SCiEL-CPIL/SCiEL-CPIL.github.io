"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """
    Compute the exact minimum common radius.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    n = len(pts)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
    return min(min_pair_dist / 2.0, min_boundary)


def _hex_init(n):
    """Generate a hexagonal lattice packing of n points in the unit disk."""
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95
    points = []
    max_coord = int(1.0 / s) + 2
    for j in range(-max_coord, max_coord + 1):
        for i in range(-max_coord, max_coord + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if x * x + y * y <= (1.0 - s / 2) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < n:
        s *= 0.9
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if x * x + y * y <= (1.0 - s / 2) ** 2:
                    points.append([x, y])
        points = np.array(points)
    if len(points) >= n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _hertzian_dynamics(pts, deadline, max_steps=8000, dt=0.002, k_repel=200.0,
                       k_boundary=80.0, damping=0.98, target_margin=0.01):
    """
    Hertzian contact dynamics simulation using Verlet integration.
    This is a structural change: instead of gradient ascent on a soft-min surrogate,
    we simulate physical forces between overlapping circles.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)
    
    v = np.zeros_like(pts)
    pos = pts.copy()
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion (Hertzian: F ~ overlap^4 * direction)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Apply force when dist < threshold (use a scale based on expected spacing)
        # For N=300 in unit disk, typical spacing ~ sqrt(pi/300) ~ 0.1
        threshold = 0.12
        mask = dists < threshold
        np.fill_diagonal(mask, False)
        
        overlap = np.where(mask, threshold - dists, 0.0)
        force_mag = k_repel * (overlap ** 4)
        
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Boundary force: push inward if too close to edge
        norms = np.linalg.norm(pos, axis=1)
        margin = 1.0 - norms
        boundary_active = margin < target_margin
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        inward_dir = -pos / safe_norms[:, np.newaxis]
        boundary_force_mag = np.where(boundary_active, k_boundary * ((target_margin - margin) / target_margin) ** 2, 0.0)
        forces += inward_dir * boundary_force_mag[:, np.newaxis]
        
        # Verlet integration with damping
        v = damping * v + dt * forces
        pos = pos + dt * v
        
        # Enforce hard boundary constraint
        norms_cand = np.linalg.norm(pos, axis=1)
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            pos[too_far] *= 0.9999 / norms_cand[too_far]
            v[too_far] *= 0.5
        
        # Check improvement
        cand_r = _compute_radius(pos)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = pos.copy()
        
        # Convergence check
        force_norm = np.linalg.norm(forces, axis=1)
        if np.max(force_norm) < 1e-8:
            break
    
    return best_pts, best_r


def _soft_min_optimize(pts, deadline, eps_start=30.0, eps_end=300.0,
                       lr_start=0.003, lr_end=1e-5, max_iter=3000):
    """
    Gradient ascent on soft-min surrogate using log-sum-exp.
    This is the refinement machinery from the inspiration.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        progress = it / max_iter
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute pairwise distances
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.linalg.norm(best_pts, axis=1)
        b_dists = 1.0 - radii
        
        # Pairwise gradient: minimize sum(exp(-eps * dist_ij))
        # dL/dpos_i = -eps * sum_j exp(-eps * dist_ij) * (pos_i - pos_j) / dist_ij
        weights_pair = np.exp(-eps * dists)
        np.fill_diagonal(weights_pair, 0.0)
        
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        grad_pair = -eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Boundary gradient: minimize sum(exp(-eps * b_i))
        # dL/dpos_i = eps * exp(-eps * b_i) * pos_i / |pos_i|
        weights_bound = np.exp(-eps * b_dists)
        safe_norms = np.where(radii > 1e-12, radii, 1.0)
        unit_radii = best_pts / safe_norms[:, np.newaxis]
        
        grad_bound = eps * weights_bound[:, np.newaxis] * unit_radii
        
        # Total gradient
        total_grad = grad_pair + grad_bound
        
        # Normalize for stability
        grad_norm = np.linalg.norm(total_grad, axis=1)
        grad_norm[grad_norm < 1e-12] = 1.0
        total_grad /= grad_norm[:, np.newaxis]
        
        # Update
        candidate = best_pts + lr * total_grad
        
        # Enforce boundary
        norms_cand = np.linalg.norm(candidate, axis=1)
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            candidate[too_far] *= 0.9999 / norms_cand[too_far]
        
        cand_r = _compute_radius(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = candidate.copy()
        else:
            lr *= 0.99
    
    return best_pts, best_r


def _polish(pts, deadline):
    """
    Final high-precision polish with very small steps and high eps.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)
    
    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
        if time.time() > deadline:
            break
        
        for it in range(200):
            if time.time() > deadline:
                break
            
            eps = 200.0
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            radii = np.linalg.norm(best_pts, axis=1)
            b_dists = 1.0 - radii
            
            weights_pair = np.exp(-eps * dists)
            np.fill_diagonal(weights_pair, 0.0)
            
            safe_dists = np.where(dists > 1e-12, dists, 1.0)
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            
            grad_pair = -eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)
            
            weights_bound = np.exp(-eps * b_dists)
            safe_norms = np.where(radii > 1e-12, radii, 1.0)
            unit_radii = best_pts / safe_norms[:, np.newaxis]
            
            grad_bound = eps * weights_bound[:, np.newaxis] * unit_radii
            
            total_grad = grad_pair + grad_bound
            grad_norm = np.linalg.norm(total_grad, axis=1)
            grad_norm[grad_norm < 1e-12] = 1.0
            total_grad /= grad_norm[:, np.newaxis]
            
            candidate = best_pts + lr * total_grad
            norms_cand = np.linalg.norm(candidate, axis=1)
            too_far = norms_cand > 0.9999
            if np.any(too_far):
                candidate[too_far] *= 0.9999 / norms_cand[too_far]
            
            cand_r = _compute_radius(candidate)
            if cand_r > best_r:
                best_r = cand_r
                best_pts = candidate.copy()
    
    return best_pts, best_r


def construct():
    """
    Construct 300 equal circle centers inside the unit disk to maximize the minimum
    common radius. Combines Hertzian dynamics (structural exploration) with
    soft-min gradient ascent (refinement machinery).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Phase 1: Hertzian dynamics for structural exploration
    pts = _hex_init(N)
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / norms[mask] * 0.999
    
    time_budget1 = deadline - 60
    pts, score = _hertzian_dynamics(pts, time_budget1)
    best_pts = pts
    best_score = score
    
    # Phase 2: Soft-min gradient ascent for refinement
    if time.time() < deadline - 30:
        time_budget2 = deadline - 20
        pts2, score2 = _soft_min_optimize(best_pts, time_budget2)
        if score2 > best_score:
            best_pts = pts2
            best_score = score2
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 5:
        time_budget3 = deadline - 2
        best_pts, best_score = _polish(best_pts, time_budget3)
    
    # Final validation
    if best_pts is None:
        best_pts = _hex_init(N)
    
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / norms[mask] * 0.99
    
    return best_pts
# EVOLVE-BLOCK-END
