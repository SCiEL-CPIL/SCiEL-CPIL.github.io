"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_radius_fast(pts):
    """
    Compute the packing radius for given points using KD-tree for efficiency.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    if len(pts) < 2:
        return 0.0
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1.0)  # pairs within distance 1.0 (should capture all relevant)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        pair_list = list(pairs)
        min_pair_dist = np.inf
        for i, j in pair_list:
            d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            if d < min_pair_dist:
                min_pair_dist = d
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _init_hex(spacing):
    """Generate hexagonal lattice points within unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))


def _hertzian_dynamics(pts, max_steps=5000, dt=0.001, k_repel=100.0, k_boundary=50.0,
                       damping=0.98, deadline=None, target_margin=0.005):
    """
    Hertzian contact dynamics simulation using Verlet integration.
    Repulsive force proportional to overlap^4 (Hertzian law).
    Uses KD-tree for neighbor finding to avoid O(N^2) pairwise computation.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    v = np.zeros_like(pts)
    pos = pts.copy()
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(pos)
        
        # Use KD-tree to find nearby pairs (within threshold)
        threshold = 0.15
        tree = cKDTree(pos)
        pairs = tree.query_pairs(threshold)
        
        for i, j in pairs:
            diff = pos[i] - pos[j]
            dist = np.hypot(diff[0], diff[1])
            if dist > 1e-12:
                overlap = max(0.0, threshold - dist)
                if overlap > 0:
                    force_mag = k_repel * (overlap ** 4)
                    unit_vec = diff / dist
                    forces[i] += force_mag * unit_vec
                    forces[j] -= force_mag * unit_vec
        
        # Boundary force
        norms = np.hypot(pos[:, 0], pos[:, 1])
        margin = 1.0 - norms
        boundary_active = margin < target_margin
        if np.any(boundary_active):
            safe_norms = np.where(norms[boundary_active] > 1e-12, norms[boundary_active], 1.0)
            inward_dir = -pos[boundary_active] / safe_norms[:, np.newaxis]
            boundary_force_mag = k_boundary * ((target_margin - margin[boundary_active]) / target_margin) ** 2
            forces[boundary_active] += inward_dir * boundary_force_mag[:, np.newaxis]
        
        # Verlet integration with damping
        v = damping * v + dt * forces
        pos = pos + dt * v
        
        # Enforce hard boundary constraint
        norms_cand = np.hypot(pos[:, 0], pos[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            pos[too_far] *= 0.9999 / norms_cand[too_far]
            v[too_far] *= 0.5
        
        # Check improvement every 50 steps to save time
        if step % 50 == 0 or step == max_steps - 1:
            cand_r = _compute_radius_fast(pos)
            if cand_r > best_r:
                best_r = cand_r
                best_pts = pos.copy()
        
        # Convergence check
        force_norm = np.hypot(forces[:, 0], forces[:, 1])
        if np.max(force_norm) < 1e-8:
            break
    
    return best_pts, best_r


def _soft_min_optimize(pts, deadline, eps_start=50.0, eps_end=500.0, 
                       lr_start=0.002, lr_end=1e-5, max_iter=4000):
    """
    Gradient ascent on soft-min surrogate using log-sum-exp with threshold masking.
    Only considers pairs within a distance threshold to reduce computation.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    # Threshold for considering pairs (pairs farther than this don't contribute significantly)
    pair_threshold = 0.3
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        progress = it / max_iter
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # Compute pairwise distances only for nearby pairs
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        np.fill_diagonal(dists, np.inf)
        
        # Mask: only consider pairs within threshold
        mask = dists < pair_threshold
        np.fill_diagonal(mask, False)
        
        # Boundary distances
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        b_dists = 1.0 - radii
        
        # Pairwise gradient: minimize sum(exp(-eps * dist_ij)) for nearby pairs
        weights_pair = np.where(mask, np.exp(-eps * dists), 0.0)
        np.fill_diagonal(weights_pair, 0.0)
        
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        grad_pair = -eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Boundary gradient
        weights_bound = np.exp(-eps * b_dists)
        safe_norms = np.where(radii > 1e-12, radii, 1.0)
        unit_radii = best_pts / safe_norms[:, np.newaxis]
        
        grad_bound = eps * weights_bound[:, np.newaxis] * unit_radii
        
        # Total gradient
        total_grad = grad_pair + grad_bound
        
        # Normalize for stability
        grad_norm = np.hypot(total_grad[:, 0], total_grad[:, 1])
        grad_norm[grad_norm < 1e-12] = 1.0
        total_grad /= grad_norm[:, np.newaxis]
        
        # Update
        candidate = best_pts + lr * total_grad
        
        # Enforce boundary
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            candidate[too_far] *= 0.9999 / norms_cand[too_far]
        
        cand_r = _compute_radius_fast(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = candidate.copy()
        else:
            lr *= 0.99
    
    return best_pts, best_r


def _polish(pts, deadline):
    """
    Final high-precision polish with very small steps and threshold masking.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    
    # Use a smaller threshold for polish since points are close to optimal
    pair_threshold = 0.2
    
    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
        if deadline is not None and time.time() > deadline:
            break
        
        for it in range(300):
            if deadline is not None and time.time() > deadline:
                break
            
            eps = 200.0
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            dists = np.sqrt(sq_dists)
            np.fill_diagonal(dists, np.inf)
            
            # Mask for nearby pairs
            mask = dists < pair_threshold
            np.fill_diagonal(mask, False)
            
            radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
            b_dists = 1.0 - radii
            
            weights_pair = np.where(mask, np.exp(-eps * dists), 0.0)
            np.fill_diagonal(weights_pair, 0.0)
            
            safe_dists = np.where(dists > 1e-12, dists, 1.0)
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            
            grad_pair = -eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)
            
            weights_bound = np.exp(-eps * b_dists)
            safe_norms = np.where(radii > 1e-12, radii, 1.0)
            unit_radii = best_pts / safe_norms[:, np.newaxis]
            
            grad_bound = eps * weights_bound[:, np.newaxis] * unit_radii
            
            total_grad = grad_pair + grad_bound
            
            grad_norm = np.hypot(total_grad[:, 0], total_grad[:, 1])
            grad_norm[grad_norm < 1e-12] = 1.0
            total_grad /= grad_norm[:, np.newaxis]
            
            candidate = best_pts + lr * total_grad
            
            norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
            too_far = norms_cand > 0.99999
            if np.any(too_far):
                candidate[too_far] *= 0.99999 / norms_cand[too_far]
            
            cand_r = _compute_radius_fast(candidate)
            if cand_r > best_r + 1e-15:
                best_r = cand_r
                best_pts = candidate.copy()
    
    return best_pts, best_r


def _random_perturb(pts, magnitude=0.008):
    """Add small random perturbation to escape local minima."""
    rng = np.random.default_rng()
    perturbed = pts + rng.normal(0, magnitude, size=pts.shape)
    norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
    too_far = norms > 0.99
    if np.any(too_far):
        perturbed[too_far] *= 0.99 / norms[too_far]
    return perturbed


def construct():
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Phase 0: Search over hexagonal spacings for good initialization
    best_pts = None
    best_r = 0.0
    
    spacings = np.linspace(0.09, 0.13, 30)
    for spacing in spacings:
        if time.time() > deadline:
            break
        pts = _init_hex(spacing)
        if len(pts) < N:
            continue
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius_fast(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    
    if best_pts is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind="stable")[:N]]
                best_r = _compute_radius_fast(best_pts)
                break
            spacing *= 0.98
    
    # Phase 1: Hertzian dynamics simulation (fast, coarse, physically motivated)
    if time.time() < deadline - 50.0:
        best_pts, best_r = _hertzian_dynamics(
            best_pts, 
            max_steps=5000, 
            dt=0.001, 
            k_repel=100.0, 
            k_boundary=50.0,
            damping=0.98,
            deadline=deadline - 50.0,
            target_margin=0.005
        )
    
    # Phase 2: Soft-min gradient optimization (finer, smooth surrogate)
    if time.time() < deadline - 25.0:
        best_pts, best_r = _soft_min_optimize(
            best_pts,
            deadline=deadline - 25.0,
            eps_start=3.0,
            eps_end=500.0,
            lr_start=0.002,
            lr_end=1e-5,
            max_iter=5000
        )
    
    # Phase 3: Random perturbations + re-optimize to escape local minima
    if time.time() < deadline - 10.0:
        for attempt in range(8):
            if time.time() > deadline - 10.0:
                break
            perturbed = _random_perturb(best_pts, magnitude=0.005)
            pert_r = _compute_radius_fast(perturbed)
            if pert_r >= best_r * 0.995:
                # Try Hertzian dynamics on perturbed version
                p_pts, p_r = _hertzian_dynamics(
                    perturbed,
                    max_steps=2000,
                    dt=0.0008,
                    k_repel=80.0,
                    k_boundary=40.0,
                    damping=0.985,
                    deadline=deadline - 10.0,
                    target_margin=0.003
                )
                # Follow with soft-min
                p_pts, p_r = _soft_min_optimize(
                    p_pts,
                    deadline=deadline - 10.0,
                    eps_start=50.0,
                    eps_end=400.0,
                    lr_start=0.001,
                    lr_end=5e-6,
                    max_iter=2000
                )
                if p_r > best_r:
                    best_pts = p_pts
                    best_r = p_r
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 1.0:
        best_pts, best_r = _polish(best_pts, deadline - 1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
