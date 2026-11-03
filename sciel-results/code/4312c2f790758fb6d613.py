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
    # Query pairs within a distance that is likely to contain the minimum.
    # Since we are maximizing the minimum distance, the current min distance is an upper bound for the search radius if we just want to find *the* min.
    # However, for radius calculation, we just need the global min.
    # A safe radius for finding the global min in a unit disk with 300 points is small.
    # But cKDTree.query_pairs requires a radius. If we set it too small, we might miss the global min if points are far apart.
    # Actually, for radius calculation, we can just use query_pairs with a large enough radius, or just compute full matrix if N is small?
    # N=300 is small enough that full matrix is ~0.1ms. The bottleneck is the optimization loops.
    # So _compute_radius_fast doesn't need to be super optimized, but let's use tree anyway for consistency.
    # To find the *minimum* distance, we need to ensure the query radius is at least the minimum distance.
    # We don't know the minimum distance a priori.
    # Alternative: Use tree.query with k=2 to get the nearest neighbor for each point.
    # The global minimum distance is min over i of dist(i, NN(i)).
    
    dists, _ = tree.query(pts, k=2)
    # dists[:, 0] is 0 (self), dists[:, 1] is distance to nearest neighbor
    min_pair_dist = np.min(dists[:, 1])
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
    Uses KD-tree for neighbor search to speed up force calculation.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    # Initialize velocity to zero
    v = np.zeros_like(pts)
    pos = pts.copy()
    
    threshold = 0.15
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Use KD-tree to find neighbors within threshold
        tree = cKDTree(pos)
        pairs = tree.query_pairs(threshold)
        
        for i, j in pairs:
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            overlap = threshold - dist
            if overlap > 0:
                force_mag = k_repel * (overlap ** 4)
                unit_vec = diff / dist
                forces[i] += force_mag * unit_vec
                forces[j] -= force_mag * unit_vec
        
        # Boundary force: push inward if too close to edge
        norms = np.hypot(pos[:, 0], pos[:, 1])
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
        norms_cand = np.hypot(pos[:, 0], pos[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            pos[too_far] *= 0.9999 / norms_cand[too_far]
            v[too_far] *= 0.5
        
        # Check improvement
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
    Gradient ascent on soft-min surrogate using log-sum-exp.
    Uses KD-tree to compute gradients only for relevant neighbors.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        progress = it / max_iter
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        # We need to consider all pairs for the gradient sum? 
        # The term is sum(exp(-eps * dist)). If eps is large, only close pairs matter.
        # If eps is small, all pairs matter.
        # To be safe and accurate, we might still need full matrix for small eps.
        # However, for eps > 50, pairs with dist > 1/eps * 10 contribute negligibly.
        # Let's use a cutoff radius R_cut = 10 / eps.
        # If R_cut > 1.0, we use full matrix. Otherwise, use KD-tree.
        
        R_cut = 10.0 / eps
        if R_cut > 1.0:
             # Fall back to full matrix for small eps
             diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
             dists = np.sqrt(np.sum(diff ** 2, axis=2))
             np.fill_diagonal(dists, np.inf)
             
             weights_pair = np.exp(-eps * dists)
             np.fill_diagonal(weights_pair, 0.0)
             
             safe_dists = np.where(dists > 1e-12, dists, 1.0)
             unit_vecs = diff / safe_dists[:, :, np.newaxis]
             
             grad_pair = -eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)
        else:
             # Use KD-tree
             tree = cKDTree(best_pts)
             pairs = tree.query_pairs(R_cut)
             
             grad_pair = np.zeros_like(best_pts)
             
             for i, j in pairs:
                 diff = best_pts[i] - best_pts[j]
                 dist = np.linalg.norm(diff)
                 if dist < 1e-12:
                     continue
                 weight = np.exp(-eps * dist)
                 unit_vec = diff / dist
                 force_vec = -eps * weight * unit_vec
                 grad_pair[i] += force_vec
                 grad_pair[j] -= force_vec
             
        # Boundary distances
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        b_dists = 1.0 - radii
        
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
            # Reduce learning rate slightly if no improvement
            lr *= 0.99
    
    return best_pts, best_r


def _polish(pts, deadline):
    """
    Final high-precision polish with very small steps.
    Uses KD-tree for efficiency.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    
    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
        if deadline is not None and time.time() > deadline:
            break
        
        for it in range(300):
            if deadline is not None and time.time() > deadline:
                break
            
            # Use soft-min with high eps
            eps = 200.0
            R_cut = 10.0 / eps # 0.05
            
            tree = cKDTree(best_pts)
            pairs = tree.query_pairs(R_cut)
            
            grad_pair = np.zeros_like(best_pts)
            
            for i, j in pairs:
                diff = best_pts[i] - best_pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                weight = np.exp(-eps * dist)
                unit_vec = diff / dist
                force_vec = -eps * weight * unit_vec
                grad_pair[i] += force_vec
                grad_pair[j] -= force_vec
            
            radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
            b_dists = 1.0 - radii
            
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
            max_steps=3000, 
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
            eps_start=50.0,
            eps_end=500.0,
            lr_start=0.002,
            lr_end=1e-5,
            max_iter=3000
        )
    
    # Phase 3: Random perturbations + re-optimize to escape local minima
    if time.time() < deadline - 10.0:
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            perturbed = _random_perturb(best_pts, magnitude=0.005)
            pert_r = _compute_radius_fast(perturbed)
            if pert_r >= best_r * 0.995:
                # Try Hertzian dynamics on perturbed version
                p_pts, p_r = _hertzian_dynamics(
                    perturbed,
                    max_steps=1500,
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
                    eps_start=100.0,
                    eps_end=400.0,
                    lr_start=0.001,
                    lr_end=5e-6,
                    max_iter=1500
                )
                if p_r > best_r:
                    best_pts = p_pts
                    best_r = p_r
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 1.0:
        best_pts, best_r = _polish(best_pts, deadline - 1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
