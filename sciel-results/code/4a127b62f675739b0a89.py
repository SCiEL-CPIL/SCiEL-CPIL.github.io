"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_radius_fast(pts):
    """
    Compute the packing radius for given points using vectorized operations.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    if len(pts) < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _compute_radius_and_active(pts, threshold_frac=0.02):
    """
    Compute packing radius and identify active (binding) constraints.
    Returns (r, active_pairs, active_boundary) where active_pairs is a list of (i,j) 
    with dist < r + threshold, and active_boundary is list of i with margin < threshold.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    norms = np.hypot(pts[:, 0], pts[:, 1])
    b_dists = 1.0 - norms
    min_boundary = np.min(b_dists)
    r = min(min_pair_dist / 2.0, min_boundary)
    
    threshold = r + threshold_frac
    active_pairs = []
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if dists[i,j] < 2*r + threshold_frac:
                active_pairs.append((i,j))
    active_boundary = []
    for i in range(len(pts)):
        if b_dists[i] < r + threshold_frac:
            active_boundary.append(i)
    return r, active_pairs, active_boundary


def _newton_refine(pts, deadline, max_iter=200, tol=1e-10):
    """
    Active-set Newton refinement: identify binding constraints and solve 
    a small linear system to push them apart precisely.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        r, active_pairs, active_boundary = _compute_radius_and_active(best_pts, threshold_frac=0.01)
        
        # If no active constraints or too many, skip
        if len(active_pairs) == 0 and len(active_boundary) == 0:
            break
        
        # Build the active constraint system
        # For each active pair (i,j): |p_i - p_j| >= 2r  =>  push apart along direction
        # For each active boundary i: |p_i| <= 1-r  =>  push inward
        
        # Collect active points
        active_indices = set()
        for i,j in active_pairs:
            active_indices.add(i)
            active_indices.add(j)
        for i in active_boundary:
            active_indices.add(i)
        active_indices = sorted(active_indices)
        
        if len(active_indices) < 2:
            break
        
        idx_map = {idx: k for k, idx in enumerate(active_indices)}
        m = len(active_indices)
        
        # Build Jacobian-like matrix for the active constraints
        # We want to find displacement d such that all active constraints are satisfied
        # Use a simple approach: for each active constraint, add a force pushing to satisfy it
        
        # Compute forces from active constraints
        forces = np.zeros((m, 2))
        
        for i,j in active_pairs:
            if i not in idx_map or j not in idx_map:
                continue
            ii = idx_map[i]
            jj = idx_map[j]
            d = best_pts[i] - best_pts[j]
            dist = np.hypot(d[0], d[1])
            if dist < 1e-12:
                continue
            # Force to increase distance: push i away from j, j away from i
            # Target: dist should be >= 2*r, so push by (2*r - dist)/2 each
            push = max(0, 2*r - dist) / 2.0
            unit = d / dist
            forces[ii] += unit * push
            forces[jj] -= unit * push
        
        for i in active_boundary:
            if i not in idx_map:
                continue
            ii = idx_map[i]
            norm = np.hypot(best_pts[i,0], best_pts[i,1])
            if norm < 1e-12:
                continue
            # Push inward: target |p_i| <= 1-r
            push = max(0, norm - (1-r))
            inward = -best_pts[i] / norm
            forces[ii] += inward * push
        
        # Apply forces with damping
        force_mags = np.hypot(forces[:,0], forces[:,1])
        max_force = np.max(force_mags) if len(force_mags) > 0 else 0
        if max_force < tol:
            break
        
        # Scale forces to avoid overshooting
        scale = min(1.0, 0.5 / max_force) if max_force > 0 else 1.0
        
        candidate = best_pts.copy()
        for k, idx in enumerate(active_indices):
            candidate[idx] += forces[k] * scale
        
        # Enforce boundary
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 1.0 - 1e-9
        if np.any(too_far):
            candidate[too_far] *= (1.0 - 1e-9) / norms_cand[too_far]
        
        cand_r = _compute_radius_fast(candidate)
        
        if cand_r > best_r + 1e-12:
            best_pts = candidate
            best_r = cand_r
        else:
            # If no improvement, try smaller step
            candidate = best_pts.copy()
            for k, idx in enumerate(active_indices):
                candidate[idx] += forces[k] * scale * 0.1
            norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
            too_far = norms_cand > 1.0 - 1e-9
            if np.any(too_far):
                candidate[too_far] *= (1.0 - 1e-9) / norms_cand[too_far]
            cand_r = _compute_radius_fast(candidate)
            if cand_r > best_r + 1e-12:
                best_pts = candidate
                best_r = cand_r
            else:
                break
    
    return best_pts, best_r


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


def _force_optimize(pts, max_iter=5000, deadline=None, initial_lr=0.005, min_lr=1e-6, k_neighbors=10):
    """
    Force-based local optimization using cKDTree for nearest neighbors.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    lr = initial_lr
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Build KDTree
        tree = cKDTree(best_pts)
        
        # Find k nearest neighbors for each point
        dists, indices = tree.query(best_pts, k=k_neighbors + 1)
        nn_dists = dists[:, 1:]
        nn_indices = indices[:, 1:]
        
        # Compute forces
        nn_pts = best_pts[nn_indices]
        safe_dists = np.where(nn_dists > 1e-12, nn_dists, 1.0)
        unit_vecs = (best_pts[:, np.newaxis, :] - nn_pts) / safe_dists[:, :, np.newaxis]
        forces = np.sum(unit_vecs, axis=1)
        
        # Boundary force
        norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
        boundary_margin = 1.0 - norms
        active = boundary_margin < 0.1
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        inward_dir = -best_pts / safe_norms[:, np.newaxis]
        force_mag = np.where(active, (0.1 - boundary_margin) / 0.1 * 0.5, 0.0)
        boundary_forces = inward_dir * force_mag[:, np.newaxis]
        
        forces += boundary_forces
        
        # Normalize forces
        force_mags = np.hypot(forces[:, 0], forces[:, 1])
        force_mags[force_mags < 1e-12] = 1.0
        forces /= force_mags[:, np.newaxis]
        
        # Apply step
        candidate = best_pts + forces * lr
        
        # Enforce boundary constraint
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 0.999
        if np.any(too_far):
            candidate[too_far] *= 0.999 / norms_cand[too_far]
        
        cand_r = _compute_radius_fast(candidate)
        
        if cand_r > best_r + 1e-14:
            best_pts = candidate
            best_r = cand_r
        else:
            lr *= 0.98
            if lr < min_lr:
                break
    
    return best_pts, best_r


def _soft_min_optimize(pts, deadline, eps_start=20.0, eps_end=200.0, 
                       lr_start=0.003, lr_end=1e-5, max_iter=3000):
    """
    Gradient ascent on soft-min surrogate using log-sum-exp.
    Maximizes soft-min of pairwise distances and boundary distances.
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
        
        # Compute pairwise distances
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        b_dists = 1.0 - radii
        
        # Soft-min of pairwise distances: -1/eps * log(sum(exp(-eps * dists)))
        # We want to maximize this, so we ascend.
        # Gradient of -1/eps * log(sum_j exp(-eps * d_ij)) wrt p_i:
        # = sum_j [ exp(-eps * d_ij) / sum_k exp(-eps * d_ik) ] * (p_i - p_j) / d_ij
        
        # Pairwise part
        neg_eps_dists = -eps * dists
        max_vals = np.max(neg_eps_dists, axis=1, keepdims=True)
        max_vals[max_vals == -np.inf] = 0
        exp_terms = np.exp(neg_eps_dists - max_vals)
        np.fill_diagonal(exp_terms, 0.0)
        
        sums = np.sum(exp_terms, axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        weights = exp_terms / sums
        
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        grad_pair = np.sum(weights[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Boundary part: maximize soft-min of (1 - |p_i|)
        # = -1/eps * log(sum_i exp(-eps * b_i)) ... but this is sum over i, not per-i
        # Actually, we want to maximize min_i b_i. 
        # Soft approximation: -1/eps * log(sum_i exp(-eps * b_i))
        # Gradient wrt p_i: 
        # d/dp_i [-1/eps * log(sum_j exp(-eps * b_j))]
        # = -1/eps * 1/(sum_j exp(-eps * b_j)) * (-eps * exp(-eps * b_i)) * db_i/dp_i
        # = exp(-eps * b_i) / sum_j exp(-eps * b_j) * db_i/dp_i
        # db_i/dp_i = -p_i / |p_i|
        
        neg_eps_b = -eps * b_dists
        max_b = np.max(neg_eps_b)
        exp_b = np.exp(neg_eps_b - max_b)
        sum_b = np.sum(exp_b)
        sum_b = max(sum_b, 1e-300)
        w_b = exp_b / sum_b
        
        safe_norms = np.where(radii > 1e-12, radii, 1.0)
        grad_bound = -w_b[:, np.newaxis] * best_pts / safe_norms[:, np.newaxis]
        
        # Total gradient (we want to ascend)
        total_grad = grad_pair + 0.5 * grad_bound
        
        # Normalize
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
    Final high-precision polish: very small steps to tighten the packing.
    Uses cKDTree-based force optimization with very small steps.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    # Use progressively smaller steps with force model
    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]:
        if deadline is not None and time.time() > deadline:
            break
        
        for it in range(300):
            if deadline is not None and time.time() > deadline:
                break
            
            # Build KDTree
            tree = cKDTree(best_pts)
            dists, indices = tree.query(best_pts, k=11) # 10 neighbors + self
            nn_dists = dists[:, 1:]
            nn_indices = indices[:, 1:]
            nn_pts = best_pts[nn_indices]
            
            safe_dists = np.where(nn_dists > 1e-12, nn_dists, 1.0)
            unit_vecs = (best_pts[:, np.newaxis, :] - nn_pts) / safe_dists[:, :, np.newaxis]
            forces = np.sum(unit_vecs, axis=1)
            
            # Boundary force
            norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
            margin = 1.0 - norms
            active = margin < 0.05
            safe_norms = np.where(norms > 1e-12, norms, 1.0)
            inward_dir = -best_pts / safe_norms[:, np.newaxis]
            force_mag = np.where(active, (0.05 - margin) / 0.05, 0.0)
            boundary_forces = inward_dir * force_mag[:, np.newaxis]
            
            forces += boundary_forces
            
            force_mags = np.hypot(forces[:, 0], forces[:, 1])
            force_mags[force_mags < 1e-12] = 1.0
            forces /= force_mags[:, np.newaxis]
            
            candidate = best_pts + forces * lr
            
            # Enforce boundary
            norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
            too_far = norms_cand > 0.9999
            if np.any(too_far):
                candidate[too_far] *= 0.9999 / norms_cand[too_far]
            
            cand_r = _compute_radius_fast(candidate)
            
            if cand_r > best_r + 1e-14:
                best_pts = candidate
                best_r = cand_r
    
    return best_pts, best_r


def _random_perturb(pts, magnitude=0.01):
    """Add small random perturbation to escape local minima."""
    rng = np.random.default_rng()
    perturbed = pts + rng.normal(0, magnitude, size=pts.shape)
    # Enforce boundary
    norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
    too_far = norms > 0.99
    if np.any(too_far):
        perturbed[too_far] *= 0.99 / norms[too_far]
    return perturbed


def construct():
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    
    # Search over hexagonal spacings
    best_pts = None
    best_r = 0.0
    
    # Try a range of spacings
    spacings = np.linspace(0.09, 0.13, 30)
    for spacing in spacings:
        if time.time() > deadline:
            break
        pts = _init_hex(spacing)
        if len(pts) < N:
            continue
        # Take N closest to center
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius_fast(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    
    if best_pts is None:
        # Fallback: use original heuristic
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
    
    # Phase 1: Force-based optimization (fast, coarse)
    if time.time() < deadline - 60.0:
        best_pts, best_r = _force_optimize(best_pts, max_iter=10000, deadline=deadline - 60.0, initial_lr=0.005)
    
    # Phase 2: Newton refinement on active constraints
    if time.time() < deadline - 50.0:
        best_pts, best_r = _newton_refine(best_pts, deadline=deadline - 50.0, max_iter=300)
    
    # Phase 3: Soft-min gradient optimization (smooth surrogate, finer)
    if time.time() < deadline - 25.0:
        best_pts, best_r = _soft_min_optimize(best_pts, deadline=deadline - 25.0, eps_start=20.0, eps_end=200.0, lr_start=0.003, lr_end=1e-5, max_iter=3000)
    
    # Phase 4: Try random perturbations + re-optimize to escape local minima
    if time.time() < deadline - 10.0:
        for attempt in range(8):
            if time.time() > deadline - 10.0:
                break
            perturbed = _random_perturb(best_pts, magnitude=0.005)
            pert_r = _compute_radius_fast(perturbed)
            if pert_r >= best_r * 0.99:
                # Optimize the perturbed version
                p_pts, p_r = _force_optimize(perturbed, max_iter=3000, deadline=deadline - 10.0, initial_lr=0.003)
                if p_r > best_r:
                    best_pts = p_pts
                    best_r = p_r
                    # Apply Newton refinement to the improved solution
                    if time.time() < deadline - 5.0:
                        p_pts, p_r = _newton_refine(p_pts, deadline=deadline - 5.0, max_iter=100)
                        if p_r > best_r:
                            best_pts = p_pts
                            best_r = p_r
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 1.0:
        best_pts, best_r = _polish(best_pts, deadline - 1.0)
    
    # Phase 6: Final Newton refinement
    if time.time() < deadline - 0.5:
        best_pts, best_r = _newton_refine(best_pts, deadline=deadline - 0.5, max_iter=100)
    
    return best_pts
# EVOLVE-BLOCK-END
