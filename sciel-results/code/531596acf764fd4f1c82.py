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


def _logsumexp_neg_dists(pts, temperature=0.01):
    """
    Compute a smooth surrogate: -logsumexp(-dists/temperature) approximates min(dists).
    Returns the surrogate value and its gradient.
    """
    n = len(pts)
    if n < 2:
        return 0.0, np.zeros_like(pts)
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, np.inf)
    
    # -logsumexp(-dists/T) = -T * log(sum(exp(-dists/T)))
    neg_dists_over_T = -dists / temperature
    max_val = np.max(neg_dists_over_T, axis=1)
    max_val[max_val == -np.inf] = 0  # handle inf
    exp_terms = np.exp(neg_dists_over_T - max_val[:, np.newaxis])
    np.fill_diagonal(exp_terms, 0.0)
    
    sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
    # Avoid division by zero
    sum_exp[sum_exp == 0] = 1.0
    weights = exp_terms / sum_exp
    
    log_sums = np.log(np.sum(exp_terms, axis=1)) + max_val
    surrogate = -temperature * log_sums
    
    # Vectorized gradient calculation
    # grad[i] = sum_j weights[i,j] * (pts[i] - pts[j]) / dists[i,j]
    # (pts[i] - pts[j]) / dists[i,j] is the unit vector u_ij
    # Note: dists[i,i] is inf, weights[i,i] is 0, so 0 * inf is nan. We handle this.
    
    # Create unit vectors matrix
    # u[i,j] = (pts[i] - pts[j]) / dists[i,j]
    # To avoid division by zero/inf issues, we can compute:
    # u = diff / dists[:, :, np.newaxis]
    # But dists has inf on diagonal.
    # Let's use a mask or safe division.
    
    # diff is shape (N, N, 2)
    # dists is shape (N, N)
    # We want to sum over j for each i: sum_j weights[i,j] * u[i,j]
    
    # u[i,j] = diff[i,j] / dists[i,j]
    # We can compute this safely:
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # Multiply by weights (broadcast weights to (N, N, 2))
    weighted_vecs = unit_vecs * weights[:, :, np.newaxis]
    
    # Sum over j (axis 1)
    grad = np.sum(weighted_vecs, axis=1)
    
    return surrogate, grad


def _gradient_optimize(pts, deadline, initial_lr=0.002, temperature=0.01, max_iter=5000):
    """
    Gradient-based optimization using log-sum-exp surrogate for min pair distance
    and a boundary penalty.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    lr = initial_lr
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute smooth surrogate gradient for pair distances
        _, pair_grad = _logsumexp_neg_dists(best_pts, temperature)
        
        # Boundary gradient: penalize points too close to boundary
        norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
        # Vectorized boundary gradient
        margin = 1.0 - norms
        # Only apply force if margin < 0.15
        active = margin < 0.15
        # Force direction is inward: -pts / norms
        # Magnitude: (0.15 - margin) / 0.15
        # We use np.where to handle zero norms safely
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        inward_dir = -best_pts / safe_norms[:, np.newaxis]
        force_mag = np.where(active, (0.15 - margin) / 0.15, 0.0)
        boundary_grad = inward_dir * force_mag[:, np.newaxis]
        
        # Combined gradient
        total_grad = pair_grad + 0.3 * boundary_grad
        
        # Normalize gradient
        grad_norm = np.hypot(total_grad[:, 0], total_grad[:, 1])
        grad_norm[grad_norm < 1e-12] = 1.0
        total_grad /= grad_norm[:, np.newaxis]
        
        # Adaptive learning rate
        candidate = best_pts + total_grad * lr
        
        # Enforce boundary constraint
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 0.999
        if np.any(too_far):
            candidate[too_far] *= 0.999 / norms_cand[too_far]
        
        cand_r = _compute_radius_fast(candidate)
        
        if cand_r > best_r + 1e-14:
            best_pts = candidate
            best_r = cand_r
            temperature *= 1.001
        else:
            lr *= 0.97
            temperature = max(temperature * 0.99, 0.005)
            if lr < 1e-8:
                break
    
    return best_pts, best_r


def _force_optimize(pts, max_iter=5000, deadline=None, initial_lr=0.005, min_lr=1e-6, k_neighbors=10):
    """
    Force-based local optimization using cKDTree for nearest neighbors.
    This is much faster than O(N^2) full matrix calculations.
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
        # query returns (distances, indices)
        dists, indices = tree.query(best_pts, k=k_neighbors + 1)
        # dists[i, 0] is 0 (distance to self), indices[i, 0] is i
        # We want distances to neighbors i, 1...k
        nn_dists = dists[:, 1:]  # shape (N, k)
        nn_indices = indices[:, 1:]  # shape (N, k)
        
        # Compute forces
        # Get neighbor points
        nn_pts = best_pts[nn_indices]  # shape (N, k, 2)
        
        # Directions: (best_pts[:, np.newaxis, :] - nn_pts) / nn_dists[:, :, np.newaxis]
        safe_dists = np.where(nn_dists > 1e-12, nn_dists, 1.0)
        unit_vecs = (best_pts[:, np.newaxis, :] - nn_pts) / safe_dists[:, :, np.newaxis]
        
        # Forces from neighbors: sum of unit vectors (repel equally)
        forces = np.sum(unit_vecs, axis=1)
        
        # Boundary force: pull inward if too close to edge
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


def _polish(pts, deadline):
    """
    Final high-precision polish: very small steps to tighten the packing.
    Uses cKDTree-based force optimization with very small steps.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    # Use progressively smaller steps with force model
    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]:
        if deadline is not None and time.time() > deadline:
            break
        
        for it in range(200):
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
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
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
    
    # Phase 1: Force-based optimization (fast, coarse) - increased iterations due to speed
    if time.time() < deadline - 60.0:
        best_pts, best_r = _force_optimize(best_pts, max_iter=10000, deadline=deadline - 60.0, initial_lr=0.005)
    
    # Phase 2: Gradient-based optimization with log-sum-exp surrogate (finer) - kept as backup
    if time.time() < deadline - 30.0:
        best_pts, best_r = _gradient_optimize(best_pts, deadline=deadline - 30.0, initial_lr=0.003, temperature=0.01, max_iter=1000)
    
    # Phase 3: Try random perturbations + re-optimize to escape local minima
    if time.time() < deadline - 10.0:
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            perturbed = _random_perturb(best_pts, magnitude=0.005)
            pert_r = _compute_radius_fast(perturbed)
            if pert_r >= best_r * 0.99:
                # Optimize the perturbed version
                p_pts, p_r = _force_optimize(perturbed, max_iter=5000, deadline=deadline - 10.0, initial_lr=0.003)
                if p_r > best_r:
                    best_pts = p_pts
                    best_r = p_r
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 1.0:
        best_pts, best_r = _polish(best_pts, deadline - 1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
