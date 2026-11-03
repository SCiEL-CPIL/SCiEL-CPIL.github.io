"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _force_optimize(pts, max_iter=2000, deadline=None, initial_lr=0.005, min_lr=1e-6):
    """
    Force-based local optimization: compute repulsion from nearest neighbors
    and boundary pressure, then move points to increase the minimum gap.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    lr = initial_lr
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find nearest neighbor for each point
        min_dists = np.min(dists, axis=1)
        min_indices = np.argmin(dists, axis=1)
        
        # Compute repulsion forces: push away from nearest neighbor
        forces = np.zeros_like(best_pts)
        for i in range(n):
            j = min_indices[i]
            d = min_dists[i]
            if d > 1e-12:
                direction = (best_pts[i] - best_pts[j]) / d
                # Weight by how close to the minimum gap
                forces[i] += direction * 1.0
        
        # Boundary force: pull inward if too close to edge
        norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
        boundary_margin = 1.0 - norms
        # Points closer to boundary get stronger inward pull
        for i in range(n):
            if boundary_margin[i] < 0.1:
                if norms[i] > 1e-12:
                    forces[i] -= best_pts[i] / norms[i] * (0.1 - boundary_margin[i]) / 0.1 * 0.5
        
        # Normalize forces
        force_mags = np.hypot(forces[:, 0], forces[:, 1])
        force_mags[force_mags < 1e-12] = 1.0
        forces /= force_mags[:, np.newaxis]
        
        # Apply step
        candidate = best_pts + forces * lr
        
        # Enforce boundary constraint: project back inside if needed
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 0.999
        if np.any(too_far):
            candidate[too_far] *= 0.999 / norms_cand[too_far]
        
        cand_r = _compute_radius_fast(candidate)
        
        if cand_r > best_r + 1e-12:
            best_pts = candidate
            best_r = cand_r
        else:
            lr *= 0.95
            if lr < min_lr:
                break
    
    return best_pts, best_r


def _polish(pts, deadline):
    """
    Final high-precision polish: very small steps to tighten the packing.
    """
    best_pts = pts.copy()
    best_r = _compute_radius_fast(best_pts)
    n = len(best_pts)
    
    # Use progressively smaller steps
    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
        if deadline is not None and time.time() > deadline:
            break
        
        for it in range(500):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute nearest neighbor forces
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            min_dists = np.min(dists, axis=1)
            min_indices = np.argmin(dists, axis=1)
            
            forces = np.zeros_like(best_pts)
            for i in range(n):
                j = min_indices[i]
                d = min_dists[i]
                if d > 1e-12:
                    direction = (best_pts[i] - best_pts[j]) / d
                    forces[i] += direction
            
            # Boundary force
            norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
            boundary_margin = 1.0 - norms
            for i in range(n):
                if boundary_margin[i] < 0.05:
                    if norms[i] > 1e-12:
                        forces[i] -= best_pts[i] / norms[i] * (0.05 - boundary_margin[i]) / 0.05 * 0.5
            
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
            
            if cand_r > best_r + 1e-13:
                best_pts = candidate
                best_r = cand_r
    
    return best_pts, best_r


def construct():
    deadline = time.time() + 140.0  # Safety margin (150s limit)
    
    # Search over hexagonal spacings
    best_pts = None
    best_r = 0.0
    
    # Try a range of spacings
    spacings = np.linspace(0.09, 0.13, 40)
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
    
    # Force-based local optimization
    if time.time() < deadline - 30.0:
        best_pts, best_r = _force_optimize(best_pts, max_iter=3000, deadline=deadline - 30.0, initial_lr=0.005)
    
    # Final high-precision polish
    if time.time() < deadline - 2.0:
        best_pts, best_r = _polish(best_pts, deadline - 2.0)
    
    return best_pts
# EVOLVE-BLOCK-END
