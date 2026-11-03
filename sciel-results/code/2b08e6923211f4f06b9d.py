"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

# Global deadline guard
_DEADLINE = time.time() + 115.0  # safety margin before 120s kill


def _score(points):
    """Compute the packing radius for a set of centers."""
    # Boundary constraint
    dist_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    # Pairwise distances
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, np.min(dist_to_boundary))


def _smooth_min(x, beta=50.0):
    """Smooth minimum using log-sum-exp."""
    x_min = np.min(x)
    return x_min - np.log(np.sum(np.exp(-beta * (x - x_min)))) / beta


def _energy(points, r, beta=50.0):
    """
    Smooth surrogate for the packing problem.
    We want to maximize r subject to:
      |c_i - c_j| >= 2r for all i < j
      |c_i| <= 1 - r for all i
    
    Energy = sum of soft violations (should be 0 at optimum)
    """
    n = len(points)
    # Pairwise constraints: |c_i - c_j| - 2r >= 0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, 0.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    pair_violations = 2.0 * r - dists[iu]
    
    # Boundary constraints: 1 - r - |c_i| >= 0
    boundary_violations = (1.0 - r) - np.linalg.norm(points, axis=1)
    
    # Soft-min style: penalize violations with a smooth function
    # Use a smooth ReLU-like penalty
    def smooth_relu(x):
        return 0.5 * (x + np.sqrt(x ** 2 + 1e-6))
    
    energy = np.sum(smooth_relu(pair_violations)) + np.sum(smooth_relu(boundary_violations))
    return energy


def _construct_ring_structure(n_points):
    """
    Initialize with a ring structure: center + rings of points.
    This gives a good starting point for the packing.
    """
    points = np.zeros((n_points, 2))
    
    # Place first point at center
    points[0] = [0.0, 0.0]
    
    # Fill remaining points in concentric rings
    remaining = n_points - 1
    ring_idx = 1
    while remaining > 0:
        # Approximate number of points that fit in this ring
        # Circle of radius r can hold ~2*pi*r / d points where d is spacing
        # For a ring at radius ~ring_idx * spacing, spacing ~ 0.13
        spacing = 0.13
        ring_radius = ring_idx * spacing
        circumference = 2 * np.pi * ring_radius
        max_in_ring = max(2, int(circumference / spacing))
        
        n_in_ring = min(remaining, max_in_ring)
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        r_offset = ring_idx * spacing * 0.85  # slightly less to avoid overlap
        
        for i, angle in enumerate(angles):
            idx = 1 + (ring_idx - 1) * max_in_ring + i
            if idx >= n_points:
                break
            points[idx, 0] = r_offset * np.cos(angle)
            points[idx, 1] = r_offset * np.sin(angle)
        
        remaining -= n_in_ring
        ring_idx += 1
    
    # Clip to unit circle
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.9
    points[mask] *= (0.9 / norms[mask])[:, np.newaxis]
    
    return points


def _local_refinement(points, r_init, max_iter=1000, tol=1e-12):
    """
    Local refinement using gradient-based optimization.
    Maximize r subject to pairwise and boundary constraints.
    """
    n = len(points)
    
    def objective_and_grad(x):
        """Objective: -r (minimize), with soft constraints via penalty."""
        r = x[-1]
        pts = x[:-1].reshape(-1, 2)
        
        # Penalty-based smooth objective
        beta = 100.0
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, 0.0)
        iu = np.triu_indices(n, k=1)
        pair_viol = 2.0 * r - dists[iu]
        
        norms = np.linalg.norm(pts, axis=1)
        bound_viol = (1.0 - r) - norms
        
        # Smooth penalty
        def srelu(v):
            return 0.5 * (v + np.sqrt(v ** 2 + 1e-10))
        
        penalty = np.sum(srelu(pair_viol)) + np.sum(srelu(bound_viol))
        
        # We want to maximize r, so objective = -r + penalty
        obj = -r + penalty
        
        # Gradient
        grad_r = -1.0
        # d(penalty)/dr from pair violations
        pair_grad_r = np.sum(np.where(pair_viol > 0, 
                                       0.5 * (1 + pair_viol / np.sqrt(pair_viol ** 2 + 1e-10)), 0.0))
        # d(penalty)/dr from boundary violations
        bound_grad_r = np.sum(np.where(bound_viol > 0,
                                        0.5 * (-1 + bound_viol / np.sqrt(bound_viol ** 2 + 1e-10)), 0.0))
        grad_r += pair_grad_r + bound_grad_r
        
        # Gradient w.r.t. points
        grad_pts = np.zeros_like(pts)
        
        # Pairwise gradient
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                viol = 2.0 * r - dist
                if viol > 0:
                    w = 0.5 * (1 + viol / np.sqrt(viol ** 2 + 1e-10))
                    grad_pts[i] -= w * d / dist
                    grad_pts[j] += w * d / dist
        
        # Boundary gradient
        for i in range(n):
            norm = np.linalg.norm(pts[i])
            if norm < 1e-10:
                continue
            viol = (1.0 - r) - norm
            if viol > 0:
                w = 0.5 * (-1 + viol / np.sqrt(viol ** 2 + 1e-10))
                grad_pts[i] += w * pts[i] / norm
        
        grad = np.concatenate([grad_pts.ravel(), [grad_r]])
        return obj, grad
    
    x0 = np.concatenate([points.ravel(), [r_init]])
    
    best_x = x0.copy()
    best_obj = objective_and_grad(x0)[0]
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE:
            break
        
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                          options={'maxiter': 50, 'gtol': tol, 'ftol': tol})
            if res.x is not None and res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
                x0 = res.x
            else:
                break
        except Exception:
            break
    
    # Extract best solution
    best_r = best_x[-1]
    best_pts = best_x[:-1].reshape(-1, 2)
    
    # Final clip to ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    max_allowed = 1.0 - best_r
    over = norms > max_allowed
    if np.any(over):
        best_pts[over] *= (max_allowed / norms[over])[:, np.newaxis]
    
    return best_pts, best_r


def _binary_search_polish(points, r_low, r_high, max_iter=50, tol=1e-10):
    """
    Binary search on r with feasibility check via local optimization.
    """
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if r_high - r_low < tol:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration at radius r_mid
        def feasibility_energy(x):
            pts = x.reshape(-1, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, 0.0)
            iu = np.triu_indices(N, k=1)
            pair_viol = 2.0 * r_mid - dists[iu]
            norms = np.linalg.norm(pts, axis=1)
            bound_viol = (1.0 - r_mid) - norms
            
            def srelu(v):
                return 0.5 * (v + np.sqrt(v ** 2 + 1e-10))
            
            return np.sum(srelu(pair_viol)) + np.sum(srelu(bound_viol))
        
        res = minimize(feasibility_energy, points.ravel(), method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.fun < 1e-6:
            r_low = r_mid
            points = res.x.reshape(-1, 2)
        else:
            r_high = r_mid
    
    return points, r_low


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    
    # Start with ring structure initialization
    points = _construct_ring_structure(N)
    
    # Initial radius estimate
    r_init = _score(points)
    
    # Local refinement
    points, r = _local_refinement(points, r_init, max_iter=500, tol=1e-12)
    
    # Binary search polish
    if time.time() < _DEADLINE:
        points, r = _binary_search_polish(points, r, r + 0.01, max_iter=30, tol=1e-10)
    
    # Final validity check and clip
    norms = np.linalg.norm(points, axis=1)
    max_allowed = 1.0 - r
    over = norms > max_allowed
    if np.any(over):
        points[over] *= (max_allowed / norms[over])[:, np.newaxis]
    
    # Ensure non-negative radius
    r = max(r, 0.0)
    
    return points
# EVOLVE-BLOCK-END

