"""Improved circle packing using hexagonal lattice with local refinement."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_lattice(spacing, n_half=20):
    """Generate hexagonal lattice points within a square of side 2*n_half*spacing."""
    points = []
    for j in range(-n_half, n_half + 1):
        for i in range(-n_half, n_half + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append((x, y))
    return np.array(points)


def _min_distance(points):
    """Compute minimum pairwise distance and minimum distance to boundary."""
    n = len(points)
    if n < 2:
        return np.inf, np.inf
    # Minimum pairwise distance
    from scipy.spatial.distance import pdist
    d = pdist(points, 'euclidean')
    min_pair = d.min() if len(d) > 0 else np.inf
    # Minimum distance to boundary (1 - |c_i|)
    dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = dists.min()
    return min_pair, min_boundary


def _score(points):
    """Compute the packing radius: min(min_pair/2, min_boundary)."""
    min_pair, min_boundary = _min_distance(points)
    return min(min_pair / 2.0, min_boundary)


def _soft_score(points, beta=50.0):
    """Soft version of the score for gradient-based optimization."""
    n = len(points)
    if n < 2:
        return 1.0
    # Compute all pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points, 'euclidean')
    # Soft min of pairwise distances / 2
    if len(d) > 0:
        log_sum = np.logaddexp.reduce(np.log(d / 2.0))  # This is log(sum(exp(log(d/2))))
        # Actually, let's use a proper soft-min
        min_pair_soft = -np.logaddexp.reduce(-d / 2.0) / beta * beta  # Hmm, let me reconsider
        # Better: soft_min(x) = -log(sum(exp(-x_i)))/beta for large beta
        min_pair_soft = -np.log(np.sum(np.exp(-d / beta))) / beta
    else:
        min_pair_soft = np.inf
    # Boundary
    dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary_soft = -np.log(np.sum(np.exp(-dists / beta))) / beta
    return min(min_pair_soft, min_boundary_soft)


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin
    
    # Start with hexagonal lattice
    # Estimate spacing: area per point ~ pi * r^2, total area = pi * 1^2
    # r ~ 0.053, so area per point ~ 0.0088, spacing ~ sqrt(0.0088/0.866) ~ 0.101
    # Let's try spacing around 0.1
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initial spacings
    for base_spacing in [0.095, 0.10, 0.105, 0.11]:
        if time.time() > deadline:
            break
        
        # Generate lattice
        pts = _hex_lattice(base_spacing, n_half=25)
        # Keep points inside unit circle with some margin
        radii = np.linalg.norm(pts, axis=1)
        pts = pts[radii < 1.0]
        
        if len(pts) < N:
            continue
        
        # Select N points - prefer those closer to center for better packing
        order = np.argsort(radii[radii < 1.0])
        pts = pts[order[:N]]
        
        # Compute initial score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # If we have a good initial solution, try to refine
    if best_points is not None and time.time() < deadline - 30.0:
        # Local refinement using gradient ascent
        # We want to maximize min(min_pair/2, min_boundary)
        # Use a simple iterative approach: move points to increase the minimum distance
        
        pts = best_points.copy()
        n = len(pts)
        
        # Simple local optimization: for each iteration, find the bottleneck
        # and try to move points to alleviate it
        for iteration in range(200):
            if time.time() > deadline - 10.0:
                break
            
            # Compute current minimum
            from scipy.spatial.distance import pdist
            d = pdist(pts, 'euclidean')
            min_pair = d.min()
            dists = 1.0 - np.linalg.norm(pts, axis=1)
            min_boundary = dists.min()
            
            current_score = min(min_pair / 2.0, min_boundary)
            
            # If boundary is the bottleneck, move the closest point inward
            if min_boundary <= min_pair / 2.0:
                idx = np.argmin(dists)
                # Move inward slightly
                direction = -pts[idx] / np.linalg.norm(pts[idx])
                pts[idx] += direction * 0.001
            else:
                # Pairwise distance is the bottleneck
                # Find the pair with minimum distance
                d_matrix = np.sqrt((pts[:, None, :] - pts[None, :, :])**2).sum(axis=2)
                np.fill_diagonal(d_matrix, np.inf)
                min_idx = np.unravel_index(d_matrix.argmin(), d_matrix.shape)
                i, j = min_idx
                # Move both points apart slightly
                direction = pts[i] - pts[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                    pts[i] += direction * 0.001
                    pts[j] -= direction * 0.001
            
            # Check if we improved
            new_score = _score(pts)
            if new_score < current_score - 1e-8:
                # Revert
                pts[idx if min_boundary <= min_pair / 2.0 else min_idx[0]] -= (direction if min_boundary <= min_pair / 2.0 else direction) * 0.001
                if min_boundary > min_pair / 2.0:
                    pts[min_idx[1]] += direction * 0.001
                break
        
        final_score = _score(pts)
        if final_score > best_score:
            best_score = final_score
            best_points = pts.copy()
    
    # Ensure all points are valid (inside unit circle, finite)
    if best_points is not None:
        # Clip to ensure inside
        radii = np.linalg.norm(best_points, axis=1)
        mask = radii <= 1.0
        if not np.all(mask):
            best_points = best_points[mask]
            # If we lost points, we need to add them back - but this shouldn't happen
            # For safety, if we have fewer than N, fall back to lattice
            if len(best_points) < N:
                # Regenerate lattice
                pts = _hex_lattice(0.10, n_half=25)
                radii = np.linalg.norm(pts, axis=1)
                pts = pts[radii < 1.0]
                order = np.argsort(radii)
                best_points = pts[order[:N]]
        
        # Final validation
        best_points = best_points[:N]
        if len(best_points) != N:
            # Pad if necessary (shouldn't happen)
            while len(best_points) < N:
                best_points = np.vstack([best_points, np.array([[0.0, 0.0]])])
        
        # Ensure finite
        if not np.all(np.isfinite(best_points)):
            # Fall back to simple lattice
            pts = _hex_lattice(0.10, n_half=25)
            radii = np.linalg.norm(pts, axis=1)
            pts = pts[radii < 1.0]
            order = np.argsort(radii)
            best_points = pts[order[:N]]
        
        return best_points
    
    # Fallback: simple hexagonal lattice
    pts = _hex_lattice(0.10, n_half=25)
    radii = np.linalg.norm(pts, axis=1)
    pts = pts[radii < 1.0]
    order = np.argsort(radii)
    return pts[order[:N]]

# EVOLVE-BLOCK-END
