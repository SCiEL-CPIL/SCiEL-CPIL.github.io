"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0

    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))

    # Min pair distance via vectorized computation
    # For N=300, 300x300 matrix is 90k elements, which is fast.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary)

def _hex_lattice_init(spacing, angle_deg=0):
    """Generate hexagonal lattice points within the unit disk."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # Generate a large enough grid
    k = int(1.0 / spacing) + 5
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            # Hex lattice basis vectors
            # v1 = (1, 0), v2 = (0.5, sqrt(3)/2) scaled by spacing
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            # Rotate
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    
    # Keep points strictly inside with margin for circle radius
    # The circle radius is r. The center must be at distance <= 1 - r.
    # Here we don't know r yet, so we keep points with radius < 1.0
    # and let the optimizer handle the boundary constraint.
    # However, for initialization, it's better to be safe.
    # We will estimate r later. For now, just keep points inside unit disk.
    mask = radius <= 1.0
    points = points[mask]
    
    if len(points) < N:
        return None

    # Sort by distance from center to pick the most "central" N points
    # This usually gives a better packing as it fills the disk uniformly
    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]

def _get_active_constraints(points, d_min, tol=1e-4):
    """
    Identify active constraints (pairs and boundary) close to the minimum distance.
    Returns:
        pairs: list of (i, j, vec, dist)
        boundaries: list of (i, vec, dist)
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    
    active_pairs = []
    active_boundaries = []
    
    # Pairs
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]
    mask_pairs = pair_dists < (d_min + tol)
    active_idx = np.where(mask_pairs)[0]
    
    if len(active_idx) > 0:
        i_vals, j_vals = iu
        for k in active_idx:
            i, j = i_vals[k], j_vals[k]
            vec = points[i] - points[j] # Gradient of dist wrt point i is vec/dist, wrt j is -vec/dist
            dist = pair_dists[k]
            if dist > 1e-12:
                active_pairs.append((i, j, vec / dist, dist))
            
    # Boundaries
    mask_bound = boundary_dists < (d_min + tol)
    active_b_idx = np.where(mask_bound)[0]
    if len(active_b_idx) > 0:
        for i in active_b_idx:
            norm = np.linalg.norm(points[i])
            if norm > 1e-12:
                vec = points[i] / norm # Gradient of (1-norm) wrt point is -vec. We want to maximize distance, so move OUT? 
                # d(1-|x|)/dx = -x/|x|. 
                # We want to INCREASE distance to boundary. 
                # So we move in direction -grad(dist) = x/|x|.
                # Wait, let's be consistent.
                # Objective: Maximize min(dists).
                # Grad of dist_boundary = -x/|x|.
                # To maximize, move in direction +grad = -x/|x|? No.
                # If I am at (0.9, 0), dist is 0.1. Grad is (-0.9, 0)/0.9 = (-1, 0).
                # Moving in direction of grad (left) increases dist to boundary?
                # (0.9, 0) + eps*(-1, 0) = (0.9-eps, 0). Dist becomes 1 - (0.9-eps) = 0.1+eps. Yes.
                # So direction is -x/|x|.
                active_boundaries.append((i, -points[i] / norm, boundary_dists[i]))
                
    return active_pairs, active_boundaries

def _active_gradient_ascent(points, max_iter=2000, deadline=None):
    """
    Optimize using gradient ascent on the active constraints (soft min of active set).
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    current_lr = 0.01
    patience = 0
    max_patience = 100
    
    start_time = time.time()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute current score and active set
        score = _compute_score(best_points)
        if score > best_score:
            best_score = score
            # best_points = best_points.copy() # Already updated below if we accept
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
            
        # Get d_min
        # Re-compute d_min for active set identification
        dists_to_boundary = 1.0 - np.linalg.norm(best_points, axis=1)
        min_boundary = float(np.min(dists_to_boundary))
        
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = float(np.min(dist_matrix))
        
        d_min = min(min_pair, min_boundary)
        
        if d_min <= 0:
            # Overlap, need to separate
            # Use a larger step or random kick? 
            # For now, just continue, gradient should push them apart.
            pass

        active_pairs, active_boundaries = _get_active_constraints(best_points, d_min, tol=1e-3)
        
        if not active_pairs and not active_boundaries:
            # No active constraints? Should not happen if d_min is min.
            break
            
        grad = np.zeros_like(best_points)
        
        # Compute soft min gradient over active set
        # We want to maximize the minimum of the active distances.
        # Approximation: Average gradient of active constraints, weighted by how close they are to min.
        
        # Distances in active set
        all_dists = [p[3] for p in active_pairs] + [b[2] for b in active_boundaries]
        if not all_dists:
            break
        min_d_active = min(all_dists)
        
        # Weights: exp(-lambda * (d_k - min_d_active))
        # If lambda is large, only the very minimum matters.
        # If lambda is small, it's an average.
        # Let's use a moderate lambda to smooth the gradient.
        lambda_t = 50.0
        
        total_weight = 0.0
        weighted_grad = np.zeros_like(best_points)
        
        for i, j, vec, dist in active_pairs:
            w = np.exp(-lambda_t * (dist - min_d_active))
            total_weight += w
            # Gradient of dist_ij wrt p_i is vec. wrt p_j is -vec.
            # We want to INCREASE dist.
            # So move p_i in direction vec, p_j in direction -vec.
            weighted_grad[i] += w * vec
            weighted_grad[j] += w * (-vec)
            
        for i, vec, dist in active_boundaries:
            w = np.exp(-lambda_t * (dist - min_d_active))
            total_weight += w
            # vec is already the direction to move (outward from center? No, vec was -x/|x|)
            # Wait, in _get_active_constraints, vec for boundary was -x/|x|.
            # Moving in direction -x/|x| increases distance to boundary.
            weighted_grad[i] += w * vec
            
        if total_weight > 0:
            grad = weighted_grad / total_weight
            
        # Normalize gradient to control step size relative to geometry?
        # The magnitude of grad depends on the number of active constraints.
        # Dividing by total_weight makes it an average, which is good.
        
        # Update
        new_points = best_points + current_lr * grad
        
        # Project back into unit circle
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / norms[mask, np.newaxis]
            
        # Check if we improved
        new_score = _compute_score(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            patience = 0
            # Increase LR slightly if making good progress
            current_lr = min(current_lr * 1.05, 0.05)
        else:
            # Decrease LR
            current_lr *= 0.95
            
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Generate multiple hex lattice inits with different rotations and spacings
    inits = []
    
    # Base spacing estimate: area per circle ~ pi / N. 
    # Hex lattice density: area per point = s^2 * sqrt(3)/2.
    # s^2 * sqrt(3)/2 ~ pi / N => s ~ sqrt(2*pi / (N*sqrt(3)))
    base_spacing = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Try a range of spacings around the estimate
    for factor in [0.95, 0.98, 1.0, 1.02, 1.05]:
        spacing = base_spacing * factor
        for angle_deg in [0, 10, 20, 30, 45, 60]:
            pts = _hex_lattice_init(spacing, angle_deg)
            if pts is not None:
                score = _compute_score(pts)
                inits.append((score, pts))
                
    # Sort by score and pick the top few
    inits.sort(key=lambda x: -x[0])
    top_inits = inits[:5]
    
    # Optimize the best initializations
    for score, pts in top_inits:
        if time.time() > deadline - 5:
            break
            
        optimized = _active_gradient_ascent(pts, max_iter=1000, deadline=deadline - 2)
        opt_score = _compute_score(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()
            
    # If we have time, try to polish the best result with a very fine step
    if best_points is not None and time.time() < deadline - 10:
        # Re-run optimization with smaller LR for polishing
        polished = _active_gradient_ascent(best_points, max_iter=500, deadline=deadline - 2)
        # Modify _active_gradient_ascent to accept initial LR? 
        # For now, just use the result. The previous run likely converged.
        # Let's just check if the polished version is better.
        pol_score = _compute_score(polished)
        if pol_score > best_score:
            best_score = pol_score
            best_points = polished.copy()

    # Fallback
    if best_points is None:
        spacing = base_spacing
        pts = _hex_lattice_init(spacing, 0)
        if pts is not None:
            best_points = pts
        else:
            # Spiral fallback
            angles = np.linspace(0, 2 * np.pi * N, N)
            radii = np.linspace(0, 0.95, N)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
    # Final validation
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.9999
        
    return best_points
# EVOLVE-BLOCK-END
