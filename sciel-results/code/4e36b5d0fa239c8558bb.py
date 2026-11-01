"""Rewrite: circle packing via deterministic ring construction + local Nesterov-style repulsion refinement."""
import numpy as np
import time

N = 50


def _score(pts):
    """Return the feasible common radius (min pairwise distance/2, min distance to boundary)."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    dists = 1.0 - np.linalg.norm(pts, axis=1)
    r = np.min(dists)
    # Pairwise distances
    # Compute all pairwise distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r = min(r, np.min(d) / 2.0)
    return r


def _make_initial():
    """Create initial points using concentric rings."""
    # Ring 0: center
    pts = [np.array([0.0, 0.0])]
    # Rings 1..k: distribute equally on circle of radius r_k
    # We want ~50 points total.
    # Try rings with counts: 1, 6, 12, 18, 13 (total 50)
    ring_counts = [6, 12, 18, 13]
    # Choose ring radii: roughly sqrt spacing
    # For 50 points in unit disk, average area per point = pi/50 ~ 0.0628
    # So typical spacing ~ 0.25
    ring_radii = [0.25, 0.5, 0.75, 0.95]
    for r, cnt in zip(ring_radii, ring_counts):
        angles = 2 * np.pi * np.arange(cnt) / cnt
        for a in angles:
            pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
    pts = np.array(pts)
    # Trim or pad to exactly N
    if pts.shape[0] > N:
        pts = pts[:N]
    elif pts.shape[0] < N:
        # Add random points near boundary
        extra = N - pts.shape[0]
        extra_pts = np.random.randn(extra, 2) * 0.05
        extra_pts += 0.95 * np.ones((extra, 1)) * (extra_pts / (np.linalg.norm(extra_pts, axis=1, keepdims=True) + 1e-9))
        pts = np.vstack([pts, extra_pts])
    return pts


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    best_pts = _make_initial()
    best_r = _score(best_pts)
    
    # Local optimization: iterative repulsion with Nesterov momentum
    pts = best_pts.copy()
    velocity = np.zeros_like(pts)
    learning_rate = 0.01
    momentum = 0.9
    repulsion_strength = 1.0
    attraction_to_center = 0.1
    
    # We want to maximize r = min(min_pairwise_dist/2, min_dist_to_boundary)
    # This is a max-min problem. We can use subgradient-style ascent.
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        # Compute all pairwise distances and distances to boundary
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        # Find the critical constraint (smallest of all distances/2 and dists)
        min_pair_dist = np.min(d)
        min_boundary_dist = np.min(dists)
        
        # Gradient of the min function: we want to increase the smallest constraint
        # Subgradient approach: push apart the pair with min distance, and push inward the point closest to boundary
        
        grad = np.zeros_like(pts)
        
        # Pairwise repulsion: for the closest pair, push them apart
        min_idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = min_idx
        if i != j:
            diff = pts[i] - pts[j]
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-9:
                direction = diff / norm_diff
                # Increase the minimum distance: move i in direction, j in -direction
                grad[i] += direction * repulsion_strength
                grad[j] -= direction * repulsion_strength
        
        # Boundary attraction: for the point closest to boundary, push it inward
        min_boundary_idx = np.argmin(dists)
        pt = pts[min_boundary_idx]
        norm_pt = np.linalg.norm(pt)
        if norm_pt > 1e-9:
            direction = -pt / norm_pt
            grad[min_boundary_idx] += direction * repulsion_strength
        
        # Nesterov momentum
        velocity = momentum * velocity + learning_rate * grad
        pts_new = pts + velocity
        
        # Clip to unit disk (with small margin)
        norms = np.linalg.norm(pts_new, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / norms[mask] * 1.0
        
        # Check if improved
        new_r = _score(pts_new)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # Revert and decrease learning rate
            learning_rate *= 0.999
            pts = pts  # keep old points
        
        # Early stop if learning rate too small
        if learning_rate < 1e-6:
            break
    
    # Final polish: try to further refine using smaller steps
    learning_rate = 0.001
    velocity = np.zeros_like(pts)
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        grad = np.zeros_like(pts)
        min_idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = min_idx
        if i != j:
            diff = pts[i] - pts[j]
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-9:
                direction = diff / norm_diff
                grad[i] += direction
                grad[j] -= direction
        
        min_boundary_idx = np.argmin(dists)
        pt = pts[min_boundary_idx]
        norm_pt = np.linalg.norm(pt)
        if norm_pt > 1e-9:
            direction = -pt / norm_pt
            grad[min_boundary_idx] += direction
        
        velocity = momentum * velocity + learning_rate * grad
        pts_new = pts + velocity
        norms = np.linalg.norm(pts_new, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / norms[mask] * 1.0
        
        new_r = _score(pts_new)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            learning_rate *= 0.999
        
        if learning_rate < 1e-7:
            break
    
    return best_pts
