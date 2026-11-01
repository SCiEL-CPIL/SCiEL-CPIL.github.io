"""Structural change: direct nearest-neighbor repulsion optimization with multi-restart."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _hex_init(N, jitter=0.0):
    """Generate hexagonal lattice points, select N closest to origin."""
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    s *= 0.95
    
    best_points = None
    for attempt in range(30):
        k = int(1.0 / s) + 4
        pts = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3.0) / 2.0
            if abs(y) > 1.0:
                continue
            x_offset = (j % 2) * s / 2.0
            for i in range(-k, k + 1):
                x = (i + x_offset) * s
                r = np.hypot(x, y)
                if r <= 1.0 - s * 0.6:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            if best_points is None:
                best_points = pts[idx]
        s *= 0.97
    if best_points is None:
        best_points = np.random.randn(N, 2) * 0.5
    
    if jitter > 0:
        best_points = best_points + np.random.randn(N, 2) * jitter
    
    return best_points

def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    n = len(pts)
    if n < 2:
        min_pair = 1.0
    else:
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=1.0)
        if len(pairs) == 0:
            min_pair = 1.0
        else:
            idx1 = np.array([p[0] for p in pairs])
            idx2 = np.array([p[1] for p in pairs])
            dists = np.linalg.norm(pts[idx1] - pts[idx2], axis=1)
            min_pair = np.min(dists) / 2.0
    
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists_boundary)
    
    return min(min_pair, min_boundary)

def _direct_optimize(pts, max_iter=5000, step_size=0.01, deadline=None):
    """
    Direct nearest-neighbor repulsion optimization.
    Each point is pushed away from its nearest neighbor and toward the center.
    """
    n = len(pts)
    pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Build KD tree
        tree = cKDTree(pts)
        
        # For each point, find nearest neighbor
        dists, idxs = tree.query(pts, k=2)
        # dists[:, 0] is 0 (self), dists[:, 1] is nearest neighbor distance
        # idxs[:, 0] is self, idxs[:, 1] is nearest neighbor index
        
        nn_dist = dists[:, 1].copy()
        nn_idx = idxs[:, 1].copy()
        
        # Boundary distance
        r = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dist = 1.0 - r
        
        # Determine which constraint is limiting for each point
        # A point is limited by its nearest neighbor if nn_dist/2 < boundary_dist
        # Otherwise limited by boundary
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        for i in range(n):
            # Nearest neighbor repulsion
            j = nn_idx[i]
            d = nn_dist[i]
            if d > 1e-12:
                diff = pts[i] - pts[j]
                diff = diff / d
                # Push away from nearest neighbor
                forces[i] += diff * 0.5
            
            # Boundary attraction (push toward center if too close to boundary)
            if r[i] > 1e-12:
                inward = -pts[i] / r[i]
                # Strength proportional to how close to boundary
                forces[i] += inward * 0.3
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms > 1e-12, force_norms, 1.0)
        forces = forces / force_norms
        
        # Adaptive step size
        # Use smaller step if things are getting worse
        new_pts = pts + forces * step_size
        
        # Project inside unit circle
        new_r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = new_r > 0.999
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_r[mask] * 0.999
        
        # Check if improvement
        old_r = _compute_radius(pts)
        new_r_val = _compute_radius(new_pts)
        
        if new_r_val >= old_r - 1e-10:
            pts = new_pts
            # Gradually reduce step size
            step_size *= 0.999
        else:
            # Try smaller step
            step_size *= 0.95
            if step_size < 1e-6:
                break
    
    return pts, _compute_radius(pts)

def _gradient_descent(pts, max_iter=2000, lr=0.005, deadline=None):
    """
    Gradient descent with soft-min objective.
    """
    n = len(pts)
    pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Build KD tree for nearest neighbors
        tree = cKDTree(pts)
        # Query for points within radius 1.0
        pairs = tree.query_pairs(r=1.0)
        
        # Build gradient
        grad = np.zeros_like(pts)
        
        # For each pair, compute repulsion
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d > 1e-12:
                diff = (pts[i] - pts[j]) / d
                # Weight by inverse distance (stronger repulsion when closer)
                w = 1.0 / (d + 0.1)
                grad[i] += diff * w
                grad[j] -= diff * w
        
        # Boundary gradient
        r = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(n):
            if r[i] > 1e-12:
                inward = -pts[i] / r[i]
                # Stronger when close to boundary
                w = 1.0 / (1.0 - r[i] + 0.1)
                grad[i] += inward * w
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Update
        new_pts = pts - grad * lr
        
        # Project inside unit circle
        new_r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = new_r > 0.999
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_r[mask] * 0.999
        
        # Check improvement
        old_r = _compute_radius(pts)
        new_r_val = _compute_radius(new_pts)
        
        if new_r_val >= old_r - 1e-10:
            pts = new_pts
            lr *= 0.999
        else:
            lr *= 0.95
            if lr < 1e-8:
                break
    
    return pts, _compute_radius(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try multiple strategies
    best_pts = None
    best_r = -1.0
    
    # Strategy 1: Hex init + direct optimization
    pts1 = _hex_init(N, jitter=0.0)
    pts1, r1 = _direct_optimize(pts1, max_iter=3000, step_size=0.01, deadline=deadline)
    if r1 > best_r:
        best_r = r1
        best_pts = pts1.copy()
    
    # Strategy 2: Hex init + gradient descent
    if time.time() < deadline - 30.0:
        pts2 = _hex_init(N, jitter=0.0)
        pts2, r2 = _gradient_descent(pts2, max_iter=2000, lr=0.005, deadline=deadline)
        if r2 > best_r:
            best_r = r2
            best_pts = pts2.copy()
    
    # Strategy 3: Random init + direct optimization
    if time.time() < deadline - 30.0:
        pts3 = np.random.randn(N, 2) * 0.8
        # Project inside
        r3_init = np.hypot(pts3[:, 0], pts3[:, 1])
        mask = r3_init > 0.95
        pts3[mask] = pts3[mask] / r3_init[mask] * 0.95
        pts3, r3 = _direct_optimize(pts3, max_iter=2000, step_size=0.01, deadline=deadline)
        if r3 > best_r:
            best_r = r3
            best_pts = pts3.copy()
    
    # Strategy 4: Perturbed hex + direct optimization
    if time.time() < deadline - 20.0:
        pts4 = _hex_init(N, jitter=0.01)
        pts4, r4 = _direct_optimize(pts4, max_iter=2000, step_size=0.005, deadline=deadline)
        if r4 > best_r:
            best_r = r4
            best_pts = pts4.copy()
    
    # Final polish: small perturbations around best
    if best_pts is not None and time.time() < deadline - 5.0:
        for _ in range(20):
            if time.time() > deadline - 3.0:
                break
            perturbation = np.random.randn(N, 2) * 0.002
            candidate = best_pts + perturbation
            dists = np.hypot(candidate[:, 0], candidate[:, 1])
            mask = dists > 0.999
            if np.any(mask):
                candidate[mask] = candidate[mask] / dists[mask] * 0.999
            cand_r = _compute_radius(candidate)
            if cand_r > best_r:
                best_r = cand_r
                best_pts = candidate.copy()
    
    if best_pts is None:
        best_pts = _hex_init(N)
    
    # Ensure validity
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = dists > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / dists[mask] * 0.999
    
    return best_pts
# EVOLVE-BLOCK-END
