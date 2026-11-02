"""Packing 200 equal circles in a unit square [-1/2, 1/2]^2.
Strategy:
1. Initialize with hexagonal lattice.
2. Use stochastic subgradient method with exact gradients for min-distance.
3. Track active constraint set (pairs and boundaries near the minimum).
4. Refine with local search on the active set.
5. Final polish with tight tolerances.
"""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import time


def _compute_radius(pts):
    """Compute the largest common radius for given centres."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    
    # Boundary distances
    dists_boundary = 0.5 - np.abs(pts)
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    dists = pdist(pts, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    
    r = min(min_pair_dist / 2.0, min_boundary)
    return r


def _hex_init():
    """Generate hexagonal lattice initialization."""
    r_est = 0.0366
    dx = 2 * r_est
    dy = r_est * np.sqrt(3)
    
    points = []
    j = 0
    while True:
        y = -0.5 + dy/2 + j * dy
        if y > 0.5:
            break
        offset = (j % 2) * dx / 2
        i = 0
        while True:
            x = -0.5 + dx/2 + offset + i * dx
            if x > 0.5:
                break
            points.append((x, y))
            i += 1
        j += 1
    
    points = np.array(points)
    
    if len(points) > N:
        # Select N points that maximize minimum distance
        # Use greedy farthest-point sampling
        selected = []
        # Start with a random point
        remaining = set(range(len(points)))
        selected_idx = np.random.choice(len(points))
        selected.append(selected_idx)
        remaining.remove(selected_idx)
        
        while len(selected) < N and remaining:
            # Find point in remaining that maximizes min distance to selected
            best_idx = None
            best_min_dist = -1
            for idx in remaining:
                d = np.min(np.linalg.norm(points[idx] - points[selected], axis=1))
                if d > best_min_dist:
                    best_min_dist = d
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        points = points[selected]
    elif len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, (N - len(points), 2))
        points = np.vstack([points, extra])
    
    points = np.clip(points, -0.49, 0.49)
    return points


def _stochastic_subgradient(pts, time_budget, margin=0.004):
    """
    Stochastic subgradient method with active constraint tracking.
    Maximizes r = min(min_pair_dist/2, min_boundary).
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Track active constraints
    # Active pairs: those with distance close to min
    # Active boundaries: those with distance close to min
    
    step_size = 0.005
    n_iter = 0
    max_iter = 100000
    patience = 5000
    no_improve_count = 0
    
    while n_iter < max_iter:
        if time.time() > time_budget:
            break
        
        # Compute current radius and identify active constraints
        r = best_r
        
        # Find active pairs (distance <= 2*r + margin)
        tree = cKDTree(best_pts)
        active_pairs = list(tree.query_pairs(2 * r + margin))
        
        # Find active boundaries (distance to boundary <= r + margin)
        active_boundaries = []
        for i in range(N):
            for wall in range(4):
                if wall == 0:  # left
                    d = 0.5 + best_pts[i, 0]
                elif wall == 1:  # right
                    d = 0.5 - best_pts[i, 0]
                elif wall == 2:  # bottom
                    d = 0.5 + best_pts[i, 1]
                else:  # top
                    d = 0.5 - best_pts[i, 1]
                if d <= r + margin:
                    active_boundaries.append((i, wall))
        
        # Compute subgradient: sum of unit vectors from active constraints
        grad = np.zeros_like(best_pts)
        
        # Pair contributions
        for i, j in active_pairs:
            diff = best_pts[i] - best_pts[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                unit = diff / dist
                grad[i] += unit
                grad[j] -= unit
        
        # Boundary contributions
        for i, wall in active_boundaries:
            if wall == 0:  # left: move right
                grad[i, 0] += 1
            elif wall == 1:  # right: move left
                grad[i, 0] -= 1
            elif wall == 2:  # bottom: move up
                grad[i, 1] += 1
            else:  # top: move down
                grad[i, 1] -= 1
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Stochastic update: pick a random point and move it
        idx = np.random.randint(0, N)
        new_pts = best_pts.copy()
        new_pts[idx] += step_size * grad[idx]
        new_pts = np.clip(new_pts, -0.49, 0.49)
        
        new_r = _compute_radius(new_pts)
        
        if new_r >= best_r:
            best_pts = new_pts
            best_r = new_r
            no_improve_count = 0
            # Occasionally reduce step size
            if n_iter % 5000 == 0:
                step_size *= 0.98
        else:
            no_improve_count += 1
            # Occasionally accept worse solutions
            if np.random.random() < np.exp(-10 * (best_r - new_r) / max(step_size, 1e-10)):
                best_pts = new_pts
                best_r = new_r
        
        # Adapt step size based on stagnation
        if no_improve_count > patience:
            step_size *= 0.9
            no_improve_count = 0
            if step_size < 1e-6:
                break
        
        n_iter += 1
    
    return best_pts, best_r


def _local_refine(pts, time_budget):
    """
    Local refinement: identify the constraining pair/boundary and move points apart.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > time_budget:
            break
        
        # Find the minimum distance pair
        tree = cKDTree(best_pts)
        pairs = list(tree.query_pairs(2 * best_r + 0.001))
        
        min_dist = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
                min_pair = (i, j)
        
        # Find the minimum boundary distance
        min_boundary = np.inf
        min_boundary_point = None
        min_boundary_wall = None
        for i in range(N):
            for wall in range(4):
                if wall == 0:
                    d = 0.5 + best_pts[i, 0]
                elif wall == 1:
                    d = 0.5 - best_pts[i, 0]
                elif wall == 2:
                    d = 0.5 + best_pts[i, 1]
                else:
                    d = 0.5 - best_pts[i, 1]
                if d < min_boundary:
                    min_boundary = d
                    min_boundary_point = i
                    min_boundary_wall = wall
        
        # Determine which constraint is active
        if min_dist / 2.0 < min_boundary:
            # Pair is constraining
            if min_pair is None:
                break
            i, j = min_pair
            diff = best_pts[i] - best_pts[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                break
            direction = diff / norm
            
            # Try moving apart
            improved = False
            for step in [0.001, 0.0005, 0.0001, 0.00005]:
                new_pts = best_pts.copy()
                new_pts[i] += step * direction
                new_pts[j] -= step * direction
                new_pts = np.clip(new_pts, -0.49, 0.49)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
                    improved = True
                    break
            
            if not improved:
                break
        else:
            # Boundary is constraining
            if min_boundary_point is None:
                break
            i = min_boundary_point
            wall = min_boundary_wall
            
            # Move away from boundary
            improved = False
            for step in [0.001, 0.0005, 0.0001, 0.00005]:
                new_pts = best_pts.copy()
                if wall == 0:
                    new_pts[i, 0] += step
                elif wall == 1:
                    new_pts[i, 0] -= step
                elif wall == 2:
                    new_pts[i, 1] += step
                else:
                    new_pts[i, 1] -= step
                new_pts = np.clip(new_pts, -0.49, 0.49)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
                    improved = True
                    break
            
            if not improved:
                break
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate initial points with hexagonal lattice
    pts = _hex_init()
    
    # Phase 1: Stochastic subgradient optimization
    pts, r = _stochastic_subgradient(pts, deadline)
    
    # Phase 2: Local refinement
    if time.time() < deadline - 5.0:
        pts, r = _local_refine(pts, deadline)
    
    # Phase 3: Final polish with very small steps
    if time.time() < deadline - 2.0:
        step_size = 0.0001
        for _ in range(1000):
            if time.time() > deadline:
                break
            idx = np.random.randint(0, N)
            delta = np.random.normal(0, step_size, 2)
            new_pts = pts.copy()
            new_pts[idx] += delta
            new_pts = np.clip(new_pts, -0.49, 0.49)
            new_r = _compute_radius(new_pts)
            if new_r >= r:
                pts = new_pts
                r = new_r
    
    # Final validity check
    pts = np.clip(pts, -0.49, 0.49)
    
    # Verify no coincident points
    min_d = pdist(pts).min()
    if min_d < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                    pts[j] += np.random.normal(0, 1e-5, 2)
                    pts[j] = np.clip(pts[j], -0.49, 0.49)
    
    return pts
# EVOLVE-BLOCK-END
