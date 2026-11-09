import numpy as np
import time
import itertools
import sys

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position.
    Uses vectorized NumPy operations on pre-generated indices.
    """
    n = len(points)
    # Pre-generate indices if not cached
    if not hasattr(_count_convex7, 'indices'):
        # Generate all combinations of 7 indices from 0..n-1
        # This is done once
        _count_convex7.indices = np.array(list(itertools.combinations(range(n), 7)))
    
    indices = _count_convex7.indices
    # points has shape (n, 2)
    # indices has shape (M, 7)
    # pts will have shape (M, 7, 2)
    pts = points[indices]
    
    # Extract points
    p0 = pts[:, 0, :]
    p1 = pts[:, 1, :]
    p2 = pts[:, 2, :]
    p3 = pts[:, 3, :]
    p4 = pts[:, 4, :]
    p5 = pts[:, 5, :]
    p6 = pts[:, 6, :]
    
    # Compute cross products for turns
    # Turn p0->p1->p2
    c0 = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p1[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p1[:, 0])
    # Turn p1->p2->p3
    c1 = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p2[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p2[:, 0])
    # Turn p2->p3->p4
    c2 = (p3[:, 0] - p2[:, 0]) * (p4[:, 1] - p3[:, 1]) - (p3[:, 1] - p2[:, 1]) * (p4[:, 0] - p3[:, 0])
    # Turn p3->p4->p5
    c3 = (p4[:, 0] - p3[:, 0]) * (p5[:, 1] - p4[:, 1]) - (p4[:, 1] - p3[:, 1]) * (p5[:, 0] - p4[:, 0])
    # Turn p4->p5->p6
    c4 = (p5[:, 0] - p4[:, 0]) * (p6[:, 1] - p5[:, 1]) - (p5[:, 1] - p4[:, 1]) * (p6[:, 0] - p5[:, 0])
    # Turn p5->p6->p0
    c5 = (p6[:, 0] - p5[:, 0]) * (p0[:, 1] - p6[:, 1]) - (p6[:, 1] - p5[:, 1]) * (p0[:, 0] - p6[:, 0])
    # Turn p6->p0->p1
    c6 = (p0[:, 0] - p6[:, 0]) * (p1[:, 1] - p0[:, 1]) - (p0[:, 1] - p6[:, 1]) * (p1[:, 0] - p6[:, 0])
    
    # Check if all turns are convex (left turns)
    # Threshold 1e-9
    is_convex = (c0 > 1e-9) & (c1 > 1e-9) & (c2 > 1e-9) & (c3 > 1e-9) & (c4 > 1e-9) & (c5 > 1e-9) & (c6 > 1e-9)
    
    return np.sum(is_convex)

def _is_valid(points: np.ndarray) -> bool:
    """Check validity constraints."""
    n = len(points)
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check minimum distance
    # Vectorized distance check
    # dist_sq[i, j]
    # This is O(N^2) which is small
    dist_sq = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Ignore self-distance
    if np.any(dist_sq < 1e-12):
        return False
    # Check no three collinear
    # This is O(N^3). N=33, so 33^3/6 ~ 6000 checks. Fast.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                val = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                if abs(val) < 1e-9:
                    return False
    return True

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres problem: place 33 points in the unit square to minimize
    the number of convex 7-gons.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    np.random.seed(42)
    n = 33
    
    # Initial random configuration
    best_points = np.random.rand(n, 2)
    if not _is_valid(best_points):
        for _ in range(1000):
            best_points = np.random.rand(n, 2)
            if _is_valid(best_points):
                break
    
    best_count = _count_convex7(best_points)
    best_value = 1.0 / (1.0 + best_count)
    
    # Simulated Annealing
    current_points = best_points.copy()
    current_count = best_count
    current_value = best_value
    
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    
    # Pre-generate indices for _count_convex7
    _count_convex7.indices = np.array(list(itertools.combinations(range(n), 7)))
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(n)
        
        # Generate perturbation
        # Scale decreases with T
        scale = 0.05 * (T / 0.05)
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        
        old_x = current_points[idx, 0]
        old_y = current_points[idx, 1]
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Clamp to unit square [0, 1]
        new_x = np.clip(new_x, 0.001, 0.999)
        new_y = np.clip(new_y, 0.001, 0.999)
        
        current_points[idx, 0] = new_x
        current_points[idx, 1] = new_y
        
        # Check validity
        if not _is_valid(current_points):
            current_points[idx, 0] = old_x
            current_points[idx, 1] = old_y
            T *= cooling
            continue
        
        new_count = _count_convex7(current_points)
        new_value = 1.0 / (1.0 + new_count)
        
        delta = new_value - current_value
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_count = new_count
            current_value = new_value
            if new_value > best_value:
                best_value = new_value
                best_points = current_points.copy()
        else:
            current_points[idx, 0] = old_x
            current_points[idx, 1] = old_y
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Optional: Restart if stuck?
        # If we haven't improved in a while, maybe reheat?
        # For now, simple SA.
        
    return best_points

# EVOLVE-BLOCK-END
