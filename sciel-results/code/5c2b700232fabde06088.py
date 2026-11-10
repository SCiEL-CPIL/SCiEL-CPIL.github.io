# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _normalize(points):
    """Center and normalize points to unit sphere."""
    points = points - np.mean(points, axis=0, keepdims=True)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    return points

def _relax(points, steps=500, cooling=0.999):
    """Repulsive force relaxation on the sphere."""
    n = points.shape[0]
    points = _normalize(points)
    temp = 0.1
    for _ in range(steps):
        if temp < 1e-6:
            break
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    # Random kick if coincident
                    diff = np.random.randn(3)
                    dist = np.linalg.norm(diff)
                # Inverse square repulsion
                force = diff / (dist * dist * dist) 
                forces[i] += force
                forces[j] -= force
        
        # Move points
        points += forces * temp
        points = _normalize(points)
        temp *= cooling
        
    return points

def _local_polish(points, steps=100):
    """Pairwise coordinate descent to fix min/max distances."""
    n = points.shape[0]
    points = _normalize(points)
    
    for _ in range(steps):
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            break
            
        # Identify critical pairs
        min_idx = np.argmin(dists)
        max_idx = np.argmax(dists)
        
        # Convert condensed index to pair indices
        def idx_to_pair(idx, n):
            # Solve k(k-1)/2 + j = idx for i, j (i < j)
            # Approximation: i ~ sqrt(2*idx)
            i = int((np.sqrt(8 * idx + 1) - 1) / 2)
            # j is the offset from the start of row i
            j = i + 1 + (idx - i * (i * (i - 1) // 2))
            return i, j

        i_min, j_min = idx_to_pair(min_idx, n)
        i_max, j_max = idx_to_pair(max_idx, n)
        
        # Try to increase d_min by moving i_min and j_min apart
        # Try to decrease d_max by moving i_max and j_max closer
        
        # Simple gradient step approximation
        # For min pair: move along vector connecting them
        vec_min = points[i_min] - points[j_min]
        norm_min = np.linalg.norm(vec_min)
        if norm_min > 1e-8:
            vec_min /= norm_min
            points[i_min] += vec_min * 0.01
            points[j_min] -= vec_min * 0.01
            
        # For max pair: move against vector connecting them
        vec_max = points[i_max] - points[j_max]
        norm_max = np.linalg.norm(vec_max)
        if norm_max > 1e-8:
            vec_max /= norm_max
            points[i_max] -= vec_max * 0.01
            points[j_max] += vec_max * 0.01
            
        points = _normalize(points)
        
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    n_points = 14
    dim = 3
    
    best_points = None
    best_score = -np.inf
    
    # Multi-start strategy
    n_restarts = 500
    
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random points on sphere, perturbed by Fibonacci sphere for diversity
        if r == 0:
            # First start: Fibonacci sphere (good initial distribution)
            points = np.zeros((n_points, 3))
            for i in range(n_points):
                y = 1 - (i / (n_points - 1)) * 2
                radius = np.sqrt(1 - y * y)
                theta = np.pi * (1 + 5) * i  # Golden angle
                points[i] = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        else:
            # Subsequent starts: Random with noise
            points = np.random.randn(n_points, dim)
            points = _normalize(points)
        
        # Stage 1: Coarse relaxation
        current = _relax(points, steps=200, cooling=0.995)
        
        # Stage 2: Fine relaxation
        current = _relax(current, steps=200, cooling=0.999)
        
        # Stage 3: Local polish
        current = _local_polish(current, steps=50)
        
        score = _score(current)
        
        if score > best_score:
            best_score = score
            best_points = current.copy()
            
            # If we found a very good score, we can try to refine it more
            if best_score > 0.5:
                 # Extra polish
                 current = _local_polish(current, steps=100)
                 score = _score(current)
                 if score > best_score:
                     best_score = score
                     best_points = current.copy()

    # If no valid points found (should not happen), return Fibonacci sphere
    if best_points is None:
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = np.pi * (1 + 5) * i
            points[i] = np.array([np.cos(theta) * radius, y, np.sin(theta) * radius])
        return points

    return best_points

# EVOLVE-BLOCK-END
