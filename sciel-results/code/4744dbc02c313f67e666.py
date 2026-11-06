import numpy as np
import time
import random

def _score_vectorized(points):
    """
    Calculate the maximum multiplicity of a positive doubled area.
    Uses vectorized NumPy operations to compute all triangle areas at once.
    """
    n = len(points)
    # points is a numpy array of shape (n, 2)
    
    # Compute vectors from p_i to p_j and p_i to p_k for all i < j < k
    # This is O(n^3) memory if done naively, but n=30 is small enough 
    # that we can use broadcasting efficiently.
    
    # However, for 30 points, 30*29*28/6 = 4060 triangles.
    # A fully vectorized approach creating a (4060, 3, 2) array is fine.
    
    # Get indices for all combinations
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_vals = i_idx[mask]
    j_vals = j_idx[mask]
    k_vals = k_idx[mask]
    
    p_i = points[i_vals]
    p_j = points[j_vals]
    p_k = points[k_vals]
    
    # Compute doubled area: |det(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    # Filter out zero areas
    areas = areas[areas > 0]
    
    if len(areas) == 0:
        return 0
        
    # Count frequencies
    # np.unique with return_counts is efficient for small arrays
    _, counts = np.unique(areas, return_counts=True)
    
    return int(np.max(counts))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial candidate: a simple grid which is a good local optimum
    # 30 points in a 5x6 grid
    init_points = [(x, y) for y in range(6) for x in range(5)]
    points = np.array(init_points, dtype=np.int64)
    
    # Pre-compute initial score
    current_score = _score_vectorized(points)
    best_points = points.copy()
    best_score = current_score
    
    # Local Search Strategy:
    # 1. Pick a random point
    # 2. Pick a random new location (integer coords in [-30, 30])
    # 3. If new location is not already occupied, try the move
    # 4. Evaluate new score
    # 5. Accept if better, or with some probability if worse (annealing)
    
    # To speed up evaluation, we can't easily incrementally update the score
    # without complex bookkeeping. But _score_vectorized is fast enough
    # (4060 triangles, vectorized ops ~ 10-50us per call).
    # 85s / 50us = ~1.7 million steps. This is plenty.
    
    temperature = 5.0
    cooling_rate = 0.999
    
    # Track valid positions to avoid duplicates
    # Using a set of tuples for O(1) lookup
    occupied = set(map(tuple, points))
    
    max_steps = 2000000 # Hard cap on steps to prevent infinite loop if time check is slow
    
    step = 0
    while time.time() < deadline and step < max_steps:
        step += 1
        
        # Pick a random point to move
        idx = np.random.randint(0, len(points))
        old_pos = points[idx]
        
        # Pick a random new position
        # Bias towards local moves? Or global?
        # Global random restarts within the box are good for escaping local minima.
        # Let's try a mix: 50% local (small delta), 50% global.
        if random.random() < 0.5:
            delta = np.array([random.randint(-3, 3), random.randint(-3, 3)])
            new_pos = old_pos + delta
        else:
            new_pos = np.array([random.randint(-30, 30), random.randint(-30, 30)])
        
        # Check bounds
        if np.any(new_pos < -30) or np.any(new_pos > 30):
            continue
            
        # Check collision
        if tuple(new_pos) in occupied:
            continue
            
        # Perform move
        occupied.remove(tuple(old_pos))
        occupied.add(tuple(new_pos))
        points[idx] = new_pos
        
        # Evaluate
        new_score = _score_vectorized(points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            # Metropolis criterion
            # Delta is negative or zero
            delta = new_score - current_score
            if delta < 0:
                # Accept with probability exp(delta / temp)
                # Since scores are integers, delta is integer.
                # If delta is -1, prob is exp(-1/temp)
                # If delta is -10, prob is exp(-10/temp)
                prob = np.exp(delta / temperature)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    points[idx] = old_pos
                    occupied.remove(tuple(new_pos))
                    occupied.add(tuple(old_pos))
            
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1

    return [tuple(p) for p in best_points]
