# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds limit, use 590 for safety
    
    d = 11
    
    # Generate candidate vectors with entries in {-2, -1, 0, 1, 2}
    # This is 5^11 = 48,828,125 vectors. Too many to generate all at once.
    # We will generate them in chunks or use a smarter enumeration.
    # However, 5^11 is manageable if we are careful.
    # Actually, let's start with entries in {-1, 0, 1} which is 3^11 = 177,147.
    # This is very fast to generate.
    
    candidates = []
    # Generate all vectors in {-1, 0, 1}^11
    # Using itertools.product
    for coords in itertools.product([-1, 0, 1], repeat=d):
        v = np.array(coords, dtype=np.int64)
        norm_sq = np.dot(v, v)
        if norm_sq > 0:
            candidates.append((norm_sq, v))
            
    # Sort by norm squared
    candidates.sort(key=lambda x: x[0])
    
    selected_vectors = []
    min_dist_sq = float('inf')
    max_norm_sq = 0
    
    # Greedy selection
    for norm_sq, v in candidates:
        if time.time() > deadline - 10:
            break
            
        new_max_norm = max(max_norm_sq, norm_sq)
        
        # Check if existing min distance is sufficient for the new max norm
        if min_dist_sq < new_max_norm:
            # We can still add v if its distance to all existing points is >= new_max_norm
            # But we must ensure the global min distance doesn't drop below new_max_norm
            # Since min_dist_sq is the min over existing pairs, and we only add v,
            # the new min distance will be min(min_dist_sq, min_dist_to_v).
            # So we require min(min_dist_sq, min_dist_to_v) >= new_max_norm.
            # This implies min_dist_sq >= new_max_norm AND min_dist_to_v >= new_max_norm.
            # If min_dist_sq < new_max_norm, we CANNOT add v unless we are willing to
            # accept that the condition might fail? No, the condition must hold for the final set.
            # So if min_dist_sq < new_max_norm, we cannot add v because the global min dist
            # would be < new_max_norm (which is the new max norm).
            # Wait, if we add v, the new max norm is new_max_norm.
            # The new min dist is min(old_min_dist, dist(v, S)).
            # We need new_min_dist >= new_max_norm.
            # So we need old_min_dist >= new_max_norm AND dist(v, S) >= new_max_norm.
            # If old_min_dist < new_max_norm, we fail.
            # So we can skip v.
            continue
        
        # Check distance to all existing vectors
        min_dist_to_v = float('inf')
        for u in selected_vectors:
            diff = v - u
            dist_sq = np.dot(diff, diff)
            if dist_sq < min_dist_to_v:
                min_dist_to_v = dist_sq
            if min_dist_to_v < new_max_norm:
                break
        
        if min_dist_to_v >= new_max_norm:
            # Add v
            selected_vectors.append(v)
            min_dist_sq = min(min_dist_sq, min_dist_to_v)
            max_norm_sq = new_max_norm
    
    if not selected_vectors:
        # Fallback to simple set
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    else:
        points = np.array(selected_vectors, dtype=np.int64)
        
    return points

# EVOLVE-BLOCK-END
