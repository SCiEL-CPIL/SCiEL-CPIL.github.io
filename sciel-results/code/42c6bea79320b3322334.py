# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that 
    their minimum pairwise squared distance is greater than their maximum squared norm.
    Uses greedy set-packing with multiple norm levels.
    """
    start_time = time.time()
    DEADLINE = 600  # seconds, with safety margin
    d = 11
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - elapsed()
    
    # Generate candidate points with small norms
    # Norm 1: single coordinate ±1
    # Norm 2: two coordinates ±1
    # Norm 3: three coordinates ±1, or one coord ±sqrt(3) not integer, so three ±1
    # Norm 4: four ±1, or one ±2
    # Norm 5: five ±1, or one ±2 and one ±1
    # etc.
    
    candidates = []
    
    # Norm squared = 1: one coordinate is ±1
    for i in range(d):
        for sign in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = sign
            candidates.append(p)
    
    # Norm squared = 2: two coordinates are ±1
    for i, j in itertools.combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                candidates.append(p)
    
    # Norm squared = 3: three coordinates are ±1
    for i, j, k in itertools.combinations(range(d), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    p[k] = sk
                    candidates.append(p)
    
    # Norm squared = 4: four ±1, or one ±2
    for i, j, k, l in itertools.combinations(range(d), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = si
                        p[j] = sj
                        p[k] = sk
                        p[l] = sl
                        candidates.append(p)
    for i in range(d):
        for sign in [2, -2]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = sign
            candidates.append(p)
    
    # Norm squared = 5: five ±1, or one ±2 and one ±1
    for combo in itertools.combinations(range(d), 5):
        for signs in itertools.product([1, -1], repeat=5):
            p = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                p[idx] = val
            candidates.append(p)
    for i, j in itertools.combinations(range(d), 2):
        for si in [2, -2]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                candidates.append(p)
    
    # Convert to array
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cand = len(cand_arr)
    
    # Compute norm squared for each candidate
    norms_sq = np.sum(cand_arr ** 2, axis=1)
    
    # For a valid set, we need: for all pairs (i,j), ||p_i - p_j||^2 > max_norm_sq
    # where max_norm_sq = max(||p_i||^2) over all i in the set
    # 
    # Strategy: Try different max_norm_sq thresholds and find the largest compatible set
    # For a given threshold T, all points must have norm_sq <= T, and all pairwise distances > T
    
    # Precompute all pairwise squared distances (this is expensive for large n_cand)
    # n_cand can be large, so let's be smart about it
    
    # Actually, let's use a greedy approach:
    # Sort candidates by norm_sq, then greedily add points that are compatible with all existing points
    
    # For efficiency, let's try multiple strategies:
    # Strategy 1: Greedy by norm level
    # For each possible max_norm, find the largest set with all norms <= max_norm and pairwise dist > max_norm
    
    best_points = None
    best_count = 0
    
    # Try max_norm_sq from 1 to 10
    for max_norm_sq in range(1, 11):
        if remaining() < 30:
            break
        
        # Filter candidates with norm_sq <= max_norm_sq
        mask = norms_sq <= max_norm_sq
        filtered = cand_arr[mask]
        n_filt = len(filtered)
        
        if n_filt < 2:
            continue
        
        # We need pairwise squared distances > max_norm_sq
        # Greedy: add points one by one if compatible with all existing
        
        selected = []
        selected_indices = []
        
        for i in range(n_filt):
            if remaining() < 60:
                break
            
            p = filtered[i]
            compatible = True
            
            for j in range(len(selected)):
                diff = p - selected[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq <= max_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                selected.append(p)
                selected_indices.append(i)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_points = np.array(selected, dtype=np.int64)
        
        if remaining() < 60:
            break
    
    # If we didn't find anything, use the trivial 2-point solution
    if best_points is None or len(best_points) < 2:
        best_points = np.array([np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)])
    
    # Final validation
    k = len(best_points)
    if k < 2:
        best_points = np.array([np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)])
    
    # Check no zero vector
    for i in range(len(best_points)):
        if np.all(best_points[i] == 0):
            best_points = np.array([np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)])
            break
    
    return best_points

# EVOLVE-BLOCK-END
