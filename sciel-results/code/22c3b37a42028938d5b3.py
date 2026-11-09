# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    such that min pairwise squared distance >= max squared norm.
    Aims to maximize the number of points.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit
    
    d = 11
    
    # Generate candidate points: all integer vectors with small coordinates
    # We'll use points with coordinates in {-2, -1, 0, 1, 2} initially
    # and expand if needed
    
    def get_candidates(max_coord, max_norm_sq=None):
        """Generate candidate integer points with bounded coordinates."""
        candidates = []
        for vals in product(range(-max_coord, max_coord + 1), repeat=d):
            v = np.array(vals, dtype=np.int64)
            norm_sq = int(np.dot(v, v))
            if norm_sq == 0:
                continue
            if max_norm_sq is not None and norm_sq > max_norm_sq:
                continue
            candidates.append(v)
        return candidates
    
    def min_pairwise_dist_sq(points):
        """Compute minimum squared distance between any pair of points."""
        if len(points) < 2:
            return float('inf')
        min_d = float('inf')
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                diff = points[i] - points[j]
                d_sq = int(np.dot(diff, diff))
                if d_sq < min_d:
                    min_d = d_sq
        return min_d
    
    def max_norm_sq(points):
        """Compute maximum squared norm of any point."""
        if len(points) == 0:
            return 0
        return max(int(np.dot(p, p)) for p in points)
    
    def is_valid(points):
        """Check if the set of points satisfies the constraint."""
        if len(points) < 2:
            return False
        if any(np.all(p == 0) for p in points):
            return False
        min_dist = min_pairwise_dist_sq(points)
        max_norm = max_norm_sq(points)
        return min_dist >= max_norm
    
    def greedy_select(candidates, points_list):
        """Try to add candidates to points_list while maintaining validity."""
        current = list(points_list)
        for cand in candidates:
            if any(np.array_equal(cand, p) for p in current):
                continue
            new_list = current + [cand]
            if is_valid(new_list):
                current = new_list
        return current
    
    def local_search(points, candidates, iterations=100):
        """Try to improve by swapping points."""
        current = list(points)
        current_set = set(tuple(p) for p in current)
        
        for _ in range(iterations):
            if time.time() > deadline - 30:
                break
            improved = False
            for i in range(len(current)):
                for cand in candidates:
                    if tuple(cand) in current_set:
                        continue
                    # Try swapping current[i] with cand
                    new_list = current[:i] + [cand] + current[i+1:]
                    if is_valid(new_list):
                        # Check if this actually improves (e.g., allows more additions later)
                        # For now, just accept if valid
                        current = new_list
                        current_set = set(tuple(p) for p in current)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return current
    
    # Start with basic symmetric configuration
    # Coordinate axes and their negatives: 2*11 = 22 points
    base_points = []
    for i in range(d):
        p_pos = np.zeros(d, dtype=np.int64)
        p_pos[i] = 1
        base_points.append(p_pos)
        p_neg = np.zeros(d, dtype=np.int64)
        p_neg[i] = -1
        base_points.append(p_neg)
    
    # Check validity of base
    if not is_valid(base_points):
        # Fallback to just two opposite points
        base_points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    
    best_points = list(base_points)
    best_k = len(best_points)
    
    # Try to expand with greedy addition
    # Generate candidates with small coordinates
    for max_coord in [1, 2, 3]:
        if time.time() > deadline - 60:
            break
        
        candidates = get_candidates(max_coord)
        # Filter candidates that could potentially be added
        # They must have norm_sq <= current min_pairwise_dist_sq
        
        current = list(best_points)
        
        # Sort candidates by norm to try adding smaller norm points first
        candidates.sort(key=lambda p: int(np.dot(p, p)))
        
        # Greedy addition
        for cand in candidates:
            if time.time() > deadline - 60:
                break
            if any(np.array_equal(cand, p) for p in current):
                continue
            new_list = current + [cand]
            if is_valid(new_list):
                current = new_list
        
        if len(current) > best_k:
            best_points = current
            best_k = len(current)
    
    # Try local search to improve
    if time.time() < deadline - 30:
        candidates = get_candidates(3)
        improved_points = local_search(best_points, candidates, iterations=50)
        if len(improved_points) > best_k:
            best_points = improved_points
            best_k = len(best_points)
    
    # Convert to numpy array
    result = np.array(best_points, dtype=np.int64)
    result = np.around(result).astype(np.int64)
    
    return result

# EVOLVE-BLOCK-END
