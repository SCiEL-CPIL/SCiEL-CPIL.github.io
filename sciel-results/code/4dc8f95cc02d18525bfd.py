import numpy as np
import itertools
import time
import sys
from collections import defaultdict

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 620.0  # Safety margin before 630s limit
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)
    
    def is_bad_5(points5):
        """Check if 5 points are co-spherical or co-planar using determinant test."""
        p = points5[0]
        others = points5[1:]
        diff = others - p  # (4, 3)
        norm_diff = (others ** 2).sum(axis=1) - (p ** 2).sum()  # (4,)
        w = np.column_stack([diff, norm_diff])  # (4, 4)
        det = np.linalg.det(w)
        return abs(det) < 0.5
    
    def count_bad_tuples(points):
        """Count how many bad 5-tuples exist in the given set of points."""
        k = len(points)
        if k < 5:
            return 0, {}
        
        bad_count = 0
        point_badness = defaultdict(int)
        
        # For efficiency, we'll sample a subset of 5-tuples if k is large
        # But for now, let's do it properly for small k
        if k <= 50:
            for combo in itertools.combinations(range(k), 5):
                pts5 = points[list(combo)]
                if is_bad_5(pts5):
                    bad_count += 1
                    for idx in combo:
                        point_badness[idx] += 1
        else:
            # For larger sets, use a heuristic: check tuples involving each point
            # with random samples of 4 other points
            np.random.seed(123)
            sample_size = min(200, k * 4)
            for _ in range(sample_size):
                combo = tuple(sorted(np.random.choice(k, size=5, replace=False)))
                pts5 = points[list(combo)]
                if is_bad_5(pts5):
                    bad_count += 1
                    for idx in combo:
                        point_badness[idx] += 1
        
        return bad_count, point_badness
    
    def greedy_reinsert(points, removed_points, grid):
        """Try to re-insert removed points in a smart order."""
        current = points.copy()
        current_set = set(map(tuple, current.tolist()))
        
        # Sort removed points by some heuristic (e.g., distance to center)
        center = np.array([3.5, 3.5, 3.5])
        dists = np.linalg.norm(removed_points - center, axis=1)
        order = np.argsort(dists + np.random.randn(len(dists)) * 0.5)
        
        for idx in order:
            p = removed_points[idx]
            if p.tobytes() in current_set:
                continue
            
            p_arr = p.reshape(1, 3)
            
            # Check if we can add p
            if len(current) < 4:
                current = np.vstack([current, p_arr])
                current_set.add(p.tobytes())
            else:
                # Check against all existing points
                diff = current - p
                norm_diff = (current ** 2).sum(axis=1) - (p ** 2).sum()
                w = np.column_stack([diff, norm_diff])
                
                can_add = True
                for combo in itertools.combinations(range(len(current)), 4):
                    sub = w[list(combo)]
                    if abs(np.linalg.det(sub)) < 0.5:
                        can_add = False
                        break
                
                if can_add:
                    current = np.vstack([current, p_arr])
                    current_set.add(p.tobytes())
        
        return current
    
    # Start with all 512 points
    current = grid.copy()
    removed = []
    
    # Iteratively remove the point involved in the most bad tuples
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        bad_count, point_badness = count_bad_tuples(current)
        
        if bad_count == 0:
            break
        
        # Find the point with the highest badness
        if not point_badness:
            break
        
        max_badness = max(point_badness.values())
        # Among points with max badness, pick one randomly for diversity
        candidates = [i for i, b in point_badness.items() if b == max_badness]
        remove_idx = np.random.choice(candidates)
        
        removed.append(current[remove_idx])
        current = np.delete(current, remove_idx, axis=0)
        
        if len(current) < 5:
            break
    
    # Now try to re-insert removed points
    if removed:
        removed_arr = np.array(removed, dtype=np.int64)
        current = greedy_reinsert(current, removed_arr, grid)
    
    # Final validation
    def validate_solution(points):
        if len(points) == 0:
            return True
        if np.any(points < 0) or np.any(points >= 8):
            return False
        # Check duplicates
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.array_equal(points[i], points[j]):
                    return False
        # Check no 5 co-spherical
        k = len(points)
        for i in range(4, k):
            p = points[i]
            prev = points[:i]
            diff = prev - p
            norm_diff = (prev ** 2).sum(axis=1) - (p ** 2).sum()
            w = np.column_stack([diff, norm_diff])
            for combo in itertools.combinations(range(i), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        return True
    
    if validate_solution(current):
        return current.astype(np.int64)
    
    # Fallback: use a known valid small set
    fallback = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    return fallback
