import time
import random
import numpy as np
from collections import defaultdict

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290

    # Precompute points and distance matrix
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    # D[i, j] is squared distance between i and j
    # Using broadcasting might be memory heavy for 4096x4096 int32? 4096*4096*4 = 64MB. OK.
    # But calculating it takes time.
    # Let's do it efficiently.
    # Actually, we can compute on the fly if needed, but precomputation is better for repeated checks.
    
    # Alternative: Compute distances on the fly for the specific points in S.
    # Since |S| is small, computing d(p, s) on the fly is O(1) per pair.
    # d(p, s) = (x_p - x_s)^2 + (y_p - y_s)^2.
    # This avoids the 64MB matrix and precomputation time.
    # Given M=4096, and we check many candidates, precomputation might still be faster
    # if the same pairs are checked often? No, candidates are random.
    # Let's stick to on-the-fly distance calculation to save memory and precompute time.
    # It's just 2 subtractions, 2 multiplications, 1 addition. Very fast in Python?
    # No, Python loops are slow.
    # Let's use the numpy matrix but maybe int16? Max dist is 63^2+63^2 = 7938. Fits in int16.
    # 4096*4096*2 = 32MB.
    
    # Let's try to precompute.
    # If precomputation takes too long, we'll fail.
    # 4096^2 = 16M elements.
    # Numpy broadcasting is fast.
    
    # Let's use a different approach:
    # We don't need the full matrix. We just need distances from a candidate p to points in S.
    # We can store S as a numpy array of indices.
    # Then get coords of S: coords_S = points[S]
    # coords_p = points[p]
    # d_sq = np.sum((coords_S - coords_p)**2, axis=1)
    # This is vectorized and fast.

    def can_add(p, S_indices, dist_sets):
        """
        S_indices: numpy array of indices in S
        dist_sets: dict mapping index -> set of squared distances to other points in S
        """
        if len(S_indices) == 0:
            return True
        
        coords_p = points[p]
        coords_S = points[S_indices]
        
        # Calculate squared distances from p to all s in S
        d_sq = np.sum((coords_S - coords_p)**2, axis=1)
        
        # Condition 1: All d_sq must be distinct
        # Check if any duplicate
        if len(np.unique(d_sq)) != len(d_sq):
            return False
            
        # Condition 2: For each s, d(p, s) not in dist_sets[s]
        # dist_sets[s] contains distances from s to other points in S.
        # We need to check if d(p, s) is in that set.
        
        # Vectorized check?
        # We have d_sq array. We have list of sets.
        # We can loop through S_indices. |S| is small (~10-20).
        for i, s in enumerate(S_indices):
            if d_sq[i] in dist_sets[s]:
                return False
                
        return True

    def add_point(p, S_indices, dist_sets):
        """
        Adds p to S. Updates S_indices and dist_sets.
        Returns new S_indices.
        """
        coords_p = points[p]
        new_dist_set = set()
        
        # Update dist_sets for existing points and compute distances for p
        new_S_indices = np.append(S_indices, p)
        
        # We need to update dist_sets for all existing s
        # and create dist_sets[p]
        
        # To avoid modifying dist_sets while iterating, we can just update.
        # dist_sets is a dict.
        
        # 1. Add p to dist_sets
        dist_sets[p] = new_dist_set
        
        # 2. Update existing points
        # We need distances from p to all s in old S
        # We can reuse the calculation from can_add? No, can_add doesn't return them.
        # Let's just recalculate or pass them.
        # For simplicity, recalculate. |S| is small.
        
        if len(S_indices) > 0:
            coords_S = points[S_indices]
            d_sq = np.sum((coords_S - coords_p)**2, axis=1)
            
            for i, s in enumerate(S_indices):
                d_val = int(d_sq[i])
                # Add to p's set
                new_dist_set.add(d_val)
                # Add to s's set
                dist_sets[s].add(d_val)
                
        # Update S_indices
        return new_S_indices

    def remove_point(p, S_indices, dist_sets):
        """
        Removes p from S.
        """
        # Remove p from dist_sets
        if p in dist_sets:
            del dist_sets[p]
            
        # Remove p from S_indices
        mask = S_indices != p
        new_S_indices = S_indices[mask]
        
        # Update dist_sets for remaining points
        # We need to remove distance d(p, s) from dist_sets[s]
        if len(new_S_indices) > 0:
            coords_p = points[p]
            coords_S = points[new_S_indices]
            d_sq = np.sum((coords_S - coords_p)**2, axis=1)
            
            for i, s in enumerate(new_S_indices):
                d_val = int(d_sq[i])
                dist_sets[s].discard(d_val)
                
        return new_S_indices

    best_S = np.array([], dtype=np.int32)
    best_dist_sets = {}
    best_size = 0

    # Greedy Phase with multiple restarts
    for restart in range(200):
        if time.time() > deadline - 30:
            break
            
        S_indices = np.array([], dtype=np.int32)
        dist_sets = {}
        
        # Random order of candidates
        candidates = np.random.permutation(M)
        
        for p in candidates:
            if time.time() > deadline - 30:
                break
            if can_add(p, S_indices, dist_sets):
                S_indices = add_point(p, S_indices, dist_sets)
        
        if len(S_indices) > best_size:
            best_size = len(S_indices)
            best_S = S_indices.copy()
            # Deep copy dist_sets
            best_dist_sets = {k: set(v) for k, v in dist_sets.items()}

    # Local Search Phase
    # We will try to improve the best set found.
    current_S = best_S.copy()
    current_dist_sets = {k: set(v) for k, v in best_dist_sets.items()}
    current_size = best_size

    for iteration in range(10000):
        if time.time() > deadline - 5:
            break
            
        if current_size == 0:
            break

        # Perturbation: Remove k random points
        k = random.randint(1, min(3, current_size))
        to_remove = np.random.choice(current_S, k, replace=False)
        
        for p in to_remove:
            current_S = remove_point(int(p), current_S, current_dist_sets)
            
        # Greedy Reconstruction
        # Try to add points from the full set of candidates not in current_S
        # To speed up, we can just iterate through all M points and try to add them
        # in a random order.
        
        candidates = np.random.permutation(M)
        
        for p in candidates:
            if time.time() > deadline - 5:
                break
            if p not in current_dist_sets: # p is not in current_S
                if can_add(p, current_S, current_dist_sets):
                    current_S = add_point(p, current_S, current_dist_sets)
                    
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = current_S.copy()
            best_dist_sets = {k: set(v) for k, v in current_dist_sets.items()}
            current_size = best_size
        else:
            # Revert to best
            current_S = best_S.copy()
            current_dist_sets = {k: set(v) for k, v in best_dist_sets.items()}
            current_size = best_size

    return [tuple(points[i]) for i in best_S]
