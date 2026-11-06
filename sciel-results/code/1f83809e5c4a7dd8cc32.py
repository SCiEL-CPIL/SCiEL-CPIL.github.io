import random
import time
import numpy as np

N = 64

def construct():
    # Precompute grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Precompute squared distance matrix
    # Shape: (num_points, num_points)
    # dist[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    coords = np.array(points)
    # Compute differences
    # dx = coords[:, 0:1] - coords[:, 0:1].T
    # dy = coords[:, 1:2] - coords[:, 1:2].T
    # dist = dx*dx + dy*dy
    
    # To save memory, we might not need the full matrix if we can compute on fly,
    # but 134MB is fine.
    dx = coords[:, 0:1] - coords[:, 0:1].T
    dy = coords[:, 1:2] - coords[:, 1:2].T
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    
    # Helper to check if a point can be added to a set represented by indices
    # S_indices: list of indices of points in S
    # dist_sets: list of sets of distances from S[i] to other points in S
    def can_add(p_idx, S_indices, dist_sets):
        if len(S_indices) == 0:
            return True
        
        # Get distances from p to all points in S
        # S_indices is a list of ints
        d_p = dist_matrix[p_idx, S_indices]
        
        # Check 1: Distances from p to S must be distinct
        # If any two distances are equal, p is apex of isosceles triangle
        if len(np.unique(d_p)) < len(S_indices):
            return False
            
        # Check 2: For each q in S, d(p, q) must not be in dist_sets[q]
        # dist_sets[i] corresponds to S_indices[i]
        for i, q_idx in enumerate(S_indices):
            d_val = d_p[i]
            if d_val in dist_sets[i]:
                return False
                
        return True

    def greedy_construct():
        S_indices = []
        dist_sets = []
        candidates = list(range(num_points))
        random.shuffle(candidates)
        
        for p_idx in candidates:
            if can_add(p_idx, S_indices, dist_sets):
                S_indices.append(p_idx)
                # Update dist_sets
                # New point is at index len(S_indices)-1
                # Its distances to existing points are d_p
                # We need to recompute d_p for the update
                # Actually, we can just compute it again or store it
                # Let's compute d_p again for simplicity in update
                if S_indices:
                    # p_idx is the last one added
                    # d_p was computed in can_add? No, local variable.
                    # Let's recompute for the update step
                    d_p = dist_matrix[p_idx, S_indices[:-1]]
                    # Update existing sets
                    for i in range(len(S_indices) - 1):
                        dist_sets[i].add(d_p[i])
                    # Add new set
                    dist_sets.append(set(d_p))
                else:
                    dist_sets.append(set())
                    
        return S_indices, dist_sets

    best_size = 0
    best_set = []
    
    start_time = time.time()
    time_limit = 290 # Leave 10s margin
    
    # Initial greedy runs
    for _ in range(100):
        S_indices, dist_sets = greedy_construct()
        if len(S_indices) > best_size:
            best_size = len(S_indices)
            best_set = S_indices
            best_dist_sets = dist_sets
        if time.time() - start_time > time_limit / 2:
            break
            
    # Local Search / ILS
    # Start from the best set found
    if best_size > 0:
        S_indices = best_set[:]
        dist_sets = best_dist_sets[:]
        # Deep copy sets
        dist_sets = [set(s) for s in dist_sets]
        
        current_size = len(S_indices)
        
        # Map from point index to position in S_indices for fast removal?
        # Or just use lists.
        
        while time.time() - start_time < time_limit:
            # Try to add a point
            added = False
            # Try random points not in S
            S_set = set(S_indices)
            # Get list of points not in S
            # This is expensive to generate every time.
            # We can maintain a list of "available" points?
            # Or just pick random indices and check if in S_set.
            
            # To speed up, try a batch of random candidates
            for _ in range(500):
                p_idx = random.randint(0, num_points - 1)
                if p_idx in S_set:
                    continue
                if can_add(p_idx, S_indices, dist_sets):
                    # Add
                    S_indices.append(p_idx)
                    S_set.add(p_idx)
                    # Update dist_sets
                    d_p = dist_matrix[p_idx, S_indices[:-1]]
                    for i in range(len(S_indices) - 1):
                        dist_sets[i].add(d_p[i])
                    dist_sets.append(set(d_p))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S_indices[:]
                        best_dist_sets = [set(s) for s in dist_sets]
                    added = True
                    break
                    
            if not added:
                # Stuck. Perform a kick.
                # Remove k random points
                k = random.randint(1, 3)
                if current_size <= k:
                    # Reset
                    S_indices, dist_sets = greedy_construct()
                    current_size = len(S_indices)
                    continue
                    
                # Pick k random indices to remove
                # We need to remove from S_indices and dist_sets
                # Removing an element from a list and updating all dist_sets is O(N^2)
                # Let's just rebuild dist_sets after removal?
                # Or do it carefully.
                
                remove_indices = random.sample(range(current_size), k)
                # Sort to remove from end to start to keep indices valid
                remove_indices.sort(reverse=True)
                
                for idx in remove_indices:
                    p_idx = S_indices[idx]
                    # Remove p_idx from S_indices
                    S_indices.pop(idx)
                    S_set.discard(p_idx)
                    # Remove p_idx from dist_sets
                    dist_sets.pop(idx)
                    # Remove distance to p_idx from all other dist_sets
                    # This is tricky because indices shift.
                    # It's safer to rebuild dist_sets from scratch if k is small?
                    # Or just mark for rebuild.
                    
                # Rebuild dist_sets for correctness
                # This is O(S^2)
                new_dist_sets = []
                for i, p_idx in enumerate(S_indices):
                    d_set = set()
                    for j, q_idx in enumerate(S_indices):
                        if i != j:
                            d_set.add(dist_matrix[p_idx, q_idx])
                    new_dist_sets.append(d_set)
                dist_sets = new_dist_sets
                current_size = len(S_indices)
                
                # Now try to greedily add points again
                # We can run a mini-greedy
                candidates = list(range(num_points))
                random.shuffle(candidates)
                for p_idx in candidates:
                    if p_idx in S_set:
                        continue
                    if can_add(p_idx, S_indices, dist_sets):
                        S_indices.append(p_idx)
                        S_set.add(p_idx)
                        d_p = dist_matrix[p_idx, S_indices[:-1]]
                        for i in range(len(S_indices) - 1):
                            dist_sets[i].add(d_p[i])
                        dist_sets.append(set(d_p))
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = S_indices[:]
                            best_dist_sets = [set(s) for s in dist_sets]
                        break # Just add one? Or continue?
                        # Let's continue adding as long as possible
                        # Actually, the loop above breaks after one add.
                        # Let's change it to a while loop or just run greedy_construct on the remaining?
                        # Simpler: just run a few greedy steps.

    return [points[i] for i in best_set]

# Run construct to get the result
# The function construct is the entry point.
