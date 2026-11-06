"""No-isosceles set construction on [0, N)^2 grid."""
import time
import random
import math
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    # Pre-compute all grid points and indices
    all_points = [(x, y) for x in range(N) for y in range(N)]
    M = len(all_points)
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # Pre-compute squared distance matrix using uint16 for memory efficiency
    coords = np.array(all_points)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
    dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
    D = (dx * dx + dy * dy).astype(np.uint16)
    
    # For faster validity checks, precompute distance arrays for each point
    # D_rows[i] = D[i, :] which is a 1D array of distances from point i to all others
    
    def is_valid_addition_fast(candidate_idx, current_set_indices):
        """Check if adding candidate_idx to current_set creates any isosceles triangle.
        Uses precomputed D matrix for speed."""
        n = len(current_set_indices)
        if n == 0:
            return True
        if n == 1:
            return True
        if n == 2:
            # Check if the three points form an isosceles triangle
            i, j = current_set_indices[0], current_set_indices[1]
            d_ci = D[candidate_idx, i]
            d_cj = D[candidate_idx, j]
            d_ij = D[i, j]
            # Isosceles if any two sides are equal
            if d_ci == d_cj or d_ci == d_ij or d_cj == d_ij:
                return False
            return True
        
        cur_arr = np.array(current_set_indices, dtype=np.int64)
        
        # Distances from candidate to all current points
        d_c_s = D[candidate_idx, cur_arr]
        
        # Check for duplicates in d_c_s (vertex at candidate)
        # If any two distances are equal, isosceles triangle formed with candidate as apex
        if len(np.unique(d_c_s)) != n:
            return False
            
        # Distances between current points
        d_s_s = D[cur_arr][:, cur_arr]
        
        # Check if any distance from candidate to A equals distance between A and B
        if (d_s_s == d_c_s[:, None]).any():
            return False
            
        return True

    def greedy_construct():
        """Greedy construction with random ordering."""
        current_set = []
        in_set = set()
        remaining = list(range(M))
        random.shuffle(remaining)
        
        for candidate_idx in remaining:
            if time.time() > deadline:
                break
            if candidate_idx in in_set:
                continue
            if is_valid_addition_fast(candidate_idx, current_set):
                current_set.append(candidate_idx)
                in_set.add(candidate_idx)
                
        return current_set

    def local_search(initial_set, time_budget=30.0):
        """Local search with add and swap operations.
        Uses a time budget instead of fixed iterations."""
        current_set = list(initial_set)
        in_set = set(current_set)
        best_set = list(current_set)
        best_size = len(current_set)
        
        # Precompute which points are not in the set for fast random selection
        not_in_set = [i for i in range(M) if i not in in_set]
        
        while time.time() < start_time + time_budget:
            # Try to add a random point from not_in_set
            if not_in_set:
                candidate_idx = random.choice(not_in_set)
                if is_valid_addition_fast(candidate_idx, current_set):
                    current_set.append(candidate_idx)
                    in_set.add(candidate_idx)
                    not_in_set.remove(candidate_idx)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_size = len(current_set)
            
            # Try swap: remove one point, try to add another
            if len(current_set) > 0 and random.random() < 0.4:
                idx_to_remove = random.randint(0, len(current_set) - 1)
                removed_idx = current_set.pop(idx_to_remove)
                in_set.discard(removed_idx)
                not_in_set.append(removed_idx)
                
                # Try to add a better point from not_in_set
                added = False
                # Sample up to 50 candidates
                sample_size = min(50, len(not_in_set))
                if sample_size > 0:
                    candidates = random.sample(not_in_set, sample_size)
                    for candidate_idx in candidates:
                        if is_valid_addition_fast(candidate_idx, current_set):
                            current_set.append(candidate_idx)
                            in_set.add(candidate_idx)
                            not_in_set.remove(candidate_idx)
                            added = True
                            if len(current_set) > best_size:
                                best_set = list(current_set)
                                best_size = len(current_set)
                            break
                
                if not added:
                    # Put the removed point back
                    current_set.append(removed_idx)
                    in_set.add(removed_idx)
                    not_in_set.remove(removed_idx)
        
        return best_set

    best_overall = []
    best_size_overall = 0
    
    # Use adaptive number of restarts based on time
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        
        random.seed(restart * 12345 + 67890)
        
        current_set = greedy_construct()
        if len(current_set) > best_size_overall:
            best_overall = list(current_set)
            best_size_overall = len(current_set)
        
        # Do local search if we have time
        time_remaining = deadline - time.time()
        if time_remaining > 20:
            search_time = min(25.0, time_remaining - 15)
            improved = local_search(current_set, time_budget=search_time)
            if len(improved) > best_size_overall:
                best_overall = list(improved)
                best_size_overall = len(improved)
    
    # Convert indices back to points
    return [all_points[i] for i in best_overall]
