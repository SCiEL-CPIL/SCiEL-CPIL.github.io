import numpy as np
from itertools import product
import random

def nikodym_set() -> np.ndarray:
    p = 7
    all_points = list(product(range(p), repeat=4))
    num_points = len(all_points)
    
    # Map point to index
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    idx_to_point = {i: pt for i, pt in enumerate(all_points)}
    
    # Precompute lines
    # We need to know for each pair of points, if they are on a line (they always are in F_q^2)
    # Actually, we just need the update rule:
    # Adding x to C reduces good_lines_count[y] by 1 for all y != x.
    
    # Initial good_lines_count is 50 for all points.
    # Why 50? There are 50 lines through any point in F_49^2 (49 slopes + vertical).
    
    def solve_greedy(seed):
        random.seed(seed)
        good_lines_count = np.full(num_points, 50, dtype=int)
        in_C = np.zeros(num_points, dtype=bool)
        
        # List of indices not in C
        candidates = list(range(num_points))
        random.shuffle(candidates)
        
        # We will iterate through candidates.
        # To be efficient, we can just pick a random candidate from the remaining ones.
        # Or iterate through the shuffled list and try to add.
        
        for x_idx in candidates:
            if in_C[x_idx]:
                continue
            
            # Check condition to add x_idx
            # 1. good_lines_count[x_idx] >= 1
            if good_lines_count[x_idx] < 1:
                continue
                
            # 2. min_{y != x_idx} good_lines_count[y] >= 2
            # We need to check if any other point has count < 2.
            # Since counts only decrease, if min_count < 2, we might have a problem.
            # Let's find the min count excluding x_idx.
            
            # Optimization: maintain global min?
            # Counts are small (max 50).
            # We can just check if there is any y != x_idx with good_lines_count[y] < 2.
            
            can_add = True
            # This check is O(N). N=2401.
            # Total operations ~ 2401 * 2401 ~ 5.7M. Fast enough.
            for y_idx in range(num_points):
                if y_idx == x_idx:
                    continue
                if good_lines_count[y_idx] < 2:
                    can_add = False
                    break
            
            if can_add:
                in_C[x_idx] = True
                # Update counts
                # For all y != x_idx, good_lines_count[y] -= 1
                # good_lines_count[x_idx] stays same.
                # We can do this by subtracting 1 from all, then adding 1 back to x_idx?
                # No, x_idx is now in C. Its count doesn't change.
                # Others decrease.
                
                # Efficient update:
                # We can't just subtract from all easily without a loop.
                # But we can maintain a global offset?
                # Let offset = number of points added to C so far.
                # Then actual_count[y] = initial_count[y] - offset + (1 if y == x_idx else 0)?
                # No, because x_idx's count doesn't decrease when it is added, but it did decrease when previous points were added.
                
                # Let's stick to the array update.
                # good_lines_count -= 1
                # good_lines_count[x_idx] += 1  # restore
                good_lines_count -= 1
                good_lines_count[x_idx] += 1
                
        # Return the set C
        C_indices = np.where(in_C)[0]
        return C_indices

    # Run multiple restarts to find a large C
    best_C = set()
    best_size = 0
    
    # We can run a few restarts with different seeds
    # Since the problem is symmetric, the size should be similar.
    # But random order might affect the final size.
    
    for seed in range(10):
        C_indices = solve_greedy(seed)
        C_size = len(C_indices)
        if C_size > best_size:
            best_size = C_size
            best_C = set(C_indices)
            
    # Convert best_C indices to points
    C_points = set()
    for idx in best_C:
        C_points.add(idx_to_point[idx])
        
    N_points = set(all_points) - C_points
    
    # Convert to required format
    points_list = []
    for pt in sorted(N_points):
        points_list.append([[pt[0], pt[1]], [pt[2], pt[3]]])
        
    if not points_list:
        for pt in sorted(all_points):
            points_list.append([[pt[0], pt[1]], [pt[2], pt[3]]])
            
    return np.array(points_list, dtype=np.int64)
