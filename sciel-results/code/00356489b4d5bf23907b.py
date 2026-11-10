import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)
    N = len(grid)
    
    def det_check(p_idx, s1, s2, s3, s4):
        """Compute |det| for the 4x4 matrix formed by p and 4 other points."""
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        
        W = np.empty((4, 4), dtype=np.float64)
        W[0, 0:3] = grid[s1] - p
        W[0, 3] = grid_sq[s1] - p_sq
        W[1, 0:3] = grid[s2] - p
        W[1, 3] = grid_sq[s2] - p_sq
        W[2, 0:3] = grid[s3] - p
        W[2, 3] = grid_sq[s3] - p_sq
        W[3, 0:3] = grid[s4] - p
        W[3, 3] = grid_sq[s4] - p_sq
        return abs(np.linalg.det(W))
    
    def is_valid_addition(p_idx, selected_indices):
        """Check if adding p_idx to selected_indices is valid (no 5 cospherical)."""
        m = len(selected_indices)
        if m < 4:
            return True
        
        # Check all 4-combinations of selected_indices
        for combo in itertools.combinations(selected_indices, 4):
            if det_check(p_idx, combo[0], combo[1], combo[2], combo[3]) < 0.5:
                return False
        return True
    
    start_time = time.time()
    time_limit = 550
    
    best_set = []
    best_size = 0
    rng = np.random.RandomState(777)
    
    # Simulated annealing with multiple restarts
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with greedy random
        current_set = []
        order = rng.permutation(N)
        for idx in order:
            if time.time() - start_time > time_limit:
                break
            if is_valid_addition(idx, current_set):
                current_set.append(idx)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        # Simulated annealing phase
        T = 1.0
        T_min = 0.05
        cooling = 0.9995
        
        for step in range(10000):
            if time.time() - start_time > time_limit:
                break
            
            # Try to add a random point
            idx = rng.randint(N)
            if idx in current_set:
                continue
            
            if is_valid_addition(idx, current_set):
                current_set.append(idx)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
            else:
                # With probability exp(-1/T), remove a random point to explore
                if random.random() < np.exp(-1.0 / max(T, T_min)):
                    if current_set:
                        remove_pos = rng.randint(len(current_set))
                        current_set.pop(remove_pos)
            
            T = max(T_min, T * cooling)
            
            # Periodic greedy improvement
            if step % 500 == 0:
                in_set = set(current_set)
                for idx in rng.permutation(N):
                    if idx in in_set:
                        continue
                    if is_valid_addition(idx, current_set):
                        current_set.append(idx)
                        in_set.add(idx)
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
    
    # Final greedy pass on best_set
    if best_set:
        in_set = set(best_set)
        for idx in rng.permutation(N):
            if idx in in_set:
                continue
            if is_valid_addition(idx, best_set):
                best_set.append(idx)
                in_set.add(idx)
                best_size = len(best_set)
    
    if not best_set:
        return np.empty((0, 3), dtype=np.int64)
    
    return grid[best_set].astype(np.int64)
