# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximizes the number of points in an 8x8x8 grid such that no 5 points are co-spherical.
    Uses Simulated Annealing to escape local optima found by greedy methods.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid) # 512
    
    # Map point index to coordinates and vice versa
    point_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    
    # Precompute squared norms for faster determinant calculation
    sq_norms = (grid ** 2).sum(axis=1)

    def is_co_spherical(p_idx, q1_idx, q2_idx, q3_idx, q4_idx):
        """
        Checks if 5 points are co-spherical.
        Uses the determinant condition:
        det([q_i - p, |q_i|^2 - |p|^2]) = 0
        """
        p = grid[p_idx]
        q1 = grid[q1_idx]
        q2 = grid[q2_idx]
        q3 = grid[q3_idx]
        q4 = grid[q4_idx]
        
        # Matrix rows: (q - p), |q|^2 - |p|^2
        # We can form a 4x4 matrix
        # To avoid float errors, we use integers if possible, but det is float.
        # The condition is |det| < 0.5
        
        v1 = q1 - p
        v2 = q2 - p
        v3 = q3 - p
        v4 = q4 - p
        
        d1 = sq_norms[q1_idx] - sq_norms[p_idx]
        d2 = sq_norms[q2_idx] - sq_norms[p_idx]
        d3 = sq_norms[q3_idx] - sq_norms[p_idx]
        d4 = sq_norms[q4_idx] - sq_norms[p_idx]
        
        M = np.array([
            [v1[0], v1[1], v1[2], d1],
            [v2[0], v2[1], v2[2], d2],
            [v3[0], v3[1], v3[2], d3],
            [v4[0], v4[1], v4[2], d4]
        ], dtype=np.float64)
        
        det_val = np.linalg.det(M)
        return abs(det_val) < 0.5

    def can_add_to_set(S, p_idx):
        """
        Checks if adding p_idx to set S (list of indices) keeps it valid.
        S must have at least 4 points for a check to be meaningful.
        """
        if len(S) < 4:
            return True
            
        # Check all combinations of 4 points from S
        # Optimization: If S is large, this is expensive.
        # But k is expected to be < 50. C(50,4) ~ 230k.
        # 230k * (fast det) is okay for a few thousand steps.
        
        S_list = list(S)
        for combo in itertools.combinations(S_list, 4):
            if is_co_spherical(p_idx, combo[0], combo[1], combo[2], combo[3]):
                return False
        return True

    def can_swap(S, out_idx, in_idx):
        """
        Checks if swapping out_idx (in S) with in_idx (not in S) keeps S valid.
        This is equivalent to:
        1. Removing out_idx from S (always valid)
        2. Adding in_idx to (S - {out_idx})
        """
        S_new = [x for x in S if x != out_idx]
        return can_add_to_set(S_new, in_idx)

    # 1. Generate Initial Solution using Greedy
    def greedy_initial():
        best_set = []
        np.random.seed(42)
        for _ in range(10): # 10 restarts
            order = np.random.permutation(N)
            current_set = []
            for idx in order:
                if can_add_to_set(current_set, idx):
                    current_set.append(idx)
            if len(current_set) > len(best_set):
                best_set = current_set
        return best_set

    start_time = time.time()
    deadline = start_time + 580 # 20s buffer
    
    best_set = greedy_initial()
    best_size = len(best_set)
    
    # 2. Simulated Annealing
    current_set = list(best_set)
    current_size = len(current_set)
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Track best found during SA
    global_best_set = list(current_set)
    global_best_size = current_size
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Choose move type
        # Bias towards adding points
        r = random.random()
        if r < 0.5:
            # Add
            candidates = [i for i in range(N) if i not in current_set]
            if not candidates:
                continue
            move_idx = random.choice(candidates)
            if can_add_to_set(current_set, move_idx):
                # Valid add, always accept
                current_set.append(move_idx)
                current_size += 1
                if current_size > global_best_size:
                    global_best_size = current_size
                    global_best_set = list(current_set)
            # If invalid, reject (stay same)
            
        elif r < 0.8:
            # Remove
            if current_size == 0:
                continue
            move_idx = random.choice(current_set)
            # Remove is always valid
            # Accept with probability exp(-1/T) to allow shrinking
            if random.random() < np.exp(-1.0 / T):
                current_set.remove(move_idx)
                current_size -= 1
                
        else:
            # Swap
            if current_size == 0 or current_size == N:
                continue
            out_idx = random.choice(current_set)
            candidates = [i for i in range(N) if i not in current_set]
            in_idx = random.choice(candidates)
            if can_swap(current_set, out_idx, in_idx):
                # Valid swap, size unchanged
                # Accept with probability exp(-0/T) = 1? 
                # We accept swaps to explore.
                # But we might cycle. 
                # Let's accept with prob 0.5 to allow backtracking?
                # Or just accept if it leads to a "better" local structure? 
                # Simple: Accept with prob 0.5.
                if random.random() < 0.5:
                    current_set.remove(out_idx)
                    current_set.append(in_idx)
                    # Size unchanged, so no update to global best size
                    # But check if this new configuration allows future growth? 
                    # SA handles that.
                    
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Periodic check for time
        if step % 100 == 0:
            if time.time() > deadline - 5:
                break

    # Convert best set to array
    if global_best_size > 0:
        result = np.array([grid[i] for i in global_best_set], dtype=np.int64)
    else:
        result = np.empty((0, 3), dtype=np.int64)
        
    return result

# EVOLVE-BLOCK-END
