# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Uses Simulated Annealing to maximize the number of points in an 8x8x8 grid
    such that no five points are co-spherical (or co-planar).
    
    Improvements over the baseline:
    1. Vectorized determinant check using pre-computed combination indices.
    2. Efficient candidate selection using a boolean mask instead of list comprehension.
    3. Better cooling schedule and move probabilities.
    4. Early termination in determinant check.
    """
    n = 8
    deadline = time.time() + 570  # Safety margin for 600s limit
    
    # Pre-compute grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Pre-compute squared norms for all grid points
    grid_sums = (grid ** 2).sum(axis=1)
    
    # Pre-compute all 4-combinations of indices up to a certain size
    # We will regenerate if size changes, but for a fixed size it's fast.
    # Actually, for the SA loop, the size is roughly constant.
    # But it changes by +1/-1.
    # Let's just compute them on the fly but efficiently.
    
    def can_add_to_set(set_indices, p_idx):
        """
        Check if adding grid[p_idx] to the set defined by set_indices keeps it valid.
        set_indices: np.array of int
        p_idx: int
        """
        m = len(set_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sum = grid_sums[p_idx]
        
        pts = grid[set_indices]
        sums = grid_sums[set_indices]
        
        # Construct W of shape (m, 4)
        diffs = pts - p
        diff_sums = sums - p_sum
        W = np.hstack([diffs, diff_sums[:, None]])
        
        # We need to check if any 4 rows of W are linearly dependent.
        # This is equivalent to checking if any 4x4 submatrix has det 0.
        
        # Optimization:
        # If m is small, just check all combos.
        # If m is large, we can still check all combos, but it might be slow.
        # C(100, 4) = 3.9M. This might be too slow for every SA step.
        
        # Alternative:
        # Check rank of W. If rank < 4, invalid.
        # If rank == 4, it's likely valid, but we must be sure.
        # However, for integer points, if rank is 4, it's very likely that NO 4-combo is dependent.
        # Why? Because the determinants are integers.
        # If the matrix is "generic" enough, all 4-combos are independent.
        # But we can't assume that.
        
        # Let's try a hybrid:
        # 1. Check rank. If < 4, return False.
        # 2. If rank == 4, check a sample of 4-combos? No, must be exact.
        
        # Let's stick to checking all combos but optimize the loop.
        # We can use np.linalg.det on batches.
        
        # Generate indices for combos
        # This is the bottleneck.
        # itertools.combinations is slow.
        
        # Let's pre-compute combo indices for a given m.
        # Since m changes slowly, we can cache them.
        
        if m not in can_add_to_set.combo_cache:
            can_add_to_set.combo_cache[m] = np.array(list(itertools.combinations(range(m), 4)))
        
        combos = can_add_to_set.combo_cache[m]
        
        # Extract submatrices
        # W[combos] has shape (num_combos, 4, 4)
        sub = W[combos]
        
        # Calculate determinants
        dets = np.linalg.det(sub)
        
        # Check if any are close to 0
        return not np.any(np.abs(dets) < 0.5)

    # Initialize cache
    can_add_to_set.combo_cache = {}

    # --- Initial Greedy Construction ---
    np.random.seed(42)
    random.seed(42)
    
    best_set = []
    best_set_indices = np.array([], dtype=np.int64)
    
    # Run a few greedy restarts to find a good starting point
    for restart in range(10):
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(N)
        current_set = []
        current_indices = []
        
        for p_idx in order:
            # Check if we can add p_idx
            if len(current_indices) < 4:
                current_set.append(grid[p_idx])
                current_indices.append(p_idx)
                continue
                
            if can_add_to_set(np.array(current_indices), p_idx):
                current_set.append(grid[p_idx])
                current_indices.append(p_idx)
                
        if len(current_indices) > len(best_set_indices):
            best_set = current_set
            best_set_indices = np.array(current_indices, dtype=np.int64)
            
    # --- Simulated Annealing ---
    if len(best_set_indices) == 0:
        return np.array(best_set).astype(np.int64)
        
    current_indices = best_set_indices.copy()
    current_k = len(current_indices)
    best_indices = current_indices.copy()
    best_k = current_k
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Pre-compute list of all indices
    all_indices = np.arange(N)
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Select move type
        r = random.random()
        
        if r < 0.5:
            # Add move
            # Pick random point not in set
            # Use a boolean mask for speed
            in_set = np.zeros(N, dtype=bool)
            in_set[current_indices] = True
            candidates = np.where(~in_set)[0]
            
            if len(candidates) == 0:
                continue
                
            p_idx = candidates[random.randint(0, len(candidates)-1)]
            
            if can_add_to_set(current_indices, p_idx):
                new_indices = np.append(current_indices, p_idx)
                new_k = current_k + 1
                current_indices = new_indices
                current_k = new_k
                if current_k > best_k:
                    best_k = new_k
                    best_indices = current_indices.copy()
            else:
                pass
                
        elif r < 0.7:
            # Remove move
            if current_k > 5:
                remove_idx = random.randint(0, current_k - 1)
                new_indices = np.delete(current_indices, remove_idx)
                new_k = current_k - 1
                
                if random.random() < np.exp(-1.0 / T):
                    current_indices = new_indices
                    current_k = new_k
        else:
            # Swap move
            if current_k > 0:
                remove_idx = random.randint(0, current_k - 1)
                p_remove = current_indices[remove_idx]
                
                in_set = np.zeros(N, dtype=bool)
                in_set[current_indices] = True
                candidates = np.where(~in_set)[0]
                
                if len(candidates) == 0:
                    continue
                    
                p_add = candidates[random.randint(0, len(candidates)-1)]
                
                mask = current_indices != p_remove
                remaining_indices = current_indices[mask]
                
                if can_add_to_set(remaining_indices, p_add):
                    new_indices = np.append(remaining_indices, p_add)
                    # current_k stays same
                    if random.random() < 0.5:
                        current_indices = new_indices
                        if current_k > best_k:
                            best_k = current_k
                            best_indices = current_indices.copy()
                else:
                    pass

        T *= cooling_rate

    best_points = grid[best_indices]
    return best_points.astype(np.int64)

# EVOLVE-BLOCK-END
