# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large subset of the 8x8x8 grid such that no 5 points are co-spherical.
    Uses precomputed combination indices for fast vectorized determinant checks,
    enabling aggressive simulated annealing local search.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute combination indices for various set sizes
    # For a set of size m, we need all 4-combinations of range(m)
    # Precompute up to max expected size
    MAX_SIZE = 80
    combo_cache = {}
    for m in range(4, MAX_SIZE + 1):
        combos = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int32)
        combo_cache[m] = combos
    
    def can_add_vectorized(p_idx, S_array):
        """
        Vectorized check: can p_idx be added to S_array (numpy array of indices)?
        Uses precomputed combination indices for speed.
        """
        m = len(S_array)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = sq_norms[p_idx]
        
        # Get all points in S
        Q = grid[S_array]  # (m, 3)
        Q_sq = sq_norms[S_array]  # (m,)
        
        # Get precomputed combinations
        if m not in combo_cache:
            combos = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int32)
        else:
            combos = combo_cache[m]
        
        n_combos = len(combos)
        if n_combos == 0:
            return True
        
        # For each combination, compute the determinant
        # V = Q[combo] - p, shape (n_combos, 4, 3)
        # W = Q_sq[combo] - p_sq, shape (n_combos, 4)
        
        V = Q[combos] - p  # (n_combos, 4, 3)
        W = Q_sq[combos] - p_sq  # (n_combos, 4)
        
        # Form 4x4 matrices
        M = np.zeros((n_combos, 4, 4), dtype=np.float64)
        M[:, :, :3] = V
        M[:, :, 3] = W
        
        # Compute determinants in batches to avoid memory issues
        BATCH = 10000
        all_valid = True
        for start in range(0, n_combos, BATCH):
            end = min(start + BATCH, n_combos)
            dets = np.linalg.det(M[start:end])
            if np.any(np.abs(dets) < 0.5):
                all_valid = False
                break
        
        return all_valid
    
    def run_search(seed, time_limit=580):
        np.random.seed(seed)
        start_time = time.time()
        
        # Phase 1: Greedy
        order = np.random.permutation(N)
        
        current_set = []
        current_set_arr = np.array([], dtype=np.int64)
        current_set_set = set()
        
        for p_idx in order:
            if time.time() - start_time > time_limit:
                break
            if can_add_vectorized(p_idx, current_set_arr):
                current_set.append(p_idx)
                current_set_arr = np.append(current_set_arr, p_idx)
                current_set_set.add(p_idx)
        
        best_size = len(current_set)
        best_set = list(current_set)
        
        # Phase 2: Simulated annealing with multi-point moves
        temperature = 1.0
        cooling = 0.998
        min_temp = 0.005
        no_improve = 0
        patience = 500
        
        while temperature > min_temp:
            if time.time() - start_time > time_limit:
                break
            
            move_type = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            
            if move_type == 0:
                # Try a random addition
                p_idx = np.random.randint(N)
                if p_idx not in current_set_set:
                    if can_add_vectorized(p_idx, current_set_arr):
                        current_set.append(p_idx)
                        current_set_arr = np.append(current_set_arr, p_idx)
                        current_set_set.add(p_idx)
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                            no_improve = 0
                        else:
                            no_improve += 1
                        temperature *= cooling
                        continue
            
            elif move_type == 1:
                # Try a swap (remove 1, add 1)
                if len(current_set) > 0:
                    u_pos = np.random.randint(len(current_set))
                    u_idx = current_set[u_pos]
                    v_idx = np.random.randint(N)
                    
                    if v_idx not in current_set_set:
                        temp_arr = np.delete(current_set_arr, u_pos)
                        if can_add_vectorized(v_idx, temp_arr):
                            current_set[u_pos] = v_idx
                            current_set_arr = np.array(current_set, dtype=np.int64)
                            current_set_set.remove(u_idx)
                            current_set_set.add(v_idx)
                            no_improve += 1
                            temperature *= cooling
                            continue
            
            elif move_type == 2:
                # Try 1-for-2 (remove 1, add 2)
                if len(current_set) > 0:
                    u_pos = np.random.randint(len(current_set))
                    u_idx = current_set[u_pos]
                    
                    v1_idx = np.random.randint(N)
                    v2_idx = np.random.randint(N)
                    
                    if (v1_idx not in current_set_set and v2_idx not in current_set_set 
                            and v1_idx != v2_idx):
                        temp_arr = np.delete(current_set_arr, u_pos)
                        if can_add_vectorized(v1_idx, temp_arr):
                            temp_arr2 = np.append(temp_arr, v1_idx)
                            if can_add_vectorized(v2_idx, temp_arr2):
                                new_set = [x for i, x in enumerate(current_set) if i != u_pos]
                                new_set.extend([v1_idx, v2_idx])
                                current_set = new_set
                                current_set_arr = np.array(current_set, dtype=np.int64)
                                current_set_set = set(current_set)
                                if len(current_set) > best_size:
                                    best_size = len(current_set)
                                    best_set = list(current_set)
                                    no_improve = 0
                                else:
                                    no_improve += 1
                                temperature *= cooling
                                continue
            
            else:
                # Try 2-for-1 (remove 2, add 1)
                if len(current_set) > 2:
                    pos1 = np.random.randint(len(current_set))
                    pos2 = np.random.randint(len(current_set) - 1)
                    if pos2 >= pos1:
                        pos2 += 1
                    
                    u1 = current_set[pos1]
                    u2 = current_set[pos2]
                    v_idx = np.random.randint(N)
                    
                    if v_idx not in current_set_set:
                        mask = np.ones(len(current_set_arr), dtype=bool)
                        mask[pos1] = False
                        mask[pos2] = False
                        temp_arr = current_set_arr[mask]
                        
                        if can_add_vectorized(v_idx, temp_arr):
                            new_set = [x for i, x in enumerate(current_set) if i not in (pos1, pos2)]
                            new_set.append(v_idx)
                            current_set = new_set
                            current_set_arr = np.array(current_set, dtype=np.int64)
                            current_set_set = set(current_set)
                            no_improve += 1
                            temperature *= cooling
                            continue
            
            temperature *= cooling
            no_improve += 1
            
            # Reset if too long without improvement (restart-like)
            if no_improve > patience:
                no_improve = 0
                temperature = max(temperature, 0.5)
        
        return np.array(best_set, dtype=np.int64), best_size
    
    # Run multiple restarts
    best_overall = None
    best_size_overall = 0
    
    num_restarts = 6
    for i in range(num_restarts):
        res, size = run_search(seed=i, time_limit=580)
        if size > best_size_overall:
            best_size_overall = size
            best_overall = res
        
        if time.time() > 560:
            break
    
    return best_overall.astype(np.int64)

# EVOLVE-BLOCK-END
