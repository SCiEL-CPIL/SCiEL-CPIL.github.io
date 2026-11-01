"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    N = 2187
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as a NumPy array for vectorized operations
    # neighbors[i] contains the indices of words within Hamming distance 1 of all_words[i]
    neighbors_arr = np.zeros((N, 15), dtype=np.int32)
    for i, w in enumerate(all_words):
        w_list = list(w)
        idxs = [word_to_idx[w]]
        for pos in range(7):
            orig = w_list[pos]
            for v in range(3):
                if v != orig:
                    w_list[pos] = v
                    idxs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        neighbors_arr[i] = idxs

    # Greedy initialization using NumPy for speed
    def greedy_init(seed):
        random.seed(seed)
        counts = np.zeros(N, dtype=np.int32)
        code = []
        code_set = set()
        
        # We iterate until all covered. 
        # To speed up, we can batch the gain calculation.
        # However, for simplicity and robustness, we'll do a fast loop.
        # The bottleneck is finding the max gain. 
        # Since N=2187 is small, a Python loop over N is ~2k ops, which is fast.
        # The inner loop over 15 neighbors is also fast.
        
        while True:
            # Check if all covered
            if np.all(counts > 0):
                break
            
            best_idx = -1
            max_gain = 0
            
            # Optimization: Only check indices not in code_set?
            # Iterating all N is 2187 * 15 ~ 32k ops per step.
            # 186 steps * 32k ~ 6M ops. This is acceptable in Python if optimized.
            # But let's try to vectorize the gain calculation for all candidates at once?
            # Gain for candidate j is sum(counts[neighbors[j]] == 0).
            # We can compute gains for all j in one go using matrix multiplication?
            # Let M be N x 15 matrix of neighbors.
            # We need sum over k of (counts[M[j,k]] == 0).
            # This is not a simple matrix product.
            # Let's stick to a tight Python loop but maybe unroll or use lists.
            
            # Actually, the previous Python loop was the bottleneck.
            # Let's try a vectorized approach for the greedy step.
            # counts is shape (N,). neighbors_arr is (N, 15).
            # We want gain[j] = np.sum(counts[neighbors_arr[j]] == 0)
            
            # We can do this:
            # nb_counts = counts[neighbors_arr] -> shape (N, 15)
            # zero_mask = (nb_counts == 0) -> shape (N, 15)
            # gains = np.sum(zero_mask, axis=1) -> shape (N,)
            
            nb_counts = counts[neighbors_arr]
            gains = np.sum(nb_counts == 0, axis=1)
            
            # Exclude already chosen words
            for idx in code_set:
                gains[idx] = -1
            
            best_idx = np.argmax(gains)
            max_gain = gains[best_idx]
            
            if max_gain <= 0:
                break # Should not happen if not all covered, but safety
                
            code.append(all_words[best_idx])
            code_set.add(best_idx)
            counts[neighbors_arr[best_idx]] += 1
        
        return code, counts, len(code), code_set

    # Simulated Annealing
    def simulated_annealing(init_code, init_counts, init_size, init_code_set, start_time, time_limit, seed):
        random.seed(seed)
        code = list(init_code)
        counts = init_counts.copy()
        size = init_size
        code_set = set(init_code_set)
        covered = N - np.sum(counts == 0)
        
        P = 10.0
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9999
        
        all_indices = list(range(N))
        
        # Precompute neighbor sets as Python sets for fast membership? 
        # Or just use the array.
        # For 'losing' calculation in swap:
        # losing = sum(1 for nb in nbs_remove if counts[nb]==1 and nb not in nbs_add_set)
        # Using sets is faster for 'not in'.
        neighbor_sets = [set(neighbors_arr[i].tolist()) for i in range(N)]
        
        last_reduction = time.time()
        reduction_interval = 1.0
        
        while time.time() - start_time < time_limit:
            # Periodic greedy reduction
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                # Try to remove redundant words
                indices = list(range(size))
                random.shuffle(indices)
                for idx in indices:
                    if idx >= size:
                        continue
                    w = code[idx]
                    w_idx = word_to_idx[w]
                    # Check if any neighbor has count 1
                    if np.any(counts[neighbors_arr[w_idx]] == 1):
                        continue
                    # Safe to remove
                    code.pop(idx)
                    code_set.remove(w_idx)
                    counts[neighbors_arr[w_idx]] -= 1
                    size -= 1
                    covered = N - np.sum(counts == 0)
                    current_cost = cost(size, covered)
                    if size < best_size:
                        best_size = size
                        best_code = list(code)
                        best_covered = covered
                        best_cost = current_cost
                    break # Restart scan
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                w_remove = code[idx_remove]
                idx_w_remove = word_to_idx[w_remove]
                idx_add = random.choice(all_indices)
                
                if idx_add == idx_w_remove or idx_add in code_set:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_remove = neighbors_arr[idx_w_remove]
                nbs_add = neighbors_arr[idx_add]
                nbs_add_set = neighbor_sets[idx_add]
                
                # Calculate losing
                # Counts of nbs_remove
                c_remove = counts[nbs_remove]
                # We lose coverage if count was 1 and the word is NOT in nbs_add
                # mask_remove = (c_remove == 1)
                # mask_not_in_add = np.array([nb not in nbs_add_set for nb in nbs_remove])
                # losing = np.sum(mask_remove & mask_not_in_add)
                
                # Faster: iterate 15 elements
                losing = 0
                for nb in nbs_remove:
                    if counts[nb] == 1 and nb not in nbs_add_set:
                        losing += 1
                
                # Calculate gaining
                # Gain coverage if count was 0
                c_add = counts[nbs_add]
                gaining = np.sum(c_add == 0)
                
                delta = gaining - losing
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code[idx_remove] = all_words[idx_add]
                    code_set.remove(idx_w_remove)
                    code_set.add(idx_add)
                    counts[nbs_remove] -= 1
                    counts[nbs_add] += 1
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            elif move_type < 0.85:
                # Remove
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx = random.randint(0, size - 1)
                w = code[idx]
                w_idx = word_to_idx[w]
                
                # Losing coverage = number of neighbors with count 1
                losing = np.sum(counts[neighbors_arr[w_idx]] == 1)
                
                new_size = size - 1
                new_covered = covered - losing
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.pop(idx)
                    code_set.remove(w_idx)
                    counts[neighbors_arr[w_idx]] -= 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            else:
                # Add
                if covered == N:
                    # Convert to swap
                    if size > 0:
                        idx_remove = random.randint(0, size - 1)
                        idx_w_remove = word_to_idx[code[idx_remove]]
                        idx_add = random.choice(all_indices)
                        if idx_add == idx_w_remove or idx_add in code_set:
                            T *= cooling
                            if T < T_min: T = T_min
                            continue
                        nbs_remove = neighbors_arr[idx_w_remove]
                        nbs_add = neighbors_arr[idx_add]
                        nbs_add_set = neighbor_sets[idx_add]
                        
                        losing = 0
                        for nb in nbs_remove:
                            if counts[nb] == 1 and nb not in nbs_add_set:
                                losing += 1
                        gaining = np.sum(counts[nbs_add] == 0)
                        
                        delta = gaining - losing
                        new_covered = covered + delta
                        new_cost = cost(size, new_covered)
                        delta_cost = new_cost - current_cost
                        
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            code[idx_remove] = all_words[idx_add]
                            code_set.remove(idx_w_remove)
                            code_set.add(idx_add)
                            counts[nbs_remove] -= 1
                            counts[nbs_add] += 1
                            covered = new_covered
                            current_cost = new_cost
                            
                            if covered == N and size < best_size:
                                best_size = size
                                best_covered = covered
                                best_code = list(code)
                                best_cost = current_cost
                        T *= cooling
                        if T < T_min: T = T_min
                    continue
                
                # Pick word with best marginal gain
                # Vectorized gain calculation for all candidates
                nb_counts = counts[neighbors_arr]
                gains = np.sum(nb_counts == 0, axis=1)
                
                for idx in code_set:
                    gains[idx] = -1
                
                best_idx_add = np.argmax(gains)
                best_gain = gains[best_idx_add]
                
                if best_gain <= 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_add = neighbors_arr[best_idx_add]
                delta_covered = best_gain
                new_size = size + 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.append(all_words[best_idx_add])
                    code_set.add(best_idx_add)
                    counts[nbs_add] += 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
        
        return best_code, best_size
    
    start_time = time.time()
    time_limit = 115.0
    
    best_overall = None
    best_overall_size = float('inf')
    
    num_restarts = 5
    time_per_restart = time_limit / num_restarts
    
    for r in range(num_restarts):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 5:
            break
        init_code, init_counts, init_size, init_code_set = greedy_init(42 + r)
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 2:
            break
        best_code, best_size = simulated_annealing(
            init_code, init_counts, init_size, init_code_set,
            start_time, time_limit, 100 + r
        )
        if best_size < best_overall_size:
            best_overall_size = best_size
            best_overall = best_code
    
    if best_overall is None:
        init_code, _, _, _ = greedy_init(42)
        best_overall = init_code
    
    return [list(w) for w in best_overall]
# EVOLVE-BLOCK-END
