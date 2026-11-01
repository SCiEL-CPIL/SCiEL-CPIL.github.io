"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precompute neighbors using bitmasks for speed
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 2187
    
    neighbors_mask = [0] * N
    neighbors_list = [[] for _ in range(N)]
    
    for w_idx, w in enumerate(all_words):
        mask = 1 << w_idx
        lst = [w_idx]
        w_list = list(w)
        for i in range(7):
            orig = w_list[i]
            for v in range(3):
                if v != orig:
                    w_list[i] = v
                    nb_idx = word_to_idx[tuple(w_list)]
                    mask |= (1 << nb_idx)
                    lst.append(nb_idx)
            w_list[i] = orig
        neighbors_mask[w_idx] = mask
        neighbors_list[w_idx] = lst

    # Greedy initialization
    def greedy_init(seed):
        random.seed(seed)
        counts = [0] * N
        covered_mask = 0
        code = []
        code_set = set()
        
        while covered_mask != (1 << N) - 1:
            best_idx = -1
            max_gain = 0
            # Sample candidates to speed up greedy if needed, but for N=2187 full scan is okay
            for w_idx in range(N):
                if w_idx in code_set:
                    continue
                # Gain is number of bits in neighbors_mask[w_idx] NOT in covered_mask
                gain = (neighbors_mask[w_idx] & ~covered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = w_idx
            if best_idx == -1:
                break
            code.append(all_words[best_idx])
            code_set.add(best_idx)
            # Update covered mask
            covered_mask |= neighbors_mask[best_idx]
            # Update counts
            for nb in neighbors_list[best_idx]:
                counts[nb] += 1
        
        covered = covered_mask.bit_count()
        return code, counts, covered, code_set, covered_mask
    
    # Simulated annealing on a single start
    def simulated_annealing(init_code, init_counts, init_covered, init_code_set, init_covered_mask, start_time, time_limit, seed):
        random.seed(seed)
        code = list(init_code)
        counts = list(init_counts)
        covered = init_covered
        size = len(code)
        code_set = set(init_code_set)
        covered_mask = init_covered_mask
        
        P = 10.0
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = 5.0
        T_min = 0.001
        cooling = 0.9999
        
        all_indices = list(range(N))
        
        # Periodic greedy reduction
        last_reduction = time.time()
        reduction_interval = 1.0
        
        while time.time() - start_time < time_limit:
            # Periodic greedy reduction: try to remove redundant codewords
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                # Try removing each codeword in random order
                indices = list(range(size))
                random.shuffle(indices)
                for idx in indices:
                    if idx >= size:
                        continue
                    w = code[idx]
                    w_idx = word_to_idx[w]
                    # Check if removing this word keeps coverage
                    # A word is redundant if all its neighbors are covered by others
                    # i.e., counts[nb] > 1 for all nb in neighbors[w_idx]
                    is_redundant = True
                    for nb in neighbors_list[w_idx]:
                        if counts[nb] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        # Safe to remove
                        code.pop(idx)
                        code_set.remove(w_idx)
                        for nb in neighbors_list[w_idx]:
                            counts[nb] -= 1
                        size -= 1
                        # Update covered mask? 
                        # Actually, if it was redundant, covered_mask doesn't change because
                        # every neighbor was covered by at least one other word.
                        # Wait, counts[nb] was 1, so after removal it becomes 0? 
                        # No, if counts[nb] == 1, it means ONLY this word covered it.
                        # So if I remove it, that neighbor becomes UNCOVERED.
                        # My check `is_redundant` above was WRONG.
                        # Redundant means: for all nb in neighbors[w_idx], counts[nb] > 1.
                        # Let's re-verify logic.
                        # If counts[nb] == 1, removing w makes counts[nb] 0 -> uncovered.
                        # So we can ONLY remove if counts[nb] > 1 for ALL nb.
                        pass 
                # The logic above was flawed in the draft. Let's implement it correctly in the loop.
                # We'll do a proper scan.
                indices = list(range(size))
                random.shuffle(indices)
                for idx in indices:
                    if idx >= size:
                        continue
                    w = code[idx]
                    w_idx = word_to_idx[w]
                    can_remove = True
                    for nb in neighbors_list[w_idx]:
                        if counts[nb] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        code.pop(idx)
                        code_set.remove(w_idx)
                        for nb in neighbors_list[w_idx]:
                            counts[nb] -= 1
                        size -= 1
                        # covered_mask remains same because no neighbor became uncovered
                        current_cost = cost(size, covered)
                        if size < best_size:
                            best_size = size
                            best_code = list(code)
                            best_covered = covered
                            best_cost = current_cost
                        break  # Restart scan after a removal

            move_type = random.random()
            
            if move_type < 0.60:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                w_remove = code[idx_remove]
                idx_w_remove = word_to_idx[w_remove]
                
                # Select candidate for add
                # Choose from top K candidates by gain
                # Gain = (neighbors_mask[idx_add] & ~covered_mask).bit_count()
                # But we also need to consider loss from removing idx_w_remove
                # Loss = (neighbors_mask[idx_w_remove] & ~neighbors_mask[idx_add] & ~covered_mask_after_add).bit_count()
                # This is complex. Simpler: just pick random candidates and check.
                
                # To be faster, pick 10 random candidates and choose the one with best net delta
                best_idx_add = -1
                best_delta = -100
                for _ in range(10):
                    idx_add = random.choice(all_indices)
                    if idx_add == idx_w_remove or idx_add in code_set:
                        continue
                    
                    # Calculate delta
                    # New covered = covered + (gain from add) - (loss from remove)
                    # Gain from add: neighbors in idx_add not currently covered
                    # But if idx_remove covers some of those, they are already covered.
                    # Actually, let's use the counts array.
                    
                    # Words that become uncovered:
                    # In neighbors[idx_w_remove] where count == 1 AND not in neighbors[idx_add]
                    # Words that become covered:
                    # In neighbors[idx_add] where count == 0
                    
                    nbs_remove = neighbors_list[idx_w_remove]
                    nbs_add = neighbors_list[idx_add]
                    
                    # Create a set of nbs_add for O(1) lookup? 
                    # Or just loop. N=15 neighbors.
                    nbs_add_set = set(nbs_add)
                    
                    losing = 0
                    for nb in nbs_remove:
                        if counts[nb] == 1 and nb not in nbs_add_set:
                            losing += 1
                            
                    gaining = 0
                    for nb in nbs_add:
                        if counts[nb] == 0:
                            gaining += 1
                            
                    delta = gaining - losing
                    if delta > best_delta:
                        best_delta = delta
                        best_idx_add = idx_add
                        
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_remove = neighbors_list[idx_w_remove]
                nbs_add = neighbors_list[best_idx_add]
                nbs_add_set = set(nbs_add)
                
                losing = 0
                for nb in nbs_remove:
                    if counts[nb] == 1 and nb not in nbs_add_set:
                        losing += 1
                        
                gaining = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        gaining += 1
                        
                delta = gaining - losing
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code[idx_remove] = all_words[best_idx_add]
                    code_set.remove(idx_w_remove)
                    code_set.add(best_idx_add)
                    for nb in nbs_remove: counts[nb] -= 1
                    for nb in nbs_add: counts[nb] += 1
                    covered = new_covered
                    current_cost = new_cost
                    
                    # Update covered_mask
                    # We can update it incrementally or recompute if needed.
                    # For cost calculation, we use `covered` count.
                    # For greedy reduction, we use `counts`.
                    # We don't strictly need covered_mask for the SA moves if we track `covered` count.
                    # But for "Add" moves, we might want to check gain quickly.
                    # Let's update covered_mask only when we need it or periodically.
                    # Actually, let's just update it.
                    # Bits that were 0 and are now 1:
                    # Bits that were 1 and are now 0:
                    # This is expensive. Let's stick to `covered` count for cost.
                    # For "Add" move gain calculation, we use counts[nb]==0.
                    
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
                
                delta_covered = 0
                for nb in neighbors_list[w_idx]:
                    if counts[nb] == 1:
                        delta_covered -= 1
                
                new_size = size - 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.pop(idx)
                    code_set.remove(w_idx)
                    for nb in neighbors_list[w_idx]:
                        counts[nb] -= 1
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
                # Pick word with best marginal gain from a few candidates
                best_idx_add = -1
                best_gain = -1
                for _ in range(10):
                    idx_add = random.choice(all_indices)
                    if idx_add in code_set:
                        continue
                    gain = 0
                    for nb in neighbors_list[idx_add]:
                        if counts[nb] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx_add = idx_add
                
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_add = neighbors_list[best_idx_add]
                delta_covered = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        delta_covered += 1
                        
                new_size = size + 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.append(all_words[best_idx_add])
                    code_set.add(best_idx_add)
                    for nb in nbs_add:
                        counts[nb] += 1
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
    
    # Multi-restart: try a few different seeds for greedy init
    best_overall = None
    best_overall_size = float('inf')
    
    num_restarts = 4
    time_per_restart = time_limit / num_restarts
    
    for r in range(num_restarts):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 5:
            break
        init_code, init_counts, init_covered, init_code_set, _ = greedy_init(42 + r)
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 2:
            break
        best_code, best_size = simulated_annealing(
            init_code, init_counts, init_covered, init_code_set, 0,
            start_time, time_limit, 100 + r
        )
        if best_size < best_overall_size:
            best_overall_size = best_size
            best_overall = best_code
    
    if best_overall is None:
        # Fallback
        init_code, _, _, _, _ = greedy_init(42)
        best_overall = init_code
    
    return [list(w) for w in best_overall]
# EVOLVE-BLOCK-END
