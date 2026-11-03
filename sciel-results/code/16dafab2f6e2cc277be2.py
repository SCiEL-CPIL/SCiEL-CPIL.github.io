import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    # For radius 1, neighbors are self + 1-bit changes
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    # Vectorized generation of neighbors is tricky due to variable length logic in python loops,
    # but total_words is 2187, so a loop is fast enough.
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    def greedy_construct(random_seed, fix_first=False):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        counts = np.zeros(total_words, dtype=np.int16)
        code = []
        
        if fix_first:
            first_code = 0
            code.append(first_code)
            uncovered[neighbors_arr[first_code, :]] = False
            counts[neighbors_arr[first_code, :]] += 1
        
        while uncovered.any():
            gains_primary = np.sum(uncovered[neighbors_arr], axis=1)
            
            valid = gains_primary > 0
            if not valid.any():
                break
            
            max_gain = np.max(gains_primary[valid])
            best_candidates = np.where((gains_primary == max_gain) & valid)[0]
            
            if len(best_candidates) > 1:
                singly_covered = (counts == 1)
                redundancy_penalty = np.sum(singly_covered[neighbors_arr[best_candidates]], axis=1)
                scores = redundancy_penalty.astype(np.float64) + np.random.random(len(best_candidates)) * 0.5
                best_idx = np.argmin(scores)
                chosen = best_candidates[best_idx]
            else:
                chosen = best_candidates[0]
            
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
            counts[neighbors_arr[chosen, :]] += 1
        
        return code

    def reduce_code(code):
        if not code:
            return code
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        # Sort code by redundancy (highest first) to try removing "least essential" first
        # Redundancy: sum of counts of neighbors
        redundancies = np.sum(counts[neighbors_arr], axis=1)
        code_indices = np.array(code)
        # We want to sort code_indices by redundancies[code_indices] descending
        order = np.argsort(-redundancies[code_indices])
        code_sorted = code_indices[order].tolist()
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code_sorted:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code_sorted = new_code
            # Re-sort? The counts changed, so redundancies changed.
            # For simplicity, we just continue with the new list.
            # If we want to be thorough, we can re-sort, but it's expensive.
            # The current loop already iterates over all.
        
        return code_sorted

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            best_move = None
            best_score = -1
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    # Identify holes
                    holes = np.where(counts == 0)[0]
                    
                    if len(holes) > 8:
                        counts[neighbors_c] += 1
                        continue
                    
                    holes_mask = np.zeros(total_words, dtype=np.bool_)
                    holes_mask[holes] = True
                    
                    covered_counts = np.sum(holes_mask[neighbors_arr], axis=1)
                    
                    full_cover = covered_counts == len(holes)
                    full_cover[c_remove] = False
                    
                    if not np.any(full_cover):
                        counts[neighbors_c] += 1
                        continue
                    
                    candidate_indices = np.where(full_cover)[0]
                    
                    if len(candidate_indices) > 50:
                        candidate_indices = candidate_indices[np.random.choice(len(candidate_indices), 50, replace=False)]
                    
                    for cand in candidate_indices:
                        if cand in code:
                            continue
                        cand_neigh = neighbors_arr[cand, :]
                        redundant = np.sum(counts[cand_neigh] > 0)
                        # Score: prefer higher redundancy (less new coverage)
                        # Also prefer candidates that are "far" from each other?
                        if redundant > best_score:
                            best_score = redundant
                            best_move = (i, cand)
                    counts[neighbors_c] += 1
                else:
                    # Can remove
                    # Score: Removal is always best
                    best_score = 1000
                    best_move = (i, -1)
            
            if best_move is not None:
                i, cand = best_move
                if cand == -1:
                    code.pop(i)
                    counts[neighbors_arr[code[i] if i < len(code) else code[-1], :]] # This is tricky to update counts correctly after pop
                    # Let's just rebuild counts or update carefully.
                    # Easier: remove, then rebuild counts if needed? No, too slow.
                    # Let's do the update carefully.
                    c_remove = code[i]
                    counts[neighbors_arr[c_remove, :]] -= 1
                    code.pop(i)
                else:
                    c_remove = code[i]
                    # Remove c_remove
                    counts[neighbors_arr[c_remove, :]] -= 1
                    # Add cand
                    code[i] = cand
                    counts[neighbors_arr[cand, :]] += 1
                improved = True
        
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        # Repair
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        uncovered = counts == 0
        
        while uncovered.any() and time.time() < deadline:
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            best_candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            counts[neighbors_arr[chosen, :]] += 1
            uncovered = counts == 0
            
        return code

    def multi_perturb(code, deadline, num_perturb=3):
        best = list(code)
        best_size = len(code)
        
        for _ in range(num_perturb):
            if time.time() >= deadline:
                break
            k = random.randint(1, min(4, len(code) - 1))
            perturbed = perturb_repair(code, k, deadline - 5)
            if len(perturbed) < best_size:
                best = perturbed
                best_size = len(perturbed)
        
        return best

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 30:
        # Alternate between standard and fixed-first greedy
        fix_first = (seed % 2 == 0)
        code = greedy_construct(seed, fix_first)
        code = reduce_code(code)
        
        if time.time() < deadline - 10:
            code = local_search(code, deadline - 5)
            code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if time.time() < deadline - 20 and best_code is not None:
            perturbed = multi_perturb(best_code, deadline - 10, num_perturb=2)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 8:
                perturbed = local_search(perturbed, deadline - 5)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
    
    if best_code is not None and time.time() < deadline - 5:
        polished = local_search(best_code, deadline - 2)
        polished = reduce_code(polished)
        if len(polished) < best_size:
            best_code = polished
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
