import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
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

    # Precompute neighbors using vectorized approach
    # For each word, its neighbors are itself plus all words at Hamming distance 1
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    # Self
    neighbors_arr[:, 0] = np.arange(total_words)
    
    # For each position and each alternative value
    col = 1
    for pos in range(n):
        for new_val in range(q):
            # Get the value at this position for all words
            vals = (np.arange(total_words) // (q ** (n - 1 - pos))) % q
            # For words where this position is NOT new_val, compute the neighbor
            mask = vals != new_val
            if mask.any():
                idxs = np.where(mask)[0]
                # Compute neighbor index: replace position pos with new_val
                # The contribution of position pos is (val - new_val) * q^(n-1-pos)
                diff = (vals[idxs] - new_val) * (q ** (n - 1 - pos))
                neighbors_arr[idxs, col] = idxs - diff
            col += 1
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = np.max(gains[valid])
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code(code):
        if not code:
            return code
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    # Find candidates that cover all holes
                    # candidate j covers hole h iff j is in neighbors_arr[h]
                    cover_count_per_candidate = np.zeros(total_words, dtype=np.int16)
                    for h in holes:
                        cover_count_per_candidate[neighbors_arr[h, :]] += 1
                    
                    all_holes = len(holes)
                    candidates = np.where(cover_count_per_candidate == all_holes)[0]
                    candidates = candidates[candidates != c_remove]
                    
                    if len(candidates) == 0:
                        counts[neighbors_c] += 1
                        continue
                    
                    cand_list = candidates.tolist()
                    if len(cand_list) > 200:
                        cand_list = random.sample(cand_list, 200)
                    
                    cand_indices = np.array(cand_list, dtype=np.int32)
                    cand_neighbors = neighbors_arr[cand_indices, :]
                    
                    # Score: minimize overlap (sum of counts over candidate's neighborhood)
                    scores = np.sum(counts[cand_neighbors], axis=1)
                    
                    min_score = np.min(scores)
                    best_cands = np.where(scores == min_score)[0]
                    
                    if len(best_cands) > 1 and random.random() < 0.2:
                        # Soft-min: pick from top candidates with some randomness
                        threshold = min_score + max(1, int(0.1 * np.std(scores)))
                        top_cands = np.where(scores <= threshold)[0]
                        chosen_idx = random.choice(top_cands.tolist())
                    else:
                        chosen_idx = random.choice(best_cands.tolist())
                    
                    best_cand = cand_list[chosen_idx]
                    
                    code[i] = best_cand
                    counts[neighbors_arr[best_cand, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        code_arr = np.array(code, dtype=np.int32)
        all_neighbors = neighbors_arr[code_arr, :]
        scores = np.sum(counts[all_neighbors], axis=1)
        
        # Remove codewords with highest overlap
        num_to_consider = min(len(code), k * 3)
        top_indices = np.argsort(scores)[-num_to_consider:]
        
        top_indices_list = top_indices.tolist()
        random.shuffle(top_indices_list)
        remove_indices = top_indices_list[:k]
        
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
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

    def two_one_swap(code, deadline):
        """Try to replace 2 codewords with 1 to reduce size by 1."""
        code = list(code)
        if len(code) < 2:
            return code
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try random pairs
            num_attempts = min(50, len(code) * (len(code) - 1) // 2)
            pairs = set()
            for _ in range(num_attempts):
                i, j = random.sample(range(len(code)), 2)
                if (i, j) in pairs or (j, i) in pairs:
                    continue
                pairs.add((i, j))
                
                c1, c2 = code[i], code[j]
                neighbors_1 = neighbors_arr[c1, :]
                neighbors_2 = neighbors_arr[c2, :]
                
                # Temporarily remove both
                counts[neighbors_1] -= 1
                counts[neighbors_2] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    # Find a single candidate that covers all holes
                    cover_count_per_candidate = np.zeros(total_words, dtype=np.int16)
                    for h in holes:
                        cover_count_per_candidate[neighbors_arr[h, :]] += 1
                    
                    all_holes = len(holes)
                    candidates = np.where(cover_count_per_candidate == all_holes)[0]
                    candidates = candidates[(candidates != c1) & (candidates != c2)]
                    
                    if len(candidates) > 0:
                        # Pick candidate with minimum overlap
                        cand_indices = np.array(candidates.tolist(), dtype=np.int32)
                        if len(cand_indices) > 100:
                            cand_indices = cand_indices[np.random.choice(len(cand_indices), 100, replace=False)]
                        cand_neighbors = neighbors_arr[cand_indices, :]
                        scores = np.sum(counts[cand_neighbors], axis=1)
                        min_idx = np.argmin(scores)
                        best_cand = cand_indices[min_idx]
                        
                        # Replace: remove i and j, add best_cand
                        # Remove the larger index first
                        if i > j:
                            i, j = j, i
                        code.pop(j)
                        code.pop(i)
                        code.append(best_cand)
                        
                        # Update counts
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        break
                    else:
                        counts[neighbors_1] += 1
                        counts[neighbors_2] += 1
                else:
                    # Both can be removed, but we need to add one back to maintain coverage
                    # Actually if no holes, we can just remove both and add one that covers well
                    # But we want to reduce by 1, so remove both and add one
                    # Find candidate that covers the union of their neighborhoods best
                    union_mask = np.zeros(total_words, dtype=np.bool_)
                    union_mask[neighbors_1] = True
                    union_mask[neighbors_2] = True
                    
                    # We need a candidate whose neighborhood covers all holes (none) 
                    # but we still need to maintain coverage. Since no holes, any candidate works.
                    # Pick one with minimum overlap
                    cand_indices = np.array(list(range(total_words)), dtype=np.int32)
                    # This is expensive, so just pick randomly from the code
                    best_cand = random.choice(code)
                    
                    if i > j:
                        i, j = j, i
                    code.pop(j)
                    code.pop(i)
                    code.append(best_cand)
                    
                    counts[neighbors_arr[best_cand, :]] += 1
                    improved = True
                    break
                continue
                
                # Reset counts if we didn't break
                counts[neighbors_1] += 1
                counts[neighbors_2] += 1
        
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
        
        if time.time() < deadline - 10 and len(code) > 2:
            code = two_one_swap(code, deadline - 5)
            code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if time.time() < deadline - 20 and best_code is not None:
            k = random.randint(2, 6)
            perturbed = perturb_repair(best_code, k, deadline - 15)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 8:
                perturbed = local_search(perturbed, deadline - 5)
                perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 5 and len(perturbed) > 2:
                perturbed = two_one_swap(perturbed, deadline - 3)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
