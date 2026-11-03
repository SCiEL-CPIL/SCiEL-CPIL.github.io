import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
    # Precompute neighbors
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        idxs = [i]
        for pos in range(n):
            original_val = (i // (q ** (n - 1 - pos))) % q
            for new_val in range(q):
                if new_val == original_val:
                    continue
                j = i + (new_val - original_val) * (q ** (n - 1 - pos))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx
    
    # Build inverse neighbors
    inv_neighbor_lists = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in range(cover_count):
            neighbor_idx = neighbors_arr[i, j]
            inv_neighbor_lists[neighbor_idx].append(i)
    
    def idx_to_list(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return list(reversed(t))
    
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
    
    def local_search(code, dl):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < dl:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= dl:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    candidates = set(inv_neighbor_lists[holes[0]])
                    for h in holes[1:]:
                        if not candidates:
                            break
                        candidates &= set(inv_neighbor_lists[h])
                    
                    candidates.discard(c_remove)
                    
                    if not candidates:
                        counts[neighbors_c] += 1
                        continue
                    
                    best_cand = None
                    best_score = -1
                    cand_list = list(candidates)
                    
                    if len(cand_list) > 30:
                        cand_list = random.sample(cand_list, 30)
                        
                    for cand in cand_list:
                        cand_neigh = neighbors_arr[cand, :]
                        score = np.sum(counts[cand_neigh])
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                            
                    if best_cand is not None:
                        code[i] = best_cand
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        break
                    else:
                        counts[neighbors_c] += 1
                else:
                    code.pop(i)
                    improved = True
                    break
        return code
    
    def perturb_repair(code, k, dl):
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        uncovered = counts == 0
        
        while uncovered.any() and time.time() < dl:
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
    
    def two_opt_swap(code, dl):
        code = list(code)
        if len(code) < 2:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < dl:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= dl:
                    break
                if i >= len(code):
                    continue
                    
                for j in range(i + 1, len(code)):
                    if time.time() >= dl:
                        break
                    
                    c1 = code[i]
                    c2 = code[j]
                    
                    counts[neighbors_arr[c1, :]] -= 1
                    counts[neighbors_arr[c2, :]] -= 1
                    
                    if np.any(counts < 0):
                        counts[neighbors_arr[c1, :]] += 1
                        counts[neighbors_arr[c2, :]] += 1
                        continue
                    
                    if np.any(counts == 0):
                        holes = np.where(counts == 0)[0]
                        
                        candidates = set(inv_neighbor_lists[holes[0]])
                        for h in holes[1:]:
                            if not candidates:
                                break
                            candidates &= set(inv_neighbor_lists[h])
                        
                        candidates.discard(c1)
                        candidates.discard(c2)
                        
                        if not candidates:
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                            continue
                        
                        cand_list = list(candidates)
                        if len(cand_list) > 15:
                            cand_list = random.sample(cand_list, 15)
                        
                        best_cand = None
                        best_score = -1
                        for cand in cand_list:
                            cand_neigh = neighbors_arr[cand, :]
                            score = np.sum(counts[cand_neigh])
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                        
                        if best_cand is not None:
                            code.pop(j)
                            code[i] = best_cand
                            counts[neighbors_arr[best_cand, :]] += 1
                            improved = True
                            break
                        else:
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                    else:
                        code.pop(j)
                        code.pop(i)
                        improved = True
                        break
        return code
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
            
        if time.time() < deadline - 10:
            code = two_opt_swap(code, deadline - 5)
            code = reduce_code(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if time.time() < deadline - 20 and best_code is not None:
            k = random.randint(3, 8)
            perturbed = perturb_repair(best_code, k, deadline - 15)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 10:
                perturbed = local_search(perturbed, deadline - 5)
                perturbed = reduce_code(perturbed)
                
            if time.time() < deadline - 5:
                perturbed = two_opt_swap(perturbed, deadline - 2)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
    
    result = []
    for idx in best_code:
        result.append(idx_to_list(idx))
    
    return result
