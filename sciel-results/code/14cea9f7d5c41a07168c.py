import time
import random
import numpy as np
from itertools import product

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

    # Precompute neighbors as numpy arrays for vectorized operations
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
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

    # Precompute reverse mapping: for each word, which codewords cover it
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_cover_arr[w, reverse_cover_counts[w]] = i
            reverse_cover_counts[w] += 1

    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            uncovered_expanded = uncovered[neighbors_arr]
            gains = uncovered_expanded.sum(axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code_fast(code):
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

    def local_search_improve(code, deadline):
        code = list(code)
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
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1
                    
                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    # Find replacement using reverse_cover
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    best_w = random.choice(list(candidate_set))
                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break
        
        return code

    def local_search_swap_improve(code, deadline):
        """
        Enhanced local search that tries swaps to potentially improve the code structure.
        """
        code = list(code)
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
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1
                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    best_w = random.choice(list(candidate_set))
                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break
        
        return code

    def local_search_21_move(code, deadline):
        """
        Try 2-for-1 moves: remove two codewords, add one that covers all their unique contributions.
        """
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            if len(code) < 2:
                break
            
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                for j in indices:
                    if time.time() >= deadline:
                        break
                    if j <= i:
                        continue
                    
                    c1 = code[i]
                    c2 = code[j]
                    
                    # Temporarily remove both
                    counts[neighbors_arr[c1, :]] -= 1
                    counts[neighbors_arr[c2, :]] -= 1
                    
                    if np.any(counts == 0):
                        # Find words that would be uncovered
                        uncovered_words = np.where(counts == 0)[0]
                        
                        # Find candidates that cover all uncovered words
                        candidate_set = None
                        for uw in uncovered_words:
                            rc = reverse_cover_arr[uw, :reverse_cover_counts[uw]]
                            if candidate_set is None:
                                candidate_set = set(rc.tolist())
                            else:
                                candidate_set &= set(rc.tolist())
                        
                        if candidate_set is None or len(candidate_set) == 0:
                            # Restore and continue
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                            continue
                        
                        # Exclude c1 and c2
                        candidate_set.discard(c1)
                        candidate_set.discard(c2)
                        
                        if len(candidate_set) == 0:
                            counts[neighbors_arr[c1, :]] += 1
                            counts[neighbors_arr[c2, :]] += 1
                            continue
                        
                        # Pick a random candidate
                        new_c = random.choice(list(candidate_set))
                        
                        # Apply the move: remove i and j, add new_c
                        # Remove the larger index first to maintain index consistency
                        if j > i:
                            code.pop(j)
                            code.pop(i)
                        else:
                            code.pop(i)
                            code.pop(j)
                        code.append(new_c)
                        
                        counts[neighbors_arr[new_c, :]] += 1
                        improved = True
                        break
                    else:
                        # Both can be removed without adding
                        # Remove the larger index first
                        if j > i:
                            code.pop(j)
                            code.pop(i)
                        else:
                            code.pop(i)
                            code.pop(j)
                        improved = True
                        break
                
                if improved:
                    break
        
        return code

    def perturbation_repair(code, deadline):
        """
        Perturb the code by removing some random codewords, then repair by greedy addition.
        """
        code = list(code)
        if len(code) < 5:
            return code
        
        # Remove a random subset of codewords
        num_remove = max(1, len(code) // 10)
        to_remove = random.sample(range(len(code)), min(num_remove, len(code)))
        to_remove.sort(reverse=True)
        for idx in to_remove:
            code.pop(idx)
        
        # Repair: add codewords until all words are covered
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        while np.any(counts == 0) and time.time() < deadline:
            # Find uncovered words
            uncovered = np.where(counts == 0)[0]
            
            # For each candidate, compute how many uncovered words it covers
            gains = np.zeros(total_words, dtype=np.int32)
            for uw in uncovered:
                rc = reverse_cover_arr[uw, :reverse_cover_counts[uw]]
                gains[rc] += 1
            
            # Only consider candidates not already in code
            code_set = set(code)
            valid_mask = np.ones(total_words, dtype=np.bool_)
            for c in code_set:
                valid_mask[c] = False
            
            gains[~valid_mask] = 0
            
            if gains.max() == 0:
                break
            
            best_idx = np.argmax(gains)
            code.append(best_idx)
            counts[neighbors_arr[best_idx, :]] += 1
        
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # Greedy construction
        code = greedy_construct_vectorized(seed)
        
        # Reduction pass
        code = reduce_code_fast(code)
        
        # Local search improvement
        if time.time() < deadline - 20:
            code = local_search_improve(code, deadline - 15)
            code = reduce_code_fast(code)
        
        # 2-1 move local search
        if time.time() < deadline - 10:
            code = local_search_21_move(code, deadline - 8)
            code = reduce_code_fast(code)
        
        # Additional local search for further improvement
        if time.time() < deadline - 5:
            code = local_search_swap_improve(code, deadline - 3)
            code = reduce_code_fast(code)
        
        # Perturbation and repair
        if time.time() < deadline - 2:
            code = perturbation_repair(code, deadline - 1)
            code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        
        seed += 1
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
