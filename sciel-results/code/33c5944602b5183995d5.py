import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115
    
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t
    
    def tuple_to_idx(t):
        idx = 0
        for v in t:
            idx = idx * q + v
        return idx
    
    # Precompute neighbors
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                j = tuple_to_idx(new_w)
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx
    
    # Precompute reverse cover: for each word, which codewords cover it
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_cover_arr[w, reverse_cover_counts[w]] = i
            reverse_cover_counts[w] += 1
    
    def greedy_construct_vectorized(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            gains = uncovered[neighbors_arr].sum(axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = rng.choice(best_candidates)
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code
    
    def reduce_code_fast(code):
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
    
    def local_search_improve(code, deadline):
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
            
            # Try 1-0 moves (remove) and 1-1 moves (swap)
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1
                    
                    # Find critical words (covered only by c_remove)
                    crit_mask = (counts[neighbors_arr[c_remove, :]] == 1)
                    critical_words = neighbors_arr[c_remove, :][crit_mask]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    # Find candidates that cover all critical words using reverse cover
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        rc_list = rc.tolist()
                        if candidate_set is None:
                            candidate_set = set(rc_list)
                        else:
                            candidate_set &= set(rc_list)
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    # Pick a candidate that maximizes additional coverage
                    best_cand = None
                    best_gain = -1
                    for cand in list(candidate_set)[:20]:  # Limit for speed
                        gain = counts[neighbors_arr[cand, :]].sum()
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = cand
                    
                    if best_cand is not None:
                        code[i] = best_cand
                        counts[neighbors_arr[c_remove, :]] -= 1
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        break
                else:
                    code.pop(i)
                    improved = True
                    break
            
            # Try 2-1 moves: remove 2, add 1
            if time.time() < deadline and len(code) >= 2:
                n_pairs = min(50, len(code) * (len(code) - 1) // 2)
                pair_indices = []
                for a in range(len(code)):
                    for b in range(a + 1, len(code)):
                        pair_indices.append((a, b))
                random.shuffle(pair_indices)
                pair_indices = pair_indices[:n_pairs]
                
                for a, b in pair_indices:
                    if time.time() >= deadline:
                        break
                    
                    c1 = code[a]
                    c2 = code[b]
                    
                    # Temporarily remove both
                    counts[neighbors_arr[c1, :]] -= 1
                    counts[neighbors_arr[c2, :]] -= 1
                    
                    if np.any(counts == 0):
                        uncovered_mask = (counts == 0)
                        uncovered_words = np.where(uncovered_mask)[0]
                        
                        # Find candidate that covers all uncovered words
                        candidate_set = None
                        for uw in uncovered_words:
                            rc = reverse_cover_arr[uw, :reverse_cover_counts[uw]]
                            rc_list = rc.tolist()
                            if candidate_set is None:
                                candidate_set = set(rc_list)
                            else:
                                candidate_set &= set(rc_list)
                        
                        if candidate_set is not None and len(candidate_set) > 0:
                            candidate_set.discard(c1)
                            candidate_set.discard(c2)
                            if len(candidate_set) > 0:
                                # Pick best candidate
                                best_cand = None
                                best_gain = -1
                                for cand in list(candidate_set)[:15]:
                                    gain = counts[neighbors_arr[cand, :]].sum()
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_cand = cand
                                
                                if best_cand is not None:
                                    # Replace c1 with best_cand, remove c2
                                    code[a] = best_cand
                                    code.pop(b)
                                    counts[neighbors_arr[c1, :]] += 1
                                    counts[neighbors_arr[c2, :]] += 1
                                    counts[neighbors_arr[best_cand, :]] += 1
                                    improved = True
                                    break
                        # Restore
                        counts[neighbors_arr[c1, :]] += 1
                        counts[neighbors_arr[c2, :]] += 1
                    else:
                        # Can remove both
                        code.pop(b)
                        code.pop(a)
                        improved = True
                        break
        
        return code
    
    # Main search loop
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 10:
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size or best_code is None:
            if time.time() < deadline - 5:
                code = local_search_improve(code, deadline - 3)
                code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # Extra polish on new best
            if time.time() < deadline - 5:
                best_code = local_search_improve(best_code, deadline - 3)
                best_code = reduce_code_fast(best_code)
        
        seed += 1
    
    if best_code is None:
        return [idx_to_tuple(i) for i in range(total_words)]
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(w)
    
    return result
