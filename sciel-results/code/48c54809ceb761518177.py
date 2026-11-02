import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Specific case: q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        total = q ** n
        if total > 100000:
            return []
        # Simple greedy for small general cases
        all_words = list(range(total))
        uncovered = set(all_words)
        codewords = []
        
        def idx_to_digits_gen(idx, qq, nn):
            digits = [0] * nn
            for i in range(nn - 1, -1, -1):
                digits[i] = idx % qq
                idx //= qq
            return digits
        
        while uncovered and time.time() < deadline:
            best = None
            best_count = -1
            for w in uncovered:
                count = 1
                for i in range(n):
                    for v in range(q):
                        nw = idx_to_digits_gen(w, q, n)
                        if nw[i] != v:
                            nw[i] = v
                            nw_idx = 0
                            for d in nw:
                                nw_idx = nw_idx * q + d
                            if nw_idx in uncovered:
                                count += 1
                if count > best_count:
                    best_count = count
                    best = w
            if best is None:
                break
            codewords.append(idx_to_digits_gen(best, q, n))
            # Remove covered
            to_remove = set()
            best_digits = idx_to_digits_gen(best, q, n)
            for i in range(n):
                for v in range(q):
                    nw = best_digits[:]
                    nw[i] = v
                    nw_idx = 0
                    for d in nw:
                        nw_idx = nw_idx * q + d
                    to_remove.add(nw_idx)
            uncovered -= to_remove
        return codewords

    total_words = 3 ** 8
    num_coords = 8
    alphabet = 3

    # Precompute neighbor lists efficiently
    # Each word has 1 + 8*2 = 17 neighbors
    neighbors = np.empty((total_words, 17), dtype=np.int32)
    
    # Powers of 3
    powers = np.array([3**i for i in range(num_coords)], dtype=np.int32)
    
    for idx in range(total_words):
        # Convert index to digits
        temp = idx
        digits = np.zeros(num_coords, dtype=np.int32)
        for i in range(num_coords - 1, -1, -1):
            digits[i] = temp % alphabet
            temp //= alphabet
        
        neighbors[idx, 0] = idx
        nb_count = 1
        for pos in range(num_coords):
            orig_val = digits[pos]
            for val in range(alphabet):
                if val == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[pos] = val
                new_idx = int(np.dot(new_digits, powers))
                neighbors[idx, nb_count] = new_idx
                nb_count += 1
    
    # Greedy Construction with vectorized count updates
    def greedy_construct():
        uncovered = np.ones(total_words, dtype=np.bool_)
        in_code = np.zeros(total_words, dtype=np.bool_)
        codewords = []
        
        # Initial counts: each word covers 17 others (itself + 16 neighbors)
        # Actually, the count for a candidate codeword c is the number of currently
        # uncovered words in its neighborhood
        counts = np.full(total_words, 17, dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Find max count
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_cw = candidates[random.randint(0, len(candidates) - 1)]
            
            in_code[best_cw] = True
            codewords.append(best_cw)
            
            # Update: mark neighbors as covered
            nbs = neighbors[best_cw]
            for nb in nbs:
                if uncovered[nb]:
                    uncovered[nb] = False
                    # Decrement counts for all candidates that cover nb
                    # A candidate c covers nb if nb is in N(c), i.e., c is in N(nb)
                    for c in neighbors[nb]:
                        if not in_code[c]:
                            counts[c] -= 1

        return codewords

    # Pruning: remove redundant codewords
    def prune_once(code_set):
        if not code_set:
            return code_set
        
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighbors[c]:
                cov_count[nb] += 1
                
        removable = []
        for c in code_set:
            is_removable = True
            for nb in neighbors[c]:
                if cov_count[nb] <= 1:
                    is_removable = False
                    break
            if is_removable:
                removable.append(c)
                
        if removable:
            for c in removable:
                code_set.remove(c)
            return prune_once(code_set)
        else:
            return code_set

    # Local Search: 1-swaps
    def local_search(code_set):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            c_list = list(code_set)
            random.shuffle(c_list)
            
            # Compute coverage counts
            cov_count = np.zeros(total_words, dtype=np.int32)
            for c in code_set:
                for nb in neighbors[c]:
                    cov_count[nb] += 1
            
            # Try 1-swaps: remove c1, add d
            for c1 in c_list:
                if time.time() > deadline - 5:
                    return code_set
                if c1 not in code_set:
                    continue
                    
                # Find words that would become uncovered if c1 removed
                u_words = []
                for w in neighbors[c1]:
                    if cov_count[w] <= 1:
                        u_words.append(w)
                
                if not u_words:
                    # c1 is redundant
                    code_set.remove(c1)
                    improved = True
                    break
                
                # Find d that covers all u_words
                # d must be within distance 1 of each w in u_words
                candidates = set(neighbors[u_words[0]].tolist())
                candidates.discard(c1)
                
                for d in candidates:
                    if d in code_set:
                        continue
                    # Check if d covers all u_words
                    d_neighbors = set(neighbors[d].tolist())
                    if all(w in d_neighbors for w in u_words):
                        # Check validity: words in N(c1) \ N(d) must have cov_count >= 2
                        c1_neighbors = set(neighbors[c1].tolist())
                        valid = True
                        for w in c1_neighbors - d_neighbors:
                            if cov_count[w] <= 1:
                                valid = False
                                break
                        
                        if valid:
                            code_set.remove(c1)
                            code_set.add(d)
                            improved = True
                            break
                
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-removals: remove c1, c2
            for i in range(len(c_list)):
                if time.time() > deadline - 5:
                    return code_set
                c1 = c_list[i]
                if c1 not in code_set:
                    continue
                
                for j in range(i + 1, len(c_list)):
                    c2 = c_list[j]
                    if c2 not in code_set:
                        continue
                    
                    c1_nbs = set(neighbors[c1].tolist())
                    c2_nbs = set(neighbors[c2].tolist())
                    union_nbs = c1_nbs | c2_nbs
                    
                    valid = True
                    for w in union_nbs:
                        rem = cov_count[w]
                        if w in c1_nbs:
                            rem -= 1
                        if w in c2_nbs:
                            rem -= 1
                        if rem <= 0:
                            valid = False
                            break
                    
                    if valid:
                        code_set.remove(c1)
                        code_set.remove(c2)
                        improved = True
                        break
                
                if improved:
                    break

        return code_set

    # Main: try multiple random restarts of greedy + local search
    best_code = None
    best_size = float('inf')
    
    # First, do one greedy to get a baseline
    codewords = greedy_construct()
    code_set = set(codewords)
    code_set = prune_once(code_set)
    code_set = local_search(code_set)
    
    best_size = len(code_set)
    best_code = set(code_set)
    
    # Try additional restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        num_restarts = min(3, int(remaining_time / 20))
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            codewords = greedy_construct()
            code_set = set(codewords)
            code_set = prune_once(code_set)
            code_set = local_search(code_set)
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)

    if best_code is None:
        # Fallback: just do one greedy
        codewords = greedy_construct()
        best_code = set(codewords)
        best_code = prune_once(best_code)

    # Final conversion
    result = []
    seen = set()
    for c in best_code:
        temp = c
        digits = []
        for i in range(num_coords - 1, -1, -1):
            digits.append(temp % alphabet)
            temp //= alphabet
        t = tuple(digits)
        if t not in seen:
            seen.add(t)
            result.append(digits)
    return result
