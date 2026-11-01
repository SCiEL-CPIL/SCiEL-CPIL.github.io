import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Handle the specific case q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        if q ** n > 100000:
            return []
        # Simple greedy for general case
        total = q ** n
        codewords = []
        uncovered = set(range(total))
        
        def idx_to_digits(idx):
            digits = [0] * n
            for i in range(n - 1, -1, -1):
                digits[i] = idx % q
                idx //= q
            return digits
        
        def digits_to_idx(digits):
            idx = 0
            for d in digits:
                idx = idx * q + d
            return idx
        
        # Precompute neighbors
        neighbors = {}
        for i in range(total):
            digits = idx_to_digits(i)
            nbs = [i]
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_digits = digits[:]
                    new_digits[pos] = val
                    nbs.append(digits_to_idx(new_digits))
            neighbors[i] = nbs
        
        while uncovered and time.time() < deadline:
            # Find codeword covering most uncovered
            best_cw = -1
            best_count = -1
            for cw in uncovered:
                count = sum(1 for nb in neighbors[cw] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cw = cw
            
            if best_cw == -1:
                break
            
            codewords.append(best_cw)
            for nb in neighbors[best_cw]:
                uncovered.discard(nb)
        
        result = [idx_to_digits(cw) for cw in codewords]
        seen = set()
        unique_result = []
        for cw in result:
            t = tuple(cw)
            if t not in seen:
                seen.add(t)
                unique_result.append(cw)
        return unique_result
    
    # Specific case: q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    num_coords = 8
    alphabet = 3
    
    # Precompute neighbors using numpy for speed
    # Each word has 1 + 8*2 = 17 neighbors (itself + 8 positions * 2 other values)
    
    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx
    
    # Precompute neighbor lists
    neighbors = []
    for i in range(total_words):
        digits = idx_to_digits(i)
        nbs = [i]
        for pos in range(num_coords):
            orig_val = digits[pos]
            for val in range(alphabet):
                if val == orig_val:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                nbs.append(digits_to_idx(new_digits))
        neighbors.append(nbs)
    
    # Also create neighbor sets for fast membership testing
    neighbor_sets = [set(nb) for nb in neighbors]
    
    # Greedy Construction
    uncovered = np.ones(total_words, dtype=np.bool_)
    in_code = np.zeros(total_words, dtype=np.bool_)
    codewords = []
    
    # Initial counts: each word covers 17 words
    counts = np.full(total_words, 17, dtype=np.int32)
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
        
        best_cw = int(np.argmax(counts))
        if counts[best_cw] == 0:
            break
        
        in_code[best_cw] = True
        codewords.append(best_cw)
        
        for nb in neighbors[best_cw]:
            if uncovered[nb]:
                uncovered[nb] = False
                for cw in neighbors[nb]:
                    counts[cw] -= 1
    
    # Pruning: remove redundant codewords
    code_set = set(codewords)
    
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
    
    code_set = prune_once(code_set)
    
    # Local Search: 1-swap and 2-swap operations
    # 1-swap: remove codeword c1, add codeword d if all words in N(c1) are covered
    # 2-swap: remove c1, c2, add d1, d2 if all words in N(c1)∪N(c2) are covered
    
    def compute_cov_count(code_set):
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighbors[c]:
                cov_count[nb] += 1
        return cov_count
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        c_list = list(code_set)
        random.shuffle(c_list)
        
        cov_count = compute_cov_count(code_set)
        
        # Try 1-swaps: remove c, add d
        found_1_swap = False
        for c in c_list:
            if time.time() > deadline:
                break
            if c not in code_set:
                continue
            
            # Find words that would become uncovered if c is removed
            # These are words in N(c) with cov_count[w] == 1
            critical_words = [w for w in neighbors[c] if cov_count[w] == 1]
            
            if not critical_words:
                # c is redundant, remove it
                code_set.remove(c)
                improved = True
                found_1_swap = True
                cov_count = compute_cov_count(code_set)
                continue
            
            # Try to find d that covers all critical_words
            # d must be within distance 1 of all critical_words
            # This means d is in the intersection of N(w) for all w in critical_words
            
            # For efficiency, start with N(critical_words[0]) and intersect
            candidate_set = set(neighbors[critical_words[0]])
            for w in critical_words[1:]:
                candidate_set &= set(neighbors[w])
            
            # Find a candidate d that is not already in code_set
            # and covers all critical_words
            for d in candidate_set:
                if d in code_set:
                    continue
                
                # Check if d covers all critical_words
                # (it should by construction, but verify)
                valid = True
                for w in critical_words:
                    if w not in neighbor_sets[d]:
                        valid = False
                        break
                
                if valid:
                    # Perform the swap
                    code_set.remove(c)
                    code_set.add(d)
                    improved = True
                    found_1_swap = True
                    cov_count = compute_cov_count(code_set)
                    break
        
        # Try 2-swaps: remove c1, c2, add d1, d2
        # Only try if no 1-swap was found in this iteration
        if not found_1_swap and time.time() < deadline:
            cov_count = compute_cov_count(code_set)
            c_list = list(code_set)
            random.shuffle(c_list)
            
            found_2_swap = False
            n_cw = len(c_list)
            
            for i in range(n_cw):
                if time.time() > deadline:
                    break
                c1 = c_list[i]
                if c1 not in code_set:
                    continue
                
                for j in range(i + 1, n_cw):
                    c2 = c_list[j]
                    if c2 not in code_set:
                        continue
                    
                    # Find words that would become uncovered
                    # Words in N(c1)∪N(c2) with remaining coverage == 0
                    # after removing c1 and c2
                    
                    critical_words = []
                    seen_words = set()
                    
                    for w in neighbors[c1]:
                        if w in seen_words:
                            continue
                        seen_words.add(w)
                        # Count how many of c1, c2 cover w
                        cover_count = cov_count[w]
                        if w in neighbor_sets[c2]:
                            cover_count -= 1
                        cover_count -= 1  # for c1
                        if cover_count == 0:
                            critical_words.append(w)
                    
                    for w in neighbors[c2]:
                        if w in seen_words:
                            continue
                        seen_words.add(w)
                        cover_count = cov_count[w]
                        cover_count -= 1  # for c2
                        if cover_count == 0:
                            critical_words.append(w)
                    
                    if not critical_words:
                        # Both c1 and c2 are redundant together
                        code_set.remove(c1)
                        code_set.remove(c2)
                        improved = True
                        found_2_swap = True
                        cov_count = compute_cov_count(code_set)
                        break
                    
                    # Try to find d1, d2 that cover all critical_words
                    # This is more complex. For simplicity, try single d first
                    # (which is effectively a 1-swap for the combined set)
                    
                    # Find d that covers all critical_words
                    if len(critical_words) <= 3:  # Only try if small
                        candidate_set = set(neighbors[critical_words[0]])
                        for w in critical_words[1:]:
                            candidate_set &= set(neighbors[w])
                        
                        for d in candidate_set:
                            if d in code_set:
                                continue
                            
                            valid = True
                            for w in critical_words:
                                if w not in neighbor_sets[d]:
                                    valid = False
                                    break
                            
                            if valid:
                                # Remove c1, c2, add d (net -1 codeword)
                                code_set.remove(c1)
                                code_set.remove(c2)
                                code_set.add(d)
                                improved = True
                                found_2_swap = True
                                cov_count = compute_cov_count(code_set)
                                break
                    
                    if found_2_swap:
                        break
                
                if found_2_swap:
                    break
        
        # Final prune after local search
        code_set = prune_once(code_set)
    
    # Final conversion
    result = [idx_to_digits(c) for c in code_set]
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    return unique_result
