import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Fallback for non-standard cases
    if q != 3 or n != 8 or radius != 1:
        total_words = q ** n
        powers = [q ** i for i in range(n)]
        neighbors_list = []
        for w in range(total_words):
            nbs = [w]
            for k in range(n):
                d_k = (w // powers[k]) % q
                for v in range(q):
                    if v != d_k:
                        w_new = w + (v - d_k) * powers[k]
                        nbs.append(w_new)
            neighbors_list.append(nbs)
        neighbors = np.array(neighbors_list, dtype=np.int32)
        covered = np.zeros(total_words, dtype=bool)
        code = []
        uncovered_count = total_words
        while uncovered_count > 0:
            scores = np.sum(~covered[neighbors], axis=1)
            max_score = np.max(scores)
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[0]
            code.append(int(best_idx))
            nbs = neighbors[best_idx]
            newly = np.sum(~covered[nbs])
            covered[nbs] = True
            uncovered_count -= newly
        result = []
        for w in code:
            digits = [0] * n
            x = w
            for i in range(n):
                digits[i] = x % q
                x //= q
            result.append(digits)
        return result
    
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    
    # Precompute neighbors
    neighbors_list = []
    for w in range(total_words):
        nbs = [w]
        for k in range(n):
            d_k = (w // powers[k]) % q
            for v in range(q):
                if v != d_k:
                    w_new = w + (v - d_k) * powers[k]
                    nbs.append(w_new)
        neighbors_list.append(nbs)
        
    neighbors = np.array(neighbors_list, dtype=np.int32)
    
    # 1. Generate initial code (729 words, valid)
    # Tetracode: [a, b, a+b, a+2b]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Convert to integers
    tc_ints = []
    for tc in tetracode:
        val = 0
        for i in range(4):
            val += tc[i] * (3 ** (3 - i)) # Big endian? Or little?
            # Let's stick to little endian for consistency with powers
            # Actually, powers[k] = 3^k. So digit 0 is least significant.
            # Let's just use the integer representation consistent with the neighbor generation.
            # In neighbor generation: d_k = (w // powers[k]) % q.
            # So digit k is at position 3^k.
            pass
        
    # Simpler: Generate all 729 words directly
    # Code = { (t, x) | t in Tetracode, x in {0,1,2}^4 }
    # Integer value: sum(t[i] * 3^i for i in 0..3) + sum(x[j] * 3^(4+j) for j in 0..3)
    tc_vals = []
    for a in range(3):
        for b in range(3):
            t0 = a
            t1 = b
            t2 = (a + b) % 3
            t3 = (a + 2 * b) % 3
            val = t0 * 1 + t1 * 3 + t2 * 9 + t3 * 27
            tc_vals.append(val)
            
    code = []
    for tv in tc_vals:
        for tail in itertools.product(range(3), repeat=4):
            # tail[0] is digit 4, tail[1] is digit 5, etc?
            # Let's say tail[0] corresponds to position 4 (3^4)
            val = tv + tail[0] * 81 + tail[1] * 243 + tail[2] * 729 + tail[3] * 2187
            code.append(val)
            
    code = set(code) # Ensure uniqueness
    code_list = list(code)
    
    # 2. Local Search
    # Build coverage
    coverage = np.zeros(total_words, dtype=np.int32)
    for cw in code_list:
        coverage[neighbors[cw]] += 1
        
    # Helper to check validity (optional, for debug)
    # assert np.all(coverage > 0)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Removal
        # Shuffle to avoid bias
        random.shuffle(code_list)
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            cw = code_list[i]
            nbs = neighbors[cw]
            # If min coverage in neighbors is >= 2, we can remove
            if np.min(coverage[nbs]) >= 2:
                coverage[nbs] -= 1
                code_list.pop(i)
                improved = True
        
        # Phase 2: 2-for-1 Swap
        if not improved and len(code_list) > 100: # Only try if stuck
            # Try to find a pair (i, j) to remove and a k to add
            # This is O(N^2). N ~ 500. 250k iterations.
            # Each iteration is fast.
            found_swap = False
            # Iterate over pairs
            # To save time, we can iterate over a subset or random pairs
            # But let's try all pairs if time allows
            n_code = len(code_list)
            # Create index map for fast "in code" check
            code_set = set(code_list)
            
            for i in range(n_code):
                if time.time() > deadline - 5:
                    break
                if found_swap:
                    break
                cw_i = code_list[i]
                nbs_i = neighbors[cw_i]
                
                for j in range(i + 1, n_code):
                    if time.time() > deadline - 5:
                        break
                    if found_swap:
                        break
                    cw_j = code_list[j]
                    nbs_j = neighbors[cw_j]
                    
                    # Temporarily remove
                    coverage[nbs_i] -= 1
                    coverage[nbs_j] -= 1
                    
                    uncovered = np.where(coverage == 0)[0]
                    
                    if len(uncovered) == 0:
                        # Both removable! (Should have been caught in Phase 1)
                        # Remove them
                        code_list.pop(j)
                        code_list.pop(i)
                        found_swap = True
                        # Note: coverage is already updated
                        # But we broke out of loops, need to handle cleanup
                        # Actually, if we remove both, coverage is correct.
                        # We need to update code_set and break
                        code_set.discard(cw_i)
                        code_set.discard(cw_j)
                        improved = True
                        break
                    
                    # Find candidate k
                    # k must be neighbor of all uncovered words
                    # Start with neighbors of first uncovered word
                    candidates = set(neighbors[uncovered[0]])
                    
                    for w in uncovered[1:]:
                        candidates.intersection_update(neighbors[w])
                        if not candidates:
                            break
                    
                    if candidates:
                        # Find a k in candidates that is not in code_set
                        for k in candidates:
                            if k not in code_set:
                                # Add k
                                code_list.append(k)
                                code_set.add(k)
                                coverage[neighbors[k]] += 1
                                
                                # Remove i and j
                                # Find indices
                                # Since we are in nested loops, i and j are known
                                # But code_list changed? No, we appended.
                                # We need to remove cw_i and cw_j
                                # pop(j) then pop(i)
                                code_list.pop(j)
                                code_list.pop(i)
                                
                                code_set.discard(cw_i)
                                code_set.discard(cw_j)
                                
                                found_swap = True
                                improved = True
                                break
                    
                    # Restore coverage if no swap found
                    if not found_swap:
                        coverage[nbs_i] += 1
                        coverage[nbs_j] += 1
                
            # If found_swap, we modified code_list and coverage.
            # We need to restart the outer loop or continue.
            # The 'improved' flag will trigger restart.

    # Final Polish: One more removal pass
    random.shuffle(code_list)
    for i in range(len(code_list) - 1, -1, -1):
        if time.time() > deadline - 1:
            break
        cw = code_list[i]
        nbs = neighbors[cw]
        if np.min(coverage[nbs]) >= 2:
            coverage[nbs] -= 1
            code_list.pop(i)

    # Convert to output
    result = []
    for w in code_list:
        digits = [0] * n
        x = w
        for i in range(n):
            digits[i] = x % q
            x //= q
        result.append(digits)
        
    return result

def _generic_construct(q, n, radius, deadline):
    total_words = q ** n
    powers = [q ** i for i in range(n)]
    neighbors_list = []
    for w in range(total_words):
        nbs = [w]
        for k in range(n):
            d_k = (w // powers[k]) % q
            for v in range(q):
                if v != d_k:
                    w_new = w + (v - d_k) * powers[k]
                    nbs.append(w_new)
        neighbors_list.append(nbs)
    neighbors = np.array(neighbors_list, dtype=np.int32)
    covered = np.zeros(total_words, dtype=bool)
    code = []
    uncovered_count = total_words
    while uncovered_count > 0 and time.time() < deadline:
        scores = np.sum(~covered[neighbors], axis=1)
        max_score = np.max(scores)
        candidates = np.where(scores == max_score)[0]
        best_idx = candidates[0]
        code.append(int(best_idx))
        nbs = neighbors[best_idx]
        newly = np.sum(~covered[nbs])
        covered[nbs] = True
        uncovered_count -= newly
    result = []
    for w in code:
        digits = [0] * n
        x = w
        for i in range(n):
            digits[i] = x % q
            x //= q
        result.append(digits)
    return result
