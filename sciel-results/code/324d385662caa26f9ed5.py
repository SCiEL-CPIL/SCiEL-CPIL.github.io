"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    total_words = q ** n

    # Fallback for large spaces where bitmask approach is infeasible
    if total_words > 200000:
        # Use the original algebraic construction as a fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n > 4:
            code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        elif n == 4:
            code = [h[:] for h in tetracode]
        else:
            code = [list(w) for w in itertools.product(range(q), repeat=n)]
            return code
        
        # Simple greedy removal for fallback
        # Convert to ints
        q_powers = np.array([q ** i for i in range(n)], dtype=np.int64)
        def word_to_int(w):
            return int(np.dot(np.array(w, dtype=np.int64), q_powers))
        def int_to_word(v):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = v % q
                v //= q
            return w
            
        code_ints = np.array([word_to_int(c) for c in code], dtype=np.int64)
        m = len(code_ints)
        all_neighbors = []
        for ci in code_ints:
            word = int_to_word(int(ci))
            neighbors = [int(ci)]
            if radius >= 1:
                for pos in range(n):
                    orig = word[pos]
                    for new_val in range(q):
                        if new_val == orig: continue
                        word[pos] = new_val
                        neighbors.append(word_to_int(word))
                    word[pos] = orig
            all_neighbors.append(np.array(neighbors, dtype=np.int64))
            
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for neighbors in all_neighbors:
            coverage_count[neighbors] += 1
            
        removed = np.zeros(m, dtype=bool)
        improvement_found = True
        while improvement_found and time.time() < deadline:
            improvement_found = False
            indices = np.argsort(-np.array([np.sum(coverage_count[all_neighbors[i]] > 1) if not removed[i] else -1 for i in range(m)]))
            for idx in indices:
                if removed[idx] or time.time() > deadline: continue
                neighbors = all_neighbors[idx]
                if np.any(coverage_count[neighbors] == 1): continue
                removed[idx] = True
                coverage_count[neighbors] -= 1
                improvement_found = True
                
        return [code[i] for i in range(m) if not removed[i]]

    # Bitmask approach for small/medium spaces
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbor masks
    neighbors = [0] * total_words
    
    def gen_neighbors(idx, r):
        w = idx_to_word(idx)
        mask = 0
        def rec(pos, cur_w, dist):
            nonlocal mask
            if dist > r:
                return
            if pos == n:
                j = word_to_idx(cur_w)
                mask |= (1 << j)
                return
            # Keep original
            rec(pos + 1, cur_w, dist)
            # Change
            for val in range(q):
                if val == cur_w[pos]:
                    continue
                cur_w[pos] = val
                rec(pos + 1, cur_w, dist + 1)
                cur_w[pos] = w[pos]
        
        cur_w = w[:]
        rec(0, cur_w, 0)
        return mask

    for i in range(total_words):
        neighbors[i] = gen_neighbors(i, radius)

    all_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = float('inf')
    
    max_iterations = 20
    
    for iter_num in range(max_iterations):
        if time.time() > deadline - 10:
            break
            
        uncovered = all_mask
        code_indices = []
        in_code = [False] * total_words
        
        while uncovered:
            best_score = -1
            best_idx = -1
            
            for i in range(total_words):
                if in_code[i]:
                    continue
                overlap = neighbors[i] & uncovered
                cnt = overlap.bit_count()
                if cnt > best_score:
                    best_score = cnt
                    best_idx = i
                    # Random tie-breaking to explore different solutions
                    # if cnt == best_score and random.random() < 0.5:
                    #     best_idx = i 
                    # (Disabled for determinism in first pass, enabled in later if needed)
                    
            if best_idx == -1:
                break
                
            code_indices.append(best_idx)
            in_code[best_idx] = True
            uncovered &= ~neighbors[best_idx]
            
            if len(code_indices) >= best_size:
                break
        
        # Local Search: Greedy Removal
        improved = True
        while improved:
            improved = False
            for i in range(len(code_indices) - 1, -1, -1):
                if len(code_indices) <= 1:
                    break
                union_others = 0
                for j in range(len(code_indices)):
                    if j != i:
                        union_others |= neighbors[code_indices[j]]
                        
                if union_others == all_mask:
                    code_indices.pop(i)
                    in_code[code_indices[i] if i < len(code_indices) else 0] = False # Careful with index
                    # Actually, in_code is only used in greedy. We can just rebuild or ignore.
                    # We don't use in_code in removal.
                    improved = True
                    if len(code_indices) < best_size:
                        best_size = len(code_indices)
                        best_code = code_indices[:]
                    break
        
        if len(code_indices) < best_size:
            best_size = len(code_indices)
            best_code = code_indices[:]
            
        if time.time() > deadline - 20:
            break

    if best_code is None:
        return [idx_to_word(i) for i in range(total_words)]

    return [idx_to_word(i) for i in best_code]

# EVOLVE-BLOCK-END
