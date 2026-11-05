import itertools
import numpy as np
import time

def construct(q, n, radius):
    # For this specific task, we assume q=3, n=7, radius=1
    # But the function signature is general.
    # If parameters differ, we might need a fallback.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies we only need to handle the specific case or a general case.
    # Given the time limit and complexity, I will specialize for q=3, n=7, r=1
    # and provide a fallback for other cases (e.g., greedy only).
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Simple Greedy
        return _greedy_construct(q, n, radius)

    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Map word index to word tuple for later conversion
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute balls as bitsets and neighbor lists
    balls = [0] * N
    neighbors = [[] for _ in range(N)]
    
    for i in range(N):
        w = words[i]
        mask = 1 << i
        neighbors[i].append(i)
        for pos in range(n):
            for delta in [1, -1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    j = word_to_idx[tuple(new_w)]
                    mask |= (1 << j)
                    neighbors[i].append(j)
        balls[i] = mask

    all_mask = (1 << N) - 1
    
    # 1. Greedy Initialization
    # Optimization: Instead of scanning all N words every time, 
    # we can maintain a list of "active" words or just scan. 
    # Scanning 2187 words is fast. 186 * 2187 = 400k ops.
    # Each op is (ball & uncovered).bit_count(). This is fast.
    
    uncovered = all_mask
    code = []
    
    while uncovered:
        best_i = -1
        best_cov = 0
        for i in range(N):
            cov = (balls[i] & uncovered).bit_count()
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        uncovered &= ~balls[best_i]
        
    # 2. Local Search
    # Use numpy for cover_count updates for speed
    cover_count = np.zeros(N, dtype=np.int32)
    for i in code:
        cover_count[neighbors[i]] += 1
        
    # Precompute masks for count 1 and 2
    # We need to update these masks efficiently.
    # Recalculating from scratch is O(N). N=2187. 
    # If we do this after every change, and we have many changes, it might be slow.
    # But changes are O(1) in terms of words affected (neighbors of i/j/k).
    # So we should update masks incrementally.
    
    mask_count_1 = 0
    mask_count_2 = 0
    for i in range(N):
        if cover_count[i] == 1:
            mask_count_1 |= (1 << i)
        elif cover_count[i] == 2:
            mask_count_2 |= (1 << i)

    improved = True
    while improved:
        improved = False
        
        # Pass 1: Single Removal
        for idx_in_code, i in enumerate(code):
            if (balls[i] & mask_count_1) == 0:
                # Remove i
                code.pop(idx_in_code)
                
                # Update cover_count and masks
                for w_idx in neighbors[i]:
                    old_c = cover_count[w_idx]
                    cover_count[w_idx] -= 1
                    new_c = cover_count[w_idx]
                    
                    if old_c == 1:
                        mask_count_1 &= ~(1 << w_idx)
                    elif old_c == 2:
                        mask_count_2 &= ~(1 << w_idx)
                    
                    if new_c == 1:
                        mask_count_1 |= (1 << w_idx)
                    elif new_c == 2:
                        mask_count_2 |= (1 << w_idx)
                
                improved = True
                break # Restart loop after modification
                
        if improved:
            continue

        # Pass 2: Pair Replacement
        len_code = len(code)
        for i_idx in range(len_code):
            i = code[i_idx]
            Bi = balls[i]
            for j_idx in range(i_idx + 1, len_code):
                j = code[j_idx]
                Bj = balls[j]
                
                Bi_only = Bi & ~Bj
                Bj_only = Bj & ~Bi
                Bi_Bj = Bi & Bj
                
                U = (Bi_only & mask_count_1) | (Bj_only & mask_count_1) | (Bi_Bj & mask_count_2)
                
                if U == 0:
                    # Both redundant
                    code.pop(j_idx)
                    for w_idx in neighbors[j]:
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] -= 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        
                    code.pop(i_idx)
                    for w_idx in neighbors[i]:
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] -= 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        
                    improved = True
                    break
                    
                # Check if U is covered by some k
                # Pick a word w in U to restrict candidates
                lsb = U & -U
                w_idx = lsb.bit_length() - 1
                
                # Candidates k are neighbors of w_idx
                # Note: k must be in the "ball" of w_idx
                # Which is exactly neighbors[w_idx]
                
                found_k = -1
                for k in neighbors[w_idx]:
                    if k == i or k == j:
                        continue
                    if (U & ~balls[k]) == 0:
                        found_k = k
                        break
                        
                if found_k != -1:
                    # Replace i, j with k
                    code.pop(j_idx)
                    for w_idx in neighbors[j]:
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] -= 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        
                    code.pop(i_idx)
                    for w_idx in neighbors[i]:
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] -= 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        
                    code.append(found_k)
                    for w_idx in neighbors[found_k]:
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] += 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        
                    improved = True
                    break
                    
            if improved:
                break
                
    return [words[i] for i in code]

def _greedy_construct(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Simple Greedy without bitsets for fallback
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_i = -1
        best_cov = 0
        for i in range(N):
            w = words[i]
            cov = 0
            for u_idx in uncovered:
                u = words[u_idx]
                if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        # Remove covered
        w = words[best_i]
        to_remove = []
        for u_idx in uncovered:
            u = words[u_idx]
            if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                to_remove.append(u_idx)
        uncovered.difference_update(to_remove)
    return [words[i] for i in code]
