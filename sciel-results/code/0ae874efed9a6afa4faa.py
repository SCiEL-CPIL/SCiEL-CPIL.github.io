import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Safety margin

    # Precompute masks for distances 0 to MIN_DIST-1 (i.e., 0 to 5)
    # These are the "conflicting" positions.
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    # Flatten masks for faster iteration
    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])
    
    # Step 1: Find a [17,8,6] linear code
    k = 8
    linear_code = None

    # Try random generator matrices
    for attempt in range(500):
        if time.time() - start_time > 2.0:
            break
        G = np.random.randint(0, 2, size=(k, N))
        
        G_ints = []
        for i in range(k):
            val = 0
            for j in range(N):
                if G[i, j] == 1:
                    val |= (1 << j)
            G_ints.append(val)
        
        codewords = [0]
        for i in range(k):
            new_codewords = [c ^ G_ints[i] for c in codewords]
            codewords.extend(new_codewords)
        
        min_d = N + 1
        for c in codewords:
            if c == 0:
                continue
            w = bin(c).count('1')
            if w < min_d:
                min_d = w
                if min_d < MIN_DIST:
                    break
        
        if min_d >= MIN_DIST:
            linear_code = codewords
            break

    # Fallback if no linear code found (unlikely)
    if linear_code is None:
        linear_code = [0]

    # Step 2: Initialize data structures
    # block_count[u] = number of words in current code within distance < MIN_DIST of u
    # We use uint16 because max count is small (code size < 400)
    block_count = np.zeros(1 << N, dtype=np.uint16)
    in_code = np.zeros(1 << N, dtype=bool)
    
    code = []

    def add_word(w):
        if in_code[w]:
            return False
        in_code[w] = True
        code.append(w)
        # Increment block counts for all neighbors
        # Using a loop is faster than numpy vectorization for small N (9402)
        # because of overhead, but let's try numpy if masks are array
        # Actually, simple loop is fine.
        for mask in all_masks:
            block_count[w ^ mask] += 1
        return True

    def remove_word(w):
        if not in_code[w]:
            return False
        in_code[w] = False
        code.remove(w) # This is O(N) but code is small
        for mask in all_masks:
            block_count[w ^ mask] -= 1
        return True

    # Add linear code
    for w in linear_code:
        add_word(w)

    # Step 3: Greedy Expansion
    # Scan all words, add if available
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if not in_code[w] and block_count[w] == 0:
            add_word(w)

    # Step 4: Local Search
    # Strategy: Remove a word, try to add as many neighbors as possible.
    # If we add more than 1, we gain size.
    # If we add 1, we swap (neutral size, but might help future).
    # We will prioritize size gain.
    
    # To speed up, we can iterate over a copy of the code list
    # and shuffle it to avoid bias
    
    improved = True
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        # Shuffle code for randomness in local search
        code_copy = code[:]
        random.shuffle(code_copy)
        
        for w in code_copy:
            if time.time() - start_time > time_limit - 2:
                break
            
            # Temporarily remove w
            # We need to handle the case where w was already removed in a previous step of this loop?
            # No, we iterate over a copy, but we modify 'code' and 'in_code'.
            # So we must check if w is still in code.
            if not in_code[w]:
                continue
                
            # Remove w
            in_code[w] = False
            code.remove(w)
            for mask in all_masks:
                block_count[w ^ mask] -= 1
            
            added_count = 0
            # Try to add neighbors
            # We only need to check neighbors of w, as only they became unblocked (or less blocked)
            # Actually, removing w decrements counts for N(w).
            # So only u in N(w) can have block_count[u] become 0.
            # So we only iterate over N(w).
            
            for mask in all_masks:
                u = w ^ mask
                if not in_code[u] and block_count[u] == 0:
                    # Add u
                    in_code[u] = True
                    code.append(u)
                    added_count += 1
                    # Update block counts for u's neighbors
                    for mask2 in all_masks:
                        block_count[u ^ mask2] += 1
            
            # If we added at least 1 word, we kept the removal of w.
            # If added_count == 0, we should restore w.
            if added_count == 0:
                # Restore w
                in_code[w] = True
                code.append(w)
                for mask in all_masks:
                    block_count[w ^ mask] += 1
            else:
                # We removed 1, added k.
                # If k > 1, size increased.
                # If k == 1, size same.
                # We mark as improved if size increased OR if we swapped (to explore).
                # But swapping might cause cycling.
                # Let's only mark improved if size increased.
                # However, sometimes a swap is necessary to reach a larger code.
                # Let's accept swaps but be careful.
                # For now, accept any addition.
                improved = True

    return code
