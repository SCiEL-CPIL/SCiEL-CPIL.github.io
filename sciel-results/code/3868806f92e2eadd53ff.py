import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Safety margin

    # Precompute masks for blocking
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)

    # Step 1: Find a [17,8,6] linear code
    k = 8
    linear_code = None

    # Try random generator matrices
    for attempt in range(200):
        if time.time() - start_time > 5.0:
            break
        G = np.random.randint(0, 2, size=(k, N))
        
        # Convert G rows to integers
        G_ints = []
        for i in range(k):
            val = 0
            for j in range(N):
                if G[i, j] == 1:
                    val |= (1 << j)
            G_ints.append(val)
        
        # Generate all codewords via iterative XOR
        codewords = [0]
        for i in range(k):
            new_codewords = [c ^ G_ints[i] for c in codewords]
            codewords.extend(new_codewords)
        
        # Check min distance: for linear codes, min weight of nonzero codewords = min distance
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

    # Step 2: If no linear code found, use greedy from 0
    if linear_code is None:
        blocked = bytearray(1 << N)
        code = []
        
        def add_word(w):
            if blocked[w]:
                return
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1
        
        add_word(0)
        for w in range(1, 1 << N):
            if time.time() - start_time > time_limit:
                break
            if not blocked[w]:
                add_word(w)
        return code

    # Step 3: Expand the linear code with greedy (weight-ordered for better packing)
    blocked = bytearray(1 << N)
    code = list(linear_code)  # 256 words
    
    # Block all neighbors of the linear code
    for w in code:
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1

    # Greedy scan for additional words, ordered by weight (lighter words pack better)
    all_words = list(range(1 << N))
    all_words.sort(key=lambda x: bin(x).count('1'))
    
    for w in all_words:
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1

    # Step 4: Iterative local search - remove and re-add in different orders
    # Multiple passes with different orderings to escape local optima
    for iteration in range(20):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try different orderings: reverse weight, random shuffle, etc.
        if iteration % 3 == 0:
            order = sorted(range(len(code)), key=lambda i: -bin(code[i]).count('1'))
        elif iteration % 3 == 1:
            order = list(range(len(code)))
            random.shuffle(order)
        else:
            order = sorted(range(len(code)), key=lambda i: bin(code[i]).count('1'))
        
        # Remove all words in order, then re-add greedily
        removed = []
        for idx in order:
            if time.time() - start_time > time_limit - 5:
                break
            w = code.pop(idx)
            # Unblock neighbors of w
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 0
            removed.append(w)
        
        # Re-add all removed words greedily
        for w in removed:
            if time.time() - start_time > time_limit - 5:
                break
            if not blocked[w]:
                code.append(w)
                for d in range(MIN_DIST):
                    for mask in masks_by_dist[d]:
                        blocked[w ^ mask] = 1
        
        # After re-adding, try to add any new words that became available
        for w in all_words:
            if time.time() - start_time > time_limit - 5:
                break
            if not blocked[w]:
                code.append(w)
                for d in range(MIN_DIST):
                    for mask in masks_by_dist[d]:
                        blocked[w ^ mask] = 1

    return code
