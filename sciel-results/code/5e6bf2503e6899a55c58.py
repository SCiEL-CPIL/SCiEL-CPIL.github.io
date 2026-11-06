import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 85  # Safety margin before 90s kill

    # Precompute masks for blocking
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    total_masks = sum(len(m) for m in masks_by_dist)

    # Step 1: Find a [17,8,6] linear code using random generator matrices
    k = 8
    linear_code = None

    # Try random generator matrices with better initialization
    for attempt in range(500):
        if time.time() - start_time > 3.0:
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

    # Step 3: Expand the linear code with greedy
    blocked = bytearray(1 << N)
    code = list(linear_code)  # 256 words
    
    # Block all neighbors of the linear code
    for w in code:
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1

    # Greedy scan for additional words
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1

    code_set = set(code)
    initial_size = len(code)

    # Step 4: Efficient local search via swap operations
    # Strategy: try removing a word and adding a better one that frees up more space
    if time.time() - start_time < time_limit - 5:
        # Build a list of candidate words that are "almost" addable
        # A word is a good candidate if it's blocked by only 1-2 code words
        
        # For each position in code, try swapping with a candidate
        for iteration in range(10):
            if time.time() - start_time > time_limit - 3:
                break
            
            improved = False
            for i in range(len(code)):
                if time.time() - start_time > time_limit - 3:
                    break
                
                removed = code[i]
                
                # Find candidates: words at distance 1 to MIN_DIST-1 from 'removed'
                # that are currently blocked but would be free if 'removed' were gone
                candidate_freed = []
                for d in range(1, MIN_DIST):
                    for mask in masks_by_dist[d]:
                        w = removed ^ mask
                        if w in code_set:
                            continue
                        if not blocked[w]:
                            continue
                        # Check if w is at distance >= MIN_DIST from all other code words
                        # Quick check: only need to verify against words that could block it
                        ok = True
                        for j, cw in enumerate(code):
                            if j == i:
                                continue
                            diff = w ^ cw
                            if bin(diff).count('1') < MIN_DIST:
                                ok = False
                                break
                        if ok:
                            candidate_freed.append(w)
                
                if not candidate_freed:
                    continue
                
                # Pick the best candidate (one that blocks the fewest other words)
                best_w = None
                best_block_count = float('inf')
                for w in candidate_freed[:10]:  # Limit to first 10 for speed
                    # Count how many words this would block
                    block_count = 0
                    for d in range(MIN_DIST):
                        for mask in masks_by_dist[d]:
                            neighbor = w ^ mask
                            if neighbor not in code_set and blocked[neighbor]:
                                block_count += 1
                    if block_count < best_block_count:
                        best_block_count = block_count
                        best_w = w
                
                if best_w is None:
                    continue
                
                # Perform the swap: remove 'removed', add 'best_w'
                code.remove(removed)
                code_set.remove(removed)
                
                # Unblock neighbors of removed
                for d in range(MIN_DIST):
                    for mask in masks_by_dist[d]:
                        blocked[removed ^ mask] = 0
                
                # Block neighbors of best_w
                for d in range(MIN_DIST):
                    for mask in masks_by_dist[d]:
                        blocked[best_w ^ mask] = 1
                
                code.append(best_w)
                code_set.add(best_w)
                
                # After swap, try to greedily add more words
                old_len = len(code)
                for w in range(1 << N):
                    if time.time() - start_time > time_limit - 3:
                        break
                    if not blocked[w] and w not in code_set:
                        code.append(w)
                        code_set.add(w)
                        for d in range(MIN_DIST):
                            for mask in masks_by_dist[d]:
                                blocked[w ^ mask] = 1
                
                # If we gained words, keep the change
                if len(code) > old_len:
                    improved = True
                else:
                    # If no gain, revert
                    # Undo additions
                    while len(code) > old_len:
                        w = code.pop()
                        code_set.remove(w)
                        for d in range(MIN_DIST):
                            for mask in masks_by_dist[d]:
                                blocked[w ^ mask] = 0
                    # Undo swap: remove best_w, re-add removed
                    code.remove(best_w)
                    code_set.remove(best_w)
                    for d in range(MIN_DIST):
                        for mask in masks_by_dist[d]:
                            blocked[best_w ^ mask] = 0
                    code.append(removed)
                    code_set.add(removed)
                    for d in range(MIN_DIST):
                        for mask in masks_by_dist[d]:
                            blocked[removed ^ mask] = 1
            
            if not improved:
                break

    # Step 5: Final greedy pass
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w] and w not in code_set:
            code.append(w)
            code_set.add(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1

    return code
