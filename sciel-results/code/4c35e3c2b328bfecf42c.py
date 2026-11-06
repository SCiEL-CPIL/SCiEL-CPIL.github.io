import numpy as np
import time
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58

    # Precompute masks for blocking
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])
        
    # Greedy construction
    blocked = bytearray(1 << N)
    code = []
    
    def add_word(w):
        if blocked[w]:
            return
        code.append(w)
        for mask in all_masks:
            blocked[w ^ mask] = 1
            
    add_word(0)
    for w in range(1, 1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            add_word(w)
            
    # Prepare for Local Search
    in_code = bytearray(1 << N)
    for w in code:
        in_code[w] = 1
        
    block_count = np.zeros(1 << N, dtype=np.int32)
    
    for w in code:
        for mask in all_masks:
            block_count[w ^ mask] += 1
            
    # Local Search
    improved = True
    while improved and time.time() - start_time < time_limit - 2:
        improved = False
        for i in range(len(code)):
            if time.time() - start_time > time_limit - 2:
                break
            w_i = code[i]
            neighbors = [w_i ^ mask for mask in all_masks]
            
            # Tentatively remove w_i
            new_candidates = []
            for v in neighbors:
                block_count[v] -= 1
                if block_count[v] == 0 and not in_code[v]:
                    new_candidates.append(v)
                    
            if len(new_candidates) > 1:
                # Accept
                # Remove w_i
                code[i] = code[-1]
                code.pop()
                in_code[w_i] = 0
                
                # Add new candidates
                for v in new_candidates:
                    code.append(v)
                    in_code[v] = 1
                    for mask in all_masks:
                        block_count[v ^ mask] += 1
                        
                improved = True
            else:
                # Revert
                for v in neighbors:
                    block_count[v] += 1
                    
    return code
