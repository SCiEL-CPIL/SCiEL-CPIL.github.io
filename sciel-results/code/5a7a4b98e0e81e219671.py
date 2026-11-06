import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Safe margin for 60s limit
    
    # Precompute masks for distances 0 to 5
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
    all_masks = np.array(all_masks, dtype=np.uint32)
    
    # Precompute pairwise distances between all masks for faster pair checking
    # This will be a lookup table: dist_table[i][j] = Hamming distance between all_masks[i] and all_masks[j]
    # But that's 128^2 = 16384 entries, which is manageable
    # Actually, we need distance between actual words, not masks
    # Let's keep the current approach but optimize the pair search
    
    blocked_count = np.zeros(1 << N, dtype=np.int32)
    
    code = []
    code_set = set()
    
    def add_word(w):
        code.append(w)
        code_set.add(w)
        for mask in all_masks:
            blocked_count[w ^ mask] += 1
            
    def remove_word(w):
        code.remove(w)
        code_set.remove(w)
        for mask in all_masks:
            blocked_count[w ^ mask] -= 1

    # Phase 1: Greedy Construction
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if blocked_count[w] == 0:
            add_word(w)
            
    best_code = list(code)
    best_size = len(code)
    
    random.seed(42)
    
    # Phase 2: Simulated Annealing
    # Model the code as a spin glass system
    # Energy = -alpha * |code| + beta * (number of pairs with distance < MIN_DIST)
    # We want to minimize energy, so maximize |code| and minimize violations
    
    T_initial = 1.0
    T_final = 0.01
    cooling_rate = 0.9995
    
    T = T_initial
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Randomly select a codeword to remove
        idx = random.randint(0, len(code) - 1)
        c = code[idx]
        
        # Remove c
        remove_word(c)
        
        # Find newly free words in the neighborhood of c
        neighbors = c ^ all_masks
        counts = blocked_count[neighbors]
        free_indices = np.where(counts == 0)[0]
        
        if len(free_indices) == 0:
            # c was blocked by other codewords, revert
            add_word(c)
            continue
            
        candidates = neighbors[free_indices]
        
        # Try to find a pair (u, v) in candidates such that d(u, v) >= MIN_DIST
        found_pair = False
        w1 = -1
        w2 = -1
        
        M = len(candidates)
        for i in range(M):
            if time.time() - start_time > time_limit:
                break
            u = candidates[i]
            if i == M - 1:
                break
            v_list = candidates[i+1:]
            diffs = u ^ v_list
            dists = np.bit_count(diffs)
            valid_indices = np.where(dists >= MIN_DIST)[0]
            if len(valid_indices) > 0:
                w2 = v_list[valid_indices[0]]
                w1 = u
                found_pair = True
                break
        
        if found_pair:
            # Add both words - this is always an improvement
            add_word(w1)
            add_word(w2)
            
            if len(code) > best_size:
                best_size = len(code)
                best_code = list(code)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # No pair found. Try a 1-for-1 swap with simulated annealing acceptance
            # Pick a random candidate
            swap_idx = random.randint(0, len(free_indices) - 1)
            w_new = neighbors[free_indices[swap_idx]]
            
            if w_new == c:
                # Revert
                add_word(c)
            else:
                # Compute the energy change for the swap
                # Old state: c is in code, w_new is not
                # New state: w_new is in code, c is not
                # The energy change depends on how many new violations are created
                # For simplicity, we accept with probability exp(-delta_E / T)
                
                # Estimate delta_E: removing c might free some words, adding w_new might block some
                # A simple heuristic: accept with high probability early, low probability late
                # This mimics the annealing schedule
                
                if random.random() < T:
                    # Accept the swap
                    add_word(w_new)
                    no_improve_count += 1
                else:
                    # Reject, revert
                    add_word(c)
                
                # Cool down
                T = max(T_final, T * cooling_rate)
                
                # If we've been stuck for too long, reset temperature to explore more
                if no_improve_count > max_no_improve:
                    T = T_initial
                    no_improve_count = 0
                    # Also try a more aggressive restart: remove a few random codewords
                    for _ in range(3):
                        if len(code) > 10:
                            ridx = random.randint(0, len(code) - 1)
                            rc = code[ridx]
                            remove_word(rc)
                            # Try to add a new word from the freed neighborhood
                            rneighbors = rc ^ all_masks
                            rcounts = blocked_count[rneighbors]
                            rfree = np.where(rcounts == 0)[0]
                            if len(rfree) > 0:
                                ridx2 = random.randint(0, len(rfree) - 1)
                                rw = rneighbors[rfree[ridx2]]
                                if rw not in code_set:
                                    add_word(rw)
                    
    return best_code
