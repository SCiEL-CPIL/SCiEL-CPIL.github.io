import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors
    neighbor_indices = []
    neighbor_masks = []
    
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbor_indices.append(np.array(neigh, dtype=np.int32))
        mask = 0
        for j in neigh:
            mask |= (1 << j)
        neighbor_masks.append(mask)
        
    def greedy_init():
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 30:
                break
            gains = np.zeros(N, dtype=np.int32)
            for i in range(N):
                if i in code_set:
                    continue
                gains[i] = np.sum(uncovered[neighbor_indices[i]])
            
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
                
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbor_indices[best_idx]] = False
            
        return code, code_set

    best_code = None
    best_len = float('inf')
    
    # Multiple greedy restarts
    for _ in range(10):
        if time.time() > deadline - 50:
            break
        code, code_set = greedy_init()
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            best_code_set = code_set
            
    code = best_code
    code_set = best_code_set

    # Initialize counts and count_2_mask
    counts = np.zeros(N, dtype=np.int32)
    count_2_mask = 0
    for idx in code:
        counts[neighbor_indices[idx]] += 1
    
    # Build initial count_2_mask
    for i in range(N):
        if counts[i] == 2:
            count_2_mask |= (1 << i)

    def update_counts_and_mask(idx, delta):
        # delta is +1 or -1
        nonlocal count_2_mask
        for p in neighbor_indices[idx]:
            old_count = counts[p]
            new_count = old_count + delta
            counts[p] = new_count
            
            if old_count == 2 and new_count != 2:
                count_2_mask &= ~(1 << p)
            elif old_count != 2 and new_count == 2:
                count_2_mask |= (1 << p)

    # Local Search
    improved = True
    stagnation_count = 0
    max_stagnation = 5
    
    while time.time() < deadline - 5:
        improved = False
        
        # --- Deletion Pass ---
        changed = True
        while changed:
            changed = False
            to_remove = []
            for c in code:
                if np.min(counts[neighbor_indices[c]]) > 1:
                    to_remove.append(c)
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    update_counts_and_mask(c, -1)
                    changed = True

        # --- Swap Pass ---
        random.shuffle(code)
        for c in code:
            if time.time() > deadline - 5:
                break
            
            # Get critical points for c
            crit_mask = 0
            for p in neighbor_indices[c]:
                if counts[p] == 1:
                    crit_mask |= (1 << p)
            
            if crit_mask == 0:
                continue
                
            best_w = -1
            best_score = -1
            
            for w in range(N):
                if w in code_set:
                    continue
                
                if (neighbor_masks[w] & crit_mask) != crit_mask:
                    continue
                
                score = (neighbor_masks[w] & count_2_mask).bit_count()
                
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                update_counts_and_mask(c, -1)
                
                code.append(best_w)
                code_set.add(best_w)
                update_counts_and_mask(best_w, 1)
                
                improved = True
                stagnation_count = 0
                break # Restart outer loop? Or continue? 
                # Continuing might be better to find more swaps in the same pass
                # But counts changed, so crit_mask for other c's is invalid.
                # Let's break and restart the whole local search loop.
        
        if not improved:
            stagnation_count += 1
            if stagnation_count >= max_stagnation:
                # Perturbation: Add a random codeword
                # Choose a random word not in code
                available = [i for i in range(N) if i not in code_set]
                if available:
                    rand_w = random.choice(available)
                    code.append(rand_w)
                    code_set.add(rand_w)
                    update_counts_and_mask(rand_w, 1)
                    # Run deletion to clean up
                    changed = True
                    while changed:
                        changed = False
                        to_remove = []
                        for c in code:
                            if np.min(counts[neighbor_indices[c]]) > 1:
                                to_remove.append(c)
                        if to_remove:
                            for c in to_remove:
                                code.remove(c)
                                code_set.remove(c)
                                update_counts_and_mask(c, -1)
                                changed = True
                    stagnation_count = 0
                else:
                    break

    # Final cleanup
    changed = True
    while changed:
        changed = False
        to_remove = []
        for c in code:
            if np.min(counts[neighbor_indices[c]]) > 1:
                to_remove.append(c)
        if to_remove:
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                update_counts_and_mask(c, -1)
                changed = True

    return [list(words[i]) for i in code]
