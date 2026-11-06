import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Leave margin for 60s limit
    
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
        
    # blocked_count[i] is the number of codewords within distance < MIN_DIST of i
    blocked_count = np.zeros(1 << N, dtype=np.int32)
    
    code = []
    
    def add_word(w):
        code.append(w)
        for mask in all_masks:
            blocked_count[w ^ mask] += 1
            
    def remove_word(w):
        code.remove(w)
        for mask in all_masks:
            blocked_count[w ^ mask] -= 1

    # Phase 1: Greedy Construction to get the 256-word linear code (or close)
    # The greedy algorithm starting from 0 on this specific problem yields the 256-word code
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if blocked_count[w] == 0:
            add_word(w)
            
    # Phase 2: Local Search
    # We try to increase the size by replacing 1 word with 2.
    # We use vectorized NumPy operations for speed.
    
    random.seed(42)
    
    # Store the best code found
    best_code = list(code)
    best_size = len(code)
    
    no_improvement_count = 0
    max_no_improvement = 50 # Restart or try harder if stuck
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Shuffle codewords to try different removals
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
            
            c = code[idx]
            
            # Temporarily remove c
            for mask in all_masks:
                blocked_count[c ^ mask] -= 1
            
            # Find words that are now free (count == 0)
            # These are candidates for new words.
            # They must be in the neighborhood of c (distance < 6)
            # because only they were blocked by c.
            # Actually, any word with blocked_count == 0 is a candidate.
            # But words far away were already free and would have been added in greedy?
            # No, greedy adds in order. If a word was free, it would have been added.
            # So any newly free word must be one that was blocked by c.
            
            newly_free = []
            for mask in all_masks:
                n = c ^ mask
                if blocked_count[n] == 0:
                    newly_free.append(n)
            
            # Try to find two compatible words in newly_free
            if len(newly_free) >= 2:
                newly_free_arr = np.array(newly_free, dtype=np.uint32)
                success = False
                w1_idx = -1
                w2_idx = -1
                
                # Iterate through candidates for w1
                for i in range(len(newly_free_arr)):
                    w1 = newly_free_arr[i]
                    # Compute distances from w1 to all other candidates
                    # We only care about j > i to avoid duplicates and self
                    if i == len(newly_free_arr) - 1:
                        break
                    
                    diffs = w1 ^ newly_free_arr[i+1:]
                    dists = np.bit_count(diffs)
                    
                    # Find indices where distance >= MIN_DIST
                    valid_indices = np.where(dists >= MIN_DIST)[0]
                    
                    if len(valid_indices) > 0:
                        # Found a valid pair
                        w2 = newly_free_arr[i + 1 + valid_indices[0]]
                        w1_idx = i
                        w2_idx = i + 1 + valid_indices[0]
                        success = True
                        break
                        
                if success:
                    w1 = newly_free_arr[w1_idx]
                    w2 = newly_free_arr[w2_idx]
                    
                    # Perform the swap
                    # Remove c from code list
                    code.remove(c)
                    
                    # Add w1
                    code.append(w1)
                    for mask in all_masks:
                        blocked_count[w1 ^ mask] += 1
                        
                    # Add w2
                    code.append(w2)
                    for mask in all_masks:
                        blocked_count[w2 ^ mask] += 1
                    
                    improved = True
                    if len(code) > best_size:
                        best_size = len(code)
                        best_code = list(code)
                    
                    # Restart the loop to re-evaluate with new code
                    break
                else:
                    # No pair found, restore c
                    for mask in all_masks:
                        blocked_count[c ^ mask] += 1
            else:
                # Not enough candidates, restore c
                for mask in all_masks:
                    blocked_count[c ^ mask] += 1
            
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Try a "swap" move to perturb the code
                # Pick a random codeword c and a random free word w
                # such that d(c, w) >= 6.
                # This keeps size same but changes structure.
                c_idx = random.randint(0, len(code) - 1)
                c = code[c_idx]
                
                # Find a free word w
                # Sample some random words and check if they are free
                # and compatible with c (distance >= 6)
                swapped = False
                for _ in range(100):
                    w = random.randint(0, (1 << N) - 1)
                    if blocked_count[w] == 0 and np.bit_count(c ^ w) >= MIN_DIST:
                        # Check if w is compatible with ALL other codewords
                        # It is free, so it is compatible with all others.
                        # So we can just swap c with w.
                        
                        # Remove c
                        for mask in all_masks:
                            blocked_count[c ^ mask] -= 1
                        code.remove(c)
                        
                        # Add w
                        code.append(w)
                        for mask in all_masks:
                            blocked_count[w ^ mask] += 1
                        
                        swapped = True
                        break
                
                if not swapped:
                    # Stuck, break to save time or try random restart
                    break
                else:
                    no_improvement_count = 0 # Reset counter

    return best_code
