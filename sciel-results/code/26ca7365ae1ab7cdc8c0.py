import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55  # Safe margin for 60s limit
    
    # Precompute masks for distances 0 to 5
    # d=0: 1 mask
    # d=1: 17 masks
    # d=2: 136 masks
    # d=3: 680 masks
    # d=4: 2380 masks
    # d=5: 6188 masks
    # Total ~ 9400 masks. This is small.
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

    # Phase 1: Greedy Construction
    # Standard greedy adds words in increasing order.
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if blocked_count[w] == 0:
            add_word(w)
            
    # Phase 2: Local Search
    # Strategy: Try to replace 1 word with 2.
    # If successful, size increases by 1.
    # We also allow 1-for-1 swaps to escape local minima.
    
    random.seed(42)
    best_code = list(code)
    best_size = len(code)
    
    # Keep track of recent failures to avoid immediate re-attempting same moves
    # but simple random shuffle is often enough.
    
    no_improvement_count = 0
    max_no_improvement = 2000 # Allow many tries
    
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
            
            # Identify newly free words.
            # These are words w such that blocked_count[w] == 0 NOW, 
            # but we need to be careful: 
            # If a word was blocked by c and NO other, it is now 0.
            # If it was blocked by c and another, it is now 1.
            # So newly free = {w | blocked_count[w] == 0 and w was in the neighborhood of c}.
            # Actually, any word with blocked_count[w]==0 is a candidate for addition.
            # However, words far away from c might have been 0 all along?
            # No, if they were 0 all along, they would have been added in Greedy Phase 1 (if smaller index) 
            # or would have been added by previous swaps.
            # Wait, Greedy Phase 1 adds ALL free words in order. So at the end of Phase 1, 
            # there are NO free words with index < max_code_index? 
            # Not necessarily. But the set of free words is limited.
            
            # To be safe and fast, we only consider words in the neighborhood of c.
            # Why? Because only words blocked by c could have become free.
            # If a word w was free before removing c, it would have been added already.
            # So any *newly* free word must have been blocked by c.
            
            neighbors = c ^ all_masks
            # Filter to find those that are now free (count == 0)
            # Note: c itself has count 0 (distance 0 < 6, so it blocked itself? 
            # Yes, d(c,c)=0 < 6. So c is in neighbors and was blocked by itself?
            # Wait, add_word(c) increments blocked_count[c] by 1 (for mask 0).
            # So before removal, blocked_count[c] >= 1.
            # After removal, if no other codeword blocks c, blocked_count[c] becomes 0.
            # So c is a candidate to be re-added, but we want 2 NEW words.
            # We should exclude c from the candidate list for the "add 2" step, 
            # or handle it carefully.
            
            # Let's get the list of candidates
            counts = blocked_count[neighbors]
            # We want indices where count == 0
            free_indices = np.where(counts == 0)[0]
            
            if len(free_indices) < 2:
                # Not enough candidates to add 2. 
                # Try 1-for-1 swap? Or just revert.
                # For now, revert.
                for mask in all_masks:
                    blocked_count[c ^ mask] += 1
                continue
            
            candidates = neighbors[free_indices]
            
            # We want to find a pair (u, v) in candidates such that d(u, v) >= 6.
            # u != v.
            # Vectorized approach:
            # Compute pairwise distances? 
            # M = len(candidates). M can be large (up to ~9000).
            # M^2 bit_count is expensive if M is 9000 (81M ops).
            # We can do it in batches or use a smarter check.
            
            # Optimized check:
            # Iterate through candidates. For each u, check if any v in candidates (v != u) is compatible.
            # We can break early.
            
            success = False
            w1_idx = -1
            w2_idx = -1
            
            # Shuffle candidates to find diverse pairs?
            # random.shuffle modifies list, candidates is array.
            # Let's just iterate.
            
            for i in range(len(candidates)):
                if time.time() - start_time > time_limit:
                    break
                u = candidates[i]
                
                # Check compatibility with all j > i (to avoid duplicate pairs and self)
                if i == len(candidates) - 1:
                    break
                
                v_list = candidates[i+1:]
                diffs = u ^ v_list
                dists = np.bit_count(diffs)
                
                valid_indices = np.where(dists >= MIN_DIST)[0]
                
                if len(valid_indices) > 0:
                    # Found a valid pair
                    w2 = v_list[valid_indices[0]]
                    w1 = u
                    w1_idx = i
                    w2_idx = i + 1 + valid_indices[0] # Index in original candidates array? 
                    # Actually we just need the values w1, w2.
                    success = True
                    break
            
            if success:
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
                # Optional: Try 1-for-1 swap if we want to escape local minima
                # But 1-for-2 is the primary driver for size increase.
                # If we can't find 1-for-2, we are likely in a tight packing.
                # Let's stick to 1-for-2 for simplicity and speed.
                # We might add a "jiggle" step later if stuck for too long.

        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturbation: Random 1-for-1 swap to change structure
                # Pick a random codeword c and a random free word w (compatible with c? No, w must be free globally)
                # If we remove c, some words become free.
                # We can pick one of the newly free words w and swap c with w.
                
                c_idx = random.randint(0, len(code) - 1)
                c = code[c_idx]
                
                # Remove c
                for mask in all_masks:
                    blocked_count[c ^ mask] -= 1
                    
                neighbors = c ^ all_masks
                counts = blocked_count[neighbors]
                free_indices = np.where(counts == 0)[0]
                
                swapped = False
                if len(free_indices) > 0:
                    # Pick a random newly free word
                    # Exclude c itself? c is in neighbors (mask 0). 
                    # If we pick c, it's no swap.
                    # Filter out c?
                    valid_free = [neighbors[i] for i in free_indices if neighbors[i] != c]
                    if valid_free:
                        w = random.choice(valid_free)
                        # Add w
                        code.remove(c)
                        code.append(w)
                        for mask in all_masks:
                            blocked_count[w ^ mask] += 1
                        swapped = True
                    else:
                        # Only c was free? Revert
                        for mask in all_masks:
                            blocked_count[c ^ mask] += 1
                else:
                    for mask in all_masks:
                        blocked_count[c ^ mask] += 1
                
                if not swapped:
                    # Stuck completely? 
                    # Try a random restart? 
                    # Or just break to save time.
                    break
                else:
                    no_improvement_count = 0 # Reset counter

    return best_code
