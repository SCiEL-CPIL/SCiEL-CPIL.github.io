import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all words and neighbors
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for radius 1
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
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
                w_list[pos] = orig_val # Restore
        neighbors.append(neigh)
        
    # 2. Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    while np.any(uncovered):
        if time.time() > deadline - 20:
            break
            
        best_idx = -1
        max_gain = 0
        
        # Calculate gain for each candidate
        # Optimization: We can skip candidates that are already in code? 
        # Greedy set cover usually allows re-picking, but since we want distinct codewords, 
        # and picking the same one again doesn't help, we can restrict to unselected?
        # Actually, if we pick a word, it covers itself. If it's already in code, 
        # its gain is 0 for *newly* uncovered points? No, the points it covers are already covered.
        # So gain for existing code words is 0. We can just iterate all N.
        
        for i in range(N):
            # Sum of uncovered in neighbors[i]
            # Using numpy sum on a slice is fast
            gain = np.sum(uncovered[neighbors[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if max_gain == 0:
            break
            
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors[best_idx]] = False
        
    # 3. Local Search
    # Initialize counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors[idx]] += 1
        
    # Helper to check validity (counts >= 1 everywhere)
    # We maintain this invariant.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # --- Deletion Pass ---
        to_remove = []
        for c in code:
            # Check if all neighbors have count > 1
            # If min count in neighbors[c] is > 1, we can remove c
            if np.min(counts[neighbors[c]]) > 1:
                to_remove.append(c)
                
        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
            
        # --- Swap Pass ---
        # Try to replace codewords to enable more deletions
        # We iterate through code, and for each one, if it's not removable, 
        # we try to find a replacement.
        
        # To avoid bias, we can shuffle the code list order for the swap pass
        random.shuffle(code)
        
        for c in code:
            if time.time() > deadline - 5:
                break
                
            # Check if c is currently removable
            if np.min(counts[neighbors[c]]) > 1:
                # It's removable, but deletion pass should have caught it.
                # Maybe counts changed? No, we didn't change counts in this loop yet.
                # So this case shouldn't happen if deletion was thorough.
                continue
                
            # Get critical points for c: points i in neighbors[c] with counts[i] == 1
            crit_points = []
            for i in neighbors[c]:
                if counts[i] == 1:
                    crit_points.append(i)
            
            if not crit_points:
                continue
                
            # We need a replacement w such that all crit_points are in neighbors[w]
            # and w is not in code_set.
            
            # Strategy: Pick a random critical point p. 
            # Candidates for w must be in neighbors[p].
            # We sample a few candidates from neighbors[p] that are not in code.
            
            found_swap = False
            random.shuffle(crit_points)
            for p in crit_points:
                if found_swap:
                    break
                    
                # Get neighbors of p
                cands = [x for x in neighbors[p] if x not in code_set]
                
                # If too many, sample?
                # neighbors[p] size is 1 + n*(q-1) = 1 + 7*2 = 15.
                # So cands is small. We can check all of them.
                
                best_w = -1
                best_score = -1
                
                for w in cands:
                    # Check if w covers ALL crit_points
                    # i.e., for all k in crit_points, k is in neighbors[w]
                    # Since neighbors are lists, checking membership is O(len(neighbors[w])) ~ 15.
                    # crit_points size is small (usually 1 or 2).
                    
                    valid = True
                    for k in crit_points:
                        if k not in neighbors[w]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Valid replacement. Score it.
                    # We want to maximize the "redundancy" created.
                    # Score: Sum of counts in neighbors[w] that are >= 2?
                    # Or simpler: Just accept the first one?
                    # Let's try to pick the one that covers the most points with count==2.
                    score = 0
                    for i in neighbors[w]:
                        if counts[i] == 2:
                            score += 1
                            
                    if score > best_score:
                        best_score = score
                        best_w = w
                        
                if best_w != -1:
                    # Perform swap
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    
                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors[best_w]] += 1
                    
                    improved = True
                    found_swap = True
                    break # Move to next c
                    
    # Final cleanup: Remove any remaining redundant words
    # Re-initialize counts to be safe
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors[idx]] += 1
        
    changed = True
    while changed:
        changed = False
        for c in list(code):
            if np.min(counts[neighbors[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
                changed = True

    return [list(words[i]) for i in code]
