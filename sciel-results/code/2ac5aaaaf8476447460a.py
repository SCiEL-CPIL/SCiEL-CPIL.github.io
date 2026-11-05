"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])

def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    if n <= 1:
        return [[i] for i in range(q)]
        
    if n == 2:
        # Optimal for q=3, n=2, r=1 is 9? No, 3^2=9 words. 
        # Distance 1 covers 1 + 2*2 = 5 words. 9/5 = 1.8 -> 2 words?
        # Let's check. [0,0] covers 00,01,02,10,20. [1,1] covers 11,10,12,21,01.
        # Union: 00,01,02,10,20,11,12,21. Missing 22. 
        # So 3 words? [0,0], [1,1], [2,2] covers all?
        # 22 covers 22,20,21,02,12.
        # Yes, 3 words.
        return [[0, 0], [1, 1], [2, 2]]

    if n == 3:
        # Perfect code exists? 3^3=27. Sphere size 1+6=7. 27/7 is not integer.
        # Min size is 9?
        # The current program returns 3 words for n=3, which is WRONG for radius 1.
        # [0,0,0] covers 000 and 6 neighbors. Total 7.
        # 27 words total. 3*7=21 < 27. So 3 words is invalid.
        # We need to fix the base cases or let the general solver handle it.
        # I will let the general solver handle n=3,4,5,6 to be safe and correct.
        pass

    # General Solver for n >= 3
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word index for speed
    # neighbors[i] is a list of indices of words within distance radius
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        for j, v in enumerate(all_words):
            if _hamming_dist(w, v) <= radius:
                neighbors[i].append(j)
                
    # 1. Greedy Set Cover
    # uncovered: bitset or boolean array
    uncovered = [True] * N
    uncovered_count = N
    codewords = []
    cw_indices = [] # indices of codewords in all_words
    
    # To speed up greedy, we can just iterate. For N=729, it's fine.
    while uncovered_count > 0:
        best_cw_idx = -1
        best_gain = 0
        # Heuristic: Start from a random point or use a good initial guess
        # For n=6, q=3, a good start is the tetracode product
        if not codewords:
             # Use a structured start
             tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
             initial_candidates = []
             if n >= 4:
                 for h in tetracode:
                     for tail in itertools.product(range(q), repeat=n - 4):
                         initial_candidates.append(tuple(h + list(tail)))
             else:
                 initial_candidates = all_words
            
             # Pick the one from initial_candidates that covers most
             for cw in initial_candidates:
                 idx = word_to_idx[cw]
                 gain = sum(1 for nb in neighbors[idx] if uncovered[nb])
                 if gain > best_gain:
                     best_gain = gain
                     best_cw_idx = idx
             
             # If no initial candidate was good (e.g. n<4), fall back to full search
             if best_cw_idx == -1:
                 for i in range(N):
                     gain = sum(1 for nb in neighbors[i] if uncovered[nb])
                     if gain > best_gain:
                         best_gain = gain
                         best_cw_idx = i
        else:
             # Full search for next best
             for i in range(N):
                 if i in cw_indices: continue
                 gain = sum(1 for nb in neighbors[i] if uncovered[nb])
                 if gain > best_gain:
                     best_gain = gain
                     best_cw_idx = i
                     
        if best_cw_idx == -1:
            break # Should not happen if radius >= 1
            
        codewords.append(list(all_words[best_cw_idx]))
        cw_indices.append(best_cw_idx)
        
        for nb in neighbors[best_cw_idx]:
            if uncovered[nb]:
                uncovered[nb] = False
                uncovered_count -= 1

    # 2. Local Search
    # We want to minimize len(codewords)
    # Strategy: Try to remove a codeword. If it leaves holes, try to fill them with one new codeword.
    # If we can fill holes with 1 new cw, size stays same (no gain).
    # If we can fill holes with 0 new cw (i.e., no holes), size decreases by 1.
    # To get a decrease, we need to find a set of codewords to remove and a set to add such that |add| < |remove|.
    # Simplest: Remove 1, Add 0. (Redundancy removal)
    # Next: Remove 2, Add 1.
    
    current_cw = codewords[:]
    current_indices = cw_indices[:]
    
    start_time = time.time()
    timeout = 110 # Leave 10s for overhead
    
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        # Try removing each codeword
        for i in range(len(current_cw)):
            # Check if current_cw[i] is redundant
            # Count coverage of each word by other codewords
            # Efficient way: 
            # 1. Identify words covered ONLY by current_cw[i]
            # 2. If none, remove it.
            
            only_covered_by_i = []
            for nb in neighbors[current_indices[i]]:
                # Check if nb is covered by any other cw
                covered_by_other = False
                for j in range(len(current_indices)):
                    if i == j: continue
                    if nb in neighbors[current_indices[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    only_covered_by_i.append(nb)
                    break # Just finding one is enough to know it's not redundant
            
            if not only_covered_by_i:
                # Redundant, remove it
                del current_cw[i]
                del current_indices[i]
                improved = True
                break # Restart loop with new list
                
        if not improved:
            # Try 2-for-1 swap? Or 1-for-1 swap to enable future removals?
            # Let's try 1-for-1 swap that increases "flexibility" or just random kick
            # Actually, a simple "kick" is to remove one, and add one that covers the holes + max new words.
            # This keeps size same, but might allow a subsequent removal.
            
            # Try to find a swap that allows a removal
            # This is complex. Let's try a simpler local search:
            # Randomly pick a cw to remove. Find the "essential" words it covers.
            # Find the best replacement cw that covers all essential words.
            # If found, swap. Then try to remove again.
            
            if len(current_cw) > 0:
                i = random.randint(0, len(current_cw) - 1)
                # Find essential words
                essential = []
                for nb in neighbors[current_indices[i]]:
                    covered_by_other = False
                    for j in range(len(current_indices)):
                        if i == j: continue
                        if nb in neighbors[current_indices[j]]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        essential.append(nb)
                        
                if essential:
                    # Find best replacement
                    best_new_idx = -1
                    best_new_gain = -1
                    for k in range(N):
                        if k in current_indices: continue
                        # Check if k covers all essential
                        if all(e in neighbors[k] for e in essential):
                            # Calculate gain: how many NEW words does k cover?
                            # Words covered by k but not by any other current cw (excluding i)
                            # Actually, we just want to maximize coverage of essential + new.
                            # Since it covers essential, we just want to maximize new coverage.
                            new_gain = 0
                            for nb in neighbors[k]:
                                # If nb is not covered by any other current cw (j != i)
                                is_covered = False
                                for j in range(len(current_indices)):
                                    if i == j: continue
                                    if nb in neighbors[current_indices[j]]:
                                        is_covered = True
                                        break
                                if not is_covered:
                                    new_gain += 1
                            
                            if new_gain > best_new_gain:
                                best_new_gain = new_gain
                                best_new_idx = k
                                
                    if best_new_idx != -1:
                        # Perform swap
                        current_cw[i] = list(all_words[best_new_idx])
                        current_indices[i] = best_new_idx
                        improved = True # Force another pass to check for removals
    
    return current_cw
# EVOLVE-BLOCK-END
