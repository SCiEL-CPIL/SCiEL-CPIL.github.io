import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute nothing heavy, just use bitwise ops.
    # However, to speed up the "Gain" calculation, we can precompute
    # the list of all masks that are comparable to a given mask?
    # No, N=4096 is small enough to just loop.
    
    # Helper: Count comparable pairs in a family of masks
    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if (family[i] & family[j]) == family[i] or (family[i] & family[j]) == family[j]:
                    count += 1
        return count

    # Greedy Initialization
    # Start with a maximal chain to ensure high initial connectivity
    # Chain: 0, 1, 3, 7, ..., 4095
    chain = []
    cur = 0
    for b in range(n):
        chain.append(cur)
        cur |= (1 << b)
    chain.append(cur) # This adds the full set at the end if we started with 0 and added bits
    # Wait, the loop above:
    # b=0: appends 0, cur becomes 1
    # b=1: appends 1, cur becomes 3
    # ...
    # b=11: appends 2047, cur becomes 4095
    # Then appends 4095.
    # So chain has 13 elements: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095.
    # All are comparable.
    
    family = chain[:]
    fam_set = set(family)
    
    # Greedily add remaining m - len(family) sets
    while len(family) < m:
        if time.time() > deadline - 5:
            break
            
        best_mask = -1
        best_score = -1
        
        # Find mask M not in family maximizing comparability with current family
        for M in range(N):
            if M in fam_set:
                continue
            
            score = 0
            for f in family:
                if (f & M) == f or (f & M) == M:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_mask = M
        
        if best_mask == -1:
            break
            
        family.append(best_mask)
        fam_set.add(best_mask)
        
    # Local Search (Hill Climbing)
    # We will perform passes. In each pass, for each index i, we try to find
    # a replacement M that increases the total score.
    
    improved = True
    passes = 0
    while improved and time.time() < deadline - 2:
        improved = False
        passes += 1
        # Randomize order of indices to escape symmetry? Or just iterate.
        # Iterating in fixed order is fine for now.
        
        for i in range(m):
            if time.time() > deadline - 2:
                break
                
            old_mask = family[i]
            
            # Calculate Loss of old_mask (comparables with others)
            loss = 0
            for j in range(m):
                if j == i:
                    continue
                if (old_mask & family[j]) == old_mask or (old_mask & family[j]) == family[j]:
                    loss += 1
            
            best_mask = -1
            best_gain = -1
            
            # Try all candidates M not in family
            for M in range(N):
                if M in fam_set:
                    continue
                
                gain = 0
                for j in range(m):
                    if j == i:
                        continue
                    if (M & family[j]) == M or (M & family[j]) == family[j]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_mask = M
            
            # If we found a better replacement
            if best_mask != -1 and best_gain > loss:
                # Update family and fam_set
                fam_set.remove(old_mask)
                fam_set.add(best_mask)
                family[i] = best_mask
                improved = True
                
    # Convert to lists of integers
    result = []
    for mask in family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
