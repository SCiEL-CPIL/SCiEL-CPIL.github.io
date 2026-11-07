import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability bitmasks
    # comp[i] is an integer with bit j set if i and j are comparable
    comp = [0] * total_masks
    
    # Optimization: 
    # i < j iff (i & j) == i and i != j
    # j < i iff (i & j) == j and i != j
    # We can compute this for all pairs.
    # 4096^2 = 16,777,216 pairs.
    # This loop might take ~1-2 seconds in Python.
    
    # To speed up, we can use the fact that if i is a subset of j, 
    # we can mark it.
    # But simple loop is okay if we are careful.
    
    # Let's try to optimize the precomputation.
    # For each i, find all subsets.
    # This is still O(3^n) or similar if done naively.
    # The double loop is O(4^n). 4^12 = 16M.
    
    # Let's run the double loop.
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i == j:
                continue
            # Check if i < j or j < i
            # i < j: (i & j) == i
            # j < i: (i & j) == j
            # Note: if i != j, they can't be equal, so one of these must hold if comparable?
            # No, they are incomparable if neither holds.
            # Example: 01 and 10. (01 & 10) = 00. 00 != 01, 00 != 10. Incomparable.
            if (i & j) == i or (i & j) == j:
                ci |= (1 << j)
        comp[i] = ci

    # Helper to count comparable pairs in a family given as list of masks
    def count_score(family):
        # Using precomputed comp
        score = 0
        # We can just sum popcount(comp[fam[i]] & mask_of_rest) / 2 ?
        # Or just iterate pairs. 40*40/2 = 800 pairs. Very fast.
        for i in range(m):
            for j in range(i + 1, m):
                if comp[family[i]] & (1 << family[j]):
                    score += 1
        return score

    # Local Search: Best Improvement
    def local_search(family):
        # family is a list of masks
        # We return the improved family and score
        current_score = count_score(family)
        
        # Precompute mask for each index? 
        # rest_mask[k] has bits for all family[j] where j != k
        
        # To speed up, we can maintain the mask of the whole family
        # and compute rest_mask[k] = full_mask ^ (1 << family[k])
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            full_mask = 0
            for mask in family:
                full_mask |= (1 << mask)
                
            for k in range(m):
                old_mask = family[k]
                rest_mask = full_mask ^ (1 << old_mask)
                
                best_cand = old_mask
                best_val = -1
                
                # Find candidate that maximizes comparability with rest
                # We iterate all 4096 masks
                # This is 40 * 4096 = 163840 checks per pass
                for cand in range(total_masks):
                    if cand == old_mask:
                        continue
                    # Check if cand is already in family? 
                    # If cand is in family, it's a duplicate. 
                    # We must maintain distinct sets.
                    # If cand is in rest_mask, it's already used.
                    if rest_mask & (1 << cand):
                        continue
                    
                    val = (comp[cand] & rest_mask).bit_count()
                    if val > best_val:
                        best_val = val
                        best_cand = cand
                
                # If we found a better candidate
                if best_cand != old_mask:
                    # Check if it actually improves the total score
                    # Old contribution of old_mask with rest:
                    old_val = (comp[old_mask] & rest_mask).bit_count()
                    
                    if best_val > old_val:
                        family[k] = best_cand
                        improved = True
                        # Update full_mask? 
                        # It will be recalculated at the start of the while loop iteration
                        # Or we can update it here for efficiency
                        # full_mask remains the same set of bits, just value changed? 
                        # No, full_mask is the set of masks present. 
                        # If we replace old_mask with best_cand, the set of masks changes.
                        # So full_mask must be updated.
                        # But we recalculate it at the top of the while loop.
                        # To save time, we can break out of for loop and restart while?
                        # Or just continue, the full_mask at top will be correct.
                        # However, if we change family[k], the rest_mask for subsequent k's in this pass
                        # will be based on the OLD full_mask.
                        # This is a "first improvement" or "synchronous" update issue.
                        # For simplicity and correctness, let's just restart the while loop 
                        # or accept that this pass uses stale masks for subsequent indices.
                        # Stale masks might lead to invalid states (duplicates) if not careful.
                        # The check `if rest_mask & (1 << cand)` prevents duplicates 
                        # relative to the OLD family.
                        # If we change family[k], family[k+1]'s rest_mask should exclude 
                        # the NEW family[k].
                        # To be safe and simple, let's just do one improvement per while iteration.
                        break 
            
            # If we broke out, loop again
            if time.time() > deadline - 5.0:
                break

        return family, count_score(family)

    # Strategy: Multiple random restarts
    best_family = None
    best_score = -1
    
    # Initial guess: A chain? 
    # Sizes 0, 1, ..., 12. We need 40.
    # We can pick all subsets of size <= 2? 
    # Number of subsets of size <= 2: 1 + 12 + 66 = 79.
    # Any two subsets of size <= 2 are comparable?
    # No. {0,1} and {2,3} are incomparable.
    # But many are comparable.
    
    # Let's just use random restarts.
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Random initialization
        family = random.sample(range(total_masks), m)
        
        # Run local search
        fam, score = local_search(family)
        
        if score > best_score:
            best_score = score
            best_family = fam
    
    # Convert to lists of integers
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
