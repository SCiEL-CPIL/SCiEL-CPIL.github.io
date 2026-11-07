import numpy as np
import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 55  # Leave 5s margin for safety

    # Precompute comparability matrix
    # comp[i, j] = 1 if i is strict subset of j or j is strict subset of i
    # Note: i != j is implicit in strict subset, but we should ensure we don't count i with itself.
    # Actually, strict subset implies i != j.
    
    # Vectorized precomputation
    masks = np.arange(total_masks, dtype=np.int32)
    # Create a matrix where rows are masks, cols are bits
    # bits[i, b] = (masks[i] >> b) & 1
    bits = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    
    # subset[i, j] = 1 if masks[i] is subset of masks[j]
    # subset[i, j] = 1 if bits[i] & ~bits[j] == 0
    # This is equivalent to: for all b, if bits[i, b] == 1 then bits[j, b] == 1
    # Or: sum(bits[i] & (1 - bits[j])) == 0
    
    # To do this efficiently in numpy without 4096x4096x12 array:
    # We can iterate over bits? No, that's 12 passes of 4096x4096.
    # 12 * 16M operations is ~200M ops. Might be slow in pure numpy if not careful.
    
    # Alternative:
    # subset = np.zeros((total_masks, total_masks), dtype=np.int8)
    # for b in range(n):
    #     # masks[i] has bit b, masks[j] does not -> not subset
    #     # if bits[i, b] == 1 and bits[j, b] == 0, then i is NOT subset of j
    #     not_subset_b = bits[:, b][:, None] & (1 - bits[:, b][None, :])
    #     subset &= (not_subset_b == 0)
    
    # This loop is 12 iterations. Each iteration creates a 4096x4096 array.
    # 4096*4096 = 16,777,216 elements.
    # 12 * 16M = 200M elements processed.
    # This should take a few seconds in NumPy.
    
    subset = np.ones((total_masks, total_masks), dtype=np.int8)
    for b in range(n):
        # if i has bit b and j doesn't, i is not subset of j
        # mask_i_has_b = bits[:, b]
        # mask_j_lacks_b = 1 - bits[:, b]
        # not_subset = mask_i_has_b[:, None] * mask_j_lacks_b[None, :]
        not_subset_b = bits[:, b][:, None] & (1 - bits[:, b])[None, :]
        subset &= (not_subset_b == 0)
        
    # Strict subset: i != j
    # diag_indices = np.diag_indices(total_masks)
    # subset[diag_indices] = 0
    
    # Comparability: i subset j OR j subset i
    comp = subset | subset.T
    # Ensure diagonal is 0 (strict subset)
    comp[np.diag_indices(total_masks)] = 0
    
    # Convert to bool for faster logical ops if needed, or keep int8 for sums
    comp = comp.astype(np.int8)

    # Greedy Initialization
    family = []
    fam_set = set()
    
    # Start with empty set (mask 0)
    family.append(0)
    fam_set.add(0)
    
    # scores[s] = number of sets in family comparable to s
    scores = np.zeros(total_masks, dtype=np.int32)
    
    # Update scores for adding 0
    # 0 is subset of everything. So comp[0, :] is all 1s except 0.
    scores += comp[0]
    
    while len(family) < m and time.time() < deadline:
        # Find best mask not in fam_set with max score
        # To avoid picking existing ones, set their scores to -1
        # But we need to restore them? No, we just pick max from remaining.
        # Efficient way: keep a list of available masks? 
        # Or just loop. 4096 is small.
        
        # Mask out existing family members
        # We can create a boolean mask of available
        available = np.ones(total_masks, dtype=bool)
        for f in family:
            available[f] = False
            
        # Get scores of available
        avail_scores = scores[available]
        if len(avail_scores) == 0:
            break
            
        best_idx_in_avail = np.argmax(avail_scores)
        # Map back to original index
        # This mapping is O(N). 
        # Alternative: just loop over all masks and check if in fam_set.
        # 4096 iterations is very fast.
        
        best_mask = -1
        best_score = -1
        for s in range(total_masks):
            if s in fam_set:
                continue
            if scores[s] > best_score:
                best_score = scores[s]
                best_mask = s
                
        if best_mask == -1:
            break
            
        family.append(best_mask)
        fam_set.add(best_mask)
        scores += comp[best_mask]

    current_score = int(np.sum(scores) // 2) # Each pair counted twice
    best_family = family[:]
    best_score = current_score
    
    # Simulated Annealing / Local Search
    # We will perform swaps.
    # State: family (list of masks), fam_set (set of masks), scores (array)
    # Score calculation: sum(scores) // 2
    
    T = 10.0
    cooling = 0.99
    step = 0
    
    # Precompute indices for family to speed up updates?
    # When we swap A with B:
    # 1. Remove A: scores -= comp[A]
    # 2. Add B: scores += comp[B]
    # 3. Update family list and fam_set
    
    # To evaluate gain of swapping A with B:
    # Gain = (comp[B, F\A] - comp[A, F\A])
    # comp[X, F\A] = sum(comp[X, k] for k in F if k != A)
    # This requires summing 39 values.
    # Doing this for all 4096 B's is 4096 * 39 ops ~ 160k ops.
    # Doing this for all 40 A's is 40 * 160k = 6.4M ops per pass.
    # If we do 1000 passes, that's 6.4B ops. Too slow in Python loops.
    
    # Optimization:
    # We can vectorize the gain calculation for a fixed A.
    # Let F_idx be the indices of the family.
    # comp[:, F_idx] gives comparability of all masks with family members.
    # Sum along axis 1 to get comp_count for all masks against current family.
    # This is exactly `scores`!
    # So `scores` IS the count of comparable sets in the family for each mask.
    
    # Gain of swapping A (index i in family) with B:
    # Current contribution of A to total score: scores[A] (which is comp_count[A])
    # Note: scores[A] includes comparability with A? No, comp[A, A] = 0.
    # So scores[A] is exactly the number of other sets in F comparable to A.
    
    # If we remove A, the new scores for all masks X would be scores[X] - comp[X, A].
    # Then we add B. The new scores for all masks X would be (scores[X] - comp[X, A]) + comp[X, B].
    # The new total score is sum(new_scores) // 2.
    # Old total score = sum(scores) // 2.
    # Delta Score = (sum(new_scores) - sum(scores)) // 2
    # sum(new_scores) = sum(scores) - sum(comp[:, A]) + sum(comp[:, B])
    # Wait, sum(comp[:, A]) is the sum of comparabilities of ALL masks with A.
    # But we only care about the change in the total score of the FAMILY.
    
    # Let's look at the change in total pairs.
    # Pairs involving A: (A, X) for X in F, X != A. Count = scores[A].
    # Pairs involving B (new): (B, X) for X in F, X != A. Count = sum(comp[B, X] for X in F, X != A).
    # All other pairs remain unchanged.
    # So Delta Score = (Count_B - Count_A)
    # Count_A = scores[A]
    # Count_B = sum(comp[B, X] for X in F, X != A)
    
    # We need to calculate Count_B for all candidate B not in F.
    # Count_B = sum(comp[B, F]) - comp[B, A]
    # sum(comp[B, F]) is just scores[B] IF B was in F? No.
    # scores[B] is defined as sum(comp[B, X] for X in F).
    # So sum(comp[B, F]) = scores[B].
    # Therefore, Count_B = scores[B] - comp[B, A].
    
    # So Gain = (scores[B] - comp[B, A]) - scores[A].
    
    # This is O(1) per candidate B!
    # We just need to pick the B not in F that maximizes (scores[B] - comp[B, A]).
    
    # Algorithm per step:
    # 1. For each A in F:
    #    a. Calculate best_B = argmax_{B not in F} (scores[B] - comp[B, A])
    #    b. Calculate best_gain = max(...) - scores[A]
    # 2. If best_gain > 0 (or accept with prob), perform swap.
    
    # To find best_B efficiently:
    # We have `scores` (size 4096) and `comp[:, A]` (size 4096).
    # We want to maximize `scores - comp[:, A]` subject to `B not in F`.
    # We can compute `val = scores - comp[:, A]` for all 4096 masks.
    # Then set `val[B] = -inf` for B in F.
    # Then `best_B = argmax(val)`.
    
    # This is O(N) per A. Total O(M*N) per pass.
    # M=40, N=4096. 163k ops per pass.
    # This is very fast in NumPy.
    
    family_arr = np.array(family, dtype=np.int32)
    
    while time.time() < deadline and T > 0.01:
        improved = False
        
        # Shuffle family to avoid bias
        np.random.shuffle(family_arr)
        
        for i in range(len(family_arr)):
            A = int(family_arr[i])
            
            # Calculate values for all B
            # val[B] = scores[B] - comp[B, A]
            # Note: comp is symmetric, so comp[B, A] is comp[A, B]
            val = scores - comp[A] # comp[A] is row A, which is comp[A, :]
            
            # Mask out existing family members
            # We need to set val[family_arr] to a very low number
            # But we are iterating, so family_arr contains A.
            # We should exclude all current family members.
            
            # Creating a mask every time is O(N).
            # We can use a boolean array `in_fam`
            # But updating `in_fam` is O(1) per swap.
            
            # Let's maintain `in_fam` boolean array.
            # Initially:
            in_fam = np.zeros(total_masks, dtype=bool)
            for f in family:
                in_fam[f] = True
            
            # Inside loop:
            # We need to temporarily ignore A? 
            # No, B must not be in F. So B cannot be A either.
            # So we set val[in_fam] = -1000
            
            val[in_fam] = -1000
            
            best_B = int(np.argmax(val))
            best_val = val[best_B]
            
            if best_val == -1000:
                continue # No valid B found? Should not happen.
                
            # Current score of A in family
            score_A = scores[A]
            
            # Gain
            gain = best_val - score_A
            
            # Accept if gain > 0 or random
            accept = False
            if gain > 0:
                accept = True
            else:
                if random.random() < np.exp(gain / T):
                    accept = True
            
            if accept:
                # Perform swap
                # Remove A
                in_fam[A] = False
                scores -= comp[A]
                
                # Add B
                in_fam[best_B] = True
                scores += comp[best_B]
                
                # Update family array
                family_arr[i] = best_B
                
                # Update best
                new_score = int(np.sum(scores) // 2)
                if new_score > best_score:
                    best_score = new_score
                    best_family = family_arr.tolist()
                    improved = True
                    
        T *= cooling
        
    # Restore best
    family = best_family
    
    return _convert(family)

def _convert(family_masks):
    result = []
    for mask in family_masks:
        s = []
        for b in range(12):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
