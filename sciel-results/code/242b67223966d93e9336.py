"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from functools import lru_cache

def construct():
    # Compute budget guard
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    
    # We want to find a subset S of indices such that sum_{i in S} recip[i] < 1
    # and 1 - sum is minimized.
    
    # Dynamic Programming approach:
    # State: (index, current_sum)
    # We iterate through items 0..n-1.
    # For each item, we either take it or not.
    # We keep track of the best (minimum) deficit found for each reachable sum.
    # Actually, since we want to minimize 1 - sum, we want to maximize sum < 1.
    # So we can just keep a set of all reachable sums < 1.
    # Then find the max sum in that set.
    
    # The number of reachable sums might be large, but with 47 items and denominators up to 48,
    # the sums are rational numbers with denominators dividing LCM(2..48).
    # LCM(2..48) is huge, but the number of *distinct* sums < 1 formed by subset sums of these reciprocals
    # might be manageable if we prune or if the space isn't too dense.
    # However, worst case is 2^47 which is too big.
    # But we can use a set of Fractions.
    # Let's estimate the size.
    # The sum of all reciprocals is ~ H_48 - 1 ~ 4.38.
    # We only care about sums < 1.
    
    # Let's try a DP that stores reachable sums.
    # reachable_sums = set()
    # reachable_sums.add(Fraction(0))
    # for i in range(n):
    #     new_sums = set()
    #     for s in reachable_sums:
    #         if s + recip[i] < 1:
    #             new_sums.add(s + recip[i])
    #     reachable_sums.update(new_sums)
    
    # This is exponential in the worst case, but often prunes well.
    # Let's add a time check.
    
    reachable_sums = {Fraction(0)}
    
    for i in range(n):
        if time.time() > deadline:
            break
            
        new_sums = set()
        r_i = recip[i]
        # Iterate over a copy of reachable_sums because we are updating it?
        # No, we should generate new sums from the *previous* state.
        # So we iterate over the current set and add to a new set, then merge.
        # Or, to save memory/time, we can just add to the same set if we iterate carefully?
        # No, standard 0/1 knapsack DP on sets requires iterating over the snapshot.
        
        # Optimization: Only keep sums that are < 1.
        # Also, if we have a sum s, and s' > s, s' is better (closer to 1).
        # But we need all sums to potentially combine with future items.
        # Actually, if s1 < s2 < 1, s1 is dominated by s2 for the final objective,
        # BUT s1 might allow adding a large item that s2 cannot.
        # So we cannot simply prune smaller sums. We must keep all reachable sums < 1.
        
        for s in list(reachable_sums):
            ns = s + r_i
            if ns < 1:
                new_sums.add(ns)
        
        reachable_sums.update(new_sums)
        
        # Optional: Prune?
        # If the set gets too large, we might run out of time.
        # But let's see how large it gets.
        
    # Find the maximum sum in reachable_sums that is strictly less than 1.
    # We also need to recover the indices.
    # The set approach above doesn't track indices.
    # We need to track the path.
    
    # Let's restart with a DP that tracks the best deficit and the corresponding subset.
    # Since we can't easily backtrack from a set of sums, let's use a different DP structure.
    # DP[i][s] = True if sum s is reachable using first i items.
    # This is still hard with Fractions as keys in a 2D array.
    
    # Alternative: Meet-in-the-Middle?
    # Split 47 items into two halves: 23 and 24.
    # 2^23 ~ 8 million. 2^24 ~ 16 million.
    # This is feasible in Python if optimized, but 8M Fractions might be slow.
    # Let's try to generate all subset sums for the first half and store them in a list/dict.
    # Then for the second half, for each sum, we need to find the best complement from the first half.
    
    # Let's try Meet-in-the-Middle.
    if time.time() > deadline:
        # Fallback to simple greedy if MITM setup takes too long?
        # Actually, let's just implement MITM directly.
        
        n1 = n // 2
        n2 = n - n1
        
        indices1 = list(range(n1))
        indices2 = list(range(n1, n))
        
        # Generate all subset sums for group 1
        sums1 = {} # sum -> bitmask or indices
        # 2^23 is ~ 8.4M. This might be too slow in pure Python with Fractions.
        # Let's check time budget. 8.4M iterations of Fraction addition is likely > 10s.
        # Maybe too slow.
        
        # Let's stick to the Set DP but with a cap on the number of states?
        # Or use the property that we only care about the *maximum* sum < 1.
        
        # Let's try a different angle:
        # The deficit is 1 - sum.
        # We want to minimize deficit.
        
        # Let's go back to the Set DP but optimize it.
        # The number of reachable sums < 1 from 1/2..1/48.
        # Let's just run it and see. If it's too slow, we return the best found so far.
        
        # Re-implementing the set DP with time check is safer.
        # But we lost the indices.
        # We can store (sum, indices_tuple) ? That's heavy.
        # Or we can store sum -> one valid subset (as a bitmask int).
        # 47 bits fits in a Python int.
        
        reachable = {Fraction(0): 0} # sum -> bitmask
        best_deficit = Fraction(1)
        best_mask = 0
        
        for i in range(n):
            if time.time() > deadline:
                break
                
            r_i = recip[i]
            bit_i = 1 << i
            updates = {}
            
            for s, mask in list(reachable.items()):
                ns = s + r_i
                if ns < 1:
                    # Check if we already have a mask for this sum?
                    # If we do, we don't need to add it again, as the sum is the same.
                    # We just need one valid mask for each sum.
                    if ns not in updates:
                        updates[ns] = mask | bit_i
                        # Update best
                        deficit = 1 - ns
                        if deficit < best_deficit:
                            best_deficit = deficit
                            best_mask = mask | bit_i
            
            # Merge updates
            for ns, mask in updates.items():
                # If ns already exists in reachable, we keep the old mask (or new, doesn't matter for value)
                # But we must ensure we don't overwrite a sum with a different mask in a way that breaks future steps?
                # No, the sum value is what matters for future additions. The mask is just for reconstruction.
                # So if ns is already in reachable, we can skip adding it to updates?
                # Wait, if ns is in reachable, it means we could form this sum without item i.
                # Adding item i to a *different* path to form the same sum is redundant for future sums.
                # So we only need to store *one* mask per sum.
                if ns not in reachable:
                    reachable[ns] = mask
        
        # After loop, find the best sum in reachable
        # We already tracked best_deficit and best_mask during the loop.
        # But let's double check by iterating reachable at the end if time permits.
        
        if best_mask == 0:
             # Should not happen, 1/2 is always valid
             best_mask = 1 # index 0
             best_deficit = Fraction(1, 2)
        
        # Convert mask to indices
        best_indices = []
        for i in range(n):
            if best_mask & (1 << i):
                best_indices.append(i)
                
        if not best_indices:
            best_indices = [0]
            
        return best_indices

    # Fallback if we didn't enter the MITM/DP block (should not happen)
    return [0]
# EVOLVE-BLOCK-END
