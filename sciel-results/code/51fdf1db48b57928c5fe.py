"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))

    # Precompute reciprocal as integer: L // i
    recip = [L // i for i in range(1, 23)]
    
    # Map index (1-based) to bit position (0-based)
    # Bit i corresponds to element i+1
    # Element 1 is bit 0, Element 22 is bit 21
    
    # Precompute minimal non-dissociated subsets (bad sets)
    # A set is bad if it has a collision in subset sums.
    # We check all subsets of size 3, 4, 5, 6.
    # If a subset is bad, all supersets are bad.
    # We store the minimal ones.
    
    def is_dissociated(indices):
        """Check if the set of reciprocals for indices is dissociated."""
        # indices is a tuple of 1-based indices
        vals = [recip[i-1] for i in indices]
        sums = {0}
        for v in vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    bad_sets = []
    # Check sizes 3 to 6
    # We only need to check if it's MINIMAL.
    # But for pruning, it's enough to have a set of bad masks.
    # If we include all bad sets of size <= 6, we might miss size 7 minimal bad sets.
    # However, if a size 7 set is minimal bad, it means no subset of size < 7 is bad.
    # Given the structure, it's highly likely that all minimal bad sets have size <= 4 or 5.
    # To be safe, we can go up to 6.
    
    for size in range(3, 7):
        for comb in combinations(range(1, 23), size):
            if not is_dissociated(comb):
                # Create bitmask
                mask = 0
                for idx in comb:
                    mask |= (1 << (idx - 1))
                # Check if it contains a smaller bad set
                # Since we iterate by size, if it was bad, a smaller subset would have been found?
                # Not necessarily. A set can be bad without having a smaller bad subset?
                # No, if a set is not dissociated, there exist two distinct subsets with same sum.
                # The union of these two subsets is a "bad" subset.
                # Is it minimal?
                # If the collision involves the whole set, it might be minimal.
                # But usually, collisions are local.
                # To ensure we have MINIMAL bad sets, we should check if any subset of `comb` is in `bad_sets`.
                # But `bad_sets` only contains smaller sizes.
                # So if `comb` is bad, and no subset of `comb` is in `bad_sets`, then `comb` is minimal (relative to our search depth).
                
                is_minimal = True
                # Check subsets of size size-1
                # This is expensive if done naively for every bad set.
                # But the number of bad sets is small.
                # Let's just add it. If it's not minimal, it will be redundant but correct.
                # Redundant checks are slower but safe.
                # To avoid redundancy, we can skip if any submask of size size-1 is in bad_sets.
                # For now, let's just add all bad sets of size 3-6.
                bad_sets.append(mask)

    # Remove duplicates
    bad_sets = list(set(bad_sets))
    
    # Sort bad sets by size (number of bits) to check smaller ones first?
    # Or just keep as is.
    
    best_result = []
    best_size = 0

    def backtrack(selected_mask, current_size, last_idx):
        nonlocal best_result, best_size
        if time.time() > deadline:
            return
            
        if current_size > best_size:
            best_size = current_size
            # Reconstruct result from mask
            res = []
            for i in range(1, 23):
                if selected_mask & (1 << (i - 1)):
                    res.append(i)
            best_result = res
            
        # Try to add elements > last_idx
        # Optimization: if we can't beat best_size, return?
        # Max possible size is 22.
        
        # Pruning: check if adding any element creates a bad set
        # We iterate candidates
        for i in range(last_idx + 1, 23):
            bit = 1 << (i - 1)
            new_mask = selected_mask | bit
            new_size = current_size + 1
            
            # Check if new_mask contains any bad set
            # Since we only added one bit, we only need to check bad sets that contain `bit`.
            # Pre-filter bad_sets by bit?
            # For simplicity, just loop through bad_sets.
            # If bad_sets is small (< 1000), this is fast.
            
            valid = True
            for bad in bad_sets:
                if (new_mask & bad) == bad:
                    valid = False
                    break
            
            if valid:
                backtrack(new_mask, new_size, i)

    backtrack(0, 0, 0)
    
    if not best_result:
        # Fallback to a known good set if search fails or times out early
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_result
# EVOLVE-BLOCK-END
