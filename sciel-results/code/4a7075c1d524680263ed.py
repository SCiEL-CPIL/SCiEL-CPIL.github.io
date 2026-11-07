import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c.
    
    Strategy:
    1. Start with all integers in [129, 256] (size 128).
    2. Greedily remove the most-conflicted element until valid.
    3. Then perform local search with 2-for-1 swaps (remove one, try to add back
       removed elements) to escape local optima.
    4. Use bitsets for fast validity checking.
    """
    deadline = time.time() + 50  # 50 second safety margin
    
    # Precompute multiples for fast divisibility checks
    # multiples[k] is a bitmask where bit i is 1 if (i+2) is a multiple of k
    # We map value v to index v-2.
    # Range 2..256 -> indices 0..254.
    max_val = 256
    size = max_val - 2 + 1  # 255 elements
    multiples_mask = [0] * (max_val + 1)
    for k in range(1, max_val + 1):
        mask = 0
        for m in range(k, max_val + 1, k):
            if m >= 2:
                idx = m - 2
                mask |= (1 << idx)
        multiples_mask[k] = mask

    def is_valid_bitset(S_set, S_bitset):
        """
        Check validity using bitsets.
        S_set: set of integers in S.
        S_bitset: integer bitmask representing S.
        """
        S_list = list(S_set)
        n = len(S_list)
        # Map value to index for quick exclusion
        # We iterate over a, b in S
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                g = math.gcd(a, b)
                k = a // g
                if k == 1:
                    continue
                
                # We need c in S, c != a, c != b, such that c % k == 0
                # Candidates are multiples of k.
                candidates = multiples_mask[k]
                # Exclude a and b
                idx_a = a - 2
                idx_b = b - 2
                # Mask to keep only bits for a and b
                exclude_mask = (1 << idx_a) | (1 << idx_b)
                # We want to see if there is any bit in S_bitset that is in candidates AND NOT in exclude_mask
                # Actually, we want to check if (S_bitset & candidates) has any bit that is NOT a or b.
                # Since a and b are in S, they will be in S_bitset.
                # If the only multiples of k in S are a and b, then it's valid for this pair.
                # If there is a third one, it's invalid.
                
                common = S_bitset & candidates
                # If common has more than 2 bits set, or if the bits set are not exactly a and b (if they are multiples), it's invalid.
                # Simpler: Remove a and b from common. If anything remains, invalid.
                common_excluded = common & ~exclude_mask
                if common_excluded != 0:
                    return False
        return True

    # Start with all numbers in [129, 256]
    S = list(range(129, 257))
    S_set = set(S)
    
    # Initialize bitset
    S_bitset = 0
    for x in S:
        S_bitset |= (1 << (x - 2))

    def count_violations(S, S_set, S_bitset):
        """Count violations and track which elements are involved."""
        conflict_count = {x: 0 for x in S}
        S_list = list(S)
        n = len(S_list)
        
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                g = math.gcd(a, b)
                k = a // g
                if k == 1:
                    continue
                
                candidates = multiples_mask[k]
                idx_a = a - 2
                idx_b = b - 2
                exclude_mask = (1 << idx_a) | (1 << idx_b)
                
                common = S_bitset & candidates
                common_excluded = common & ~exclude_mask
                if common_excluded != 0:
                    # Conflict found involving a, b, and some c
                    # Count for a, b, and the specific c's?
                    # For simplicity, count for a and b.
                    conflict_count[a] += 1
                    conflict_count[b] += 1
                    # Identify c's to count them too?
                    # We can iterate bits in common_excluded
                    temp = common_excluded
                    while temp:
                        # Get lowest set bit
                        lsb = temp & -temp
                        idx_c = lsb.bit_length() - 1
                        c = idx_c + 2
                        conflict_count[c] += 1
                        temp ^= lsb
        return conflict_count

    # Phase 1: Greedy removal
    while time.time() < deadline:
        conflict_count = count_violations(S, S_set, S_bitset)
        
        max_conflicts = max(conflict_count.values()) if conflict_count else 0
        
        if max_conflicts == 0:
            break
        
        # Remove the element with the most conflicts
        worst_element = None
        max_conf = -1
        for x in S:
            if conflict_count[x] > max_conf:
                max_conf = conflict_count[x]
                worst_element = x
        
        if worst_element is not None:
            S.remove(worst_element)
            S_set.remove(worst_element)
            S_bitset &= ~(1 << (worst_element - 2))
        else:
            break
    
    # Phase 2: Local search with swaps
    removed = [x for x in range(129, 257) if x not in S_set]
    random.shuffle(removed)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        for x in removed[:]:
            if x in S_set:
                continue
            
            # Try to add x back by removing one element from S
            for y in S[:]:
                if time.time() > deadline:
                    break
                
                # Check if removing y and adding x gives a valid set
                # Create new bitset
                new_S_bitset = S_bitset
                new_S_bitset &= ~(1 << (y - 2))
                new_S_bitset |= (1 << (x - 2))
                
                # We need to check validity of the new set.
                # is_valid_bitset requires a set or list. 
                # For speed, we can just check the "diff" or re-run full check if small.
                # Given N~100, full check is fast.
                # But we need a set for the function signature.
                # Let's create a temporary set.
                new_S_set = S_set.copy()
                new_S_set.remove(y)
                new_S_set.add(x)
                
                if is_valid_bitset(new_S_set, new_S_bitset):
                    # Accept the swap
                    S = list(new_S_set)
                    S_set = new_S_set
                    S_bitset = new_S_bitset
                    removed.remove(x)
                    removed.append(y)
                    random.shuffle(removed)
                    improved = True
                    break
    
    # Final verification
    if not is_valid_bitset(S_set, S_bitset):
        # If somehow invalid, fall back to greedy removal again
        while not is_valid_bitset(S_set, S_bitset) and time.time() < deadline:
            conflict_count = count_violations(S, S_set, S_bitset)
            max_conflicts = max(conflict_count.values()) if conflict_count else 0
            if max_conflicts == 0:
                break
            worst_element = None
            max_conf = -1
            for x in S:
                if conflict_count[x] > max_conf:
                    max_conf = conflict_count[x]
                    worst_element = x
            if worst_element is not None:
                S.remove(worst_element)
                S_set.remove(worst_element)
                S_bitset &= ~(1 << (worst_element - 2))
            else:
                break
    
    return S
# EVOLVE-BLOCK-END
