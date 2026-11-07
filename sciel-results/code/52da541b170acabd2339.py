import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096 possible subsets
    
    # Precompute popcount
    popcount = [0] * total_masks
    for mask in range(1, total_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute strict subset/superset relationships using bitsets
    # For each mask, precompute all masks that are strict subsets
    # And all masks that are strict supersets
    # We'll use a different approach: for the 4096 masks, precompute
    # a bitset of which of the 4096 masks are strict subsets of each mask
    
    # Actually, for efficiency, let's precompute for each mask:
    # subset_count[mask] = number of masks that are strict subsets
    # But we need pairwise comparability for the 40 selected masks
    
    # Better: precompute a 4096x4096 bit matrix as a list of 4096 integers (bitsets)
    # bitset_subset[a] = bitmask over 4096 positions, bit j set if mask j is strict subset of mask a
    # This is 4096 * 512 bytes = 2MB, manageable
    
    # Actually, let's use a simpler approach: precompute comparability on the fly
    # but optimize the local search
    
    # Precompute: for each pair of masks, whether they're comparable
    # Use a 2D list of booleans: 4096x4096 = 16.7M booleans
    # That's about 16.7MB, acceptable
    
    # Actually, let's just use the direct check which is fast enough for local search
    
    def is_comparable(a, b):
        """Return True if a and b are comparable (one is strict subset of other)"""
        if a == b:
            return False
        if (a & b) == a:  # a is subset of b
            return True
        if (b & a) == b:  # b is subset of a
            return True
        return False
    
    def count_comparable(family):
        count = 0
        k = len(family)
        for i in range(k):
            a = family[i]
            for j in range(i + 1, k):
                b = family[j]
                if is_comparable(a, b):
                    count += 1
        return count
    
    # Precompute comparability matrix for all 4096 masks
    # comp[i][j] = True if masks i and j are comparable
    # This is 4096*4096/2 = 8.4M pairs, precompute once
    # Use a list of bytearrays for memory efficiency
    comp = [bytearray(total_masks) for _ in range(total_masks)]
    for i in range(total_masks):
        mi = i
        for j in range(i + 1, total_masks):
            mj = j
            if (mi & mj) == mi or (mi & mj) == mj:
                if mi != mj:
                    comp[i][j] = 1
                    comp[j][i] = 1
    
    def count_comparable_fast(family):
        count = 0
        k = len(family)
        for i in range(k):
            a = family[i]
            row = comp[a]
            for j in range(i + 1, k):
                if row[family[j]]:
                    count += 1
        return count
    
    # Precompute gain for adding a mask to a family
    # gain_of_adding(mask, family_set, family_list) = number of comparable pairs with existing family
    
    # For local search, we need:
    # - net gain of replacing family[i] with new_mask
    # This equals: (comparability of new_mask with all other family members) 
    #              - (comparability of family[i] with all other family members)
    
    # Precompute for each mask, a bitset of which masks it's comparable with
    # Then for a family, the degree of mask m = popcount of (comparable_bitset[m] & family_bitset)
    
    # Let's use bitsets for the family membership
    # family_bits: 4096-bit integer, bit i set if mask i is in family
    
    # comparable_bits[m]: 4096-bit integer, bit i set if mask m is comparable with mask i
    comparable_bits = [0] * total_masks
    for i in range(total_masks):
        bits = 0
        for j in range(total_masks):
            if i != j and comp[i][j]:
                bits |= (1 << j)
        comparable_bits[i] = bits
    
    # Now degree of mask m in family = popcount(comparable_bits[m] & family_bits)
    # Python int can handle 4096 bits natively
    
    def popcount_int(x):
        return x.bit_count()
    
    random.seed(42)
    
    # Start with a chain
    chain = [(1 << size) - 1 for size in range(13)]
    family = list(chain)
    family_set = set(family)
    family_bits = 0
    for mask in family:
        family_bits |= (1 << mask)
    
    # Greedily add 27 more masks
    candidates = list(range(total_masks))
    random.shuffle(candidates)
    candidates = [c for c in candidates if c not in family_set]
    
    for _ in range(m - len(family)):
        best_mask = None
        best_gain = -1
        for c in candidates[:2000]:
            gain = popcount_int(comparable_bits[c] & family_bits)
            if gain > best_gain:
                best_gain = gain
                best_mask = c
        if best_mask is not None:
            family.append(best_mask)
            family_set.add(best_mask)
            family_bits |= (1 << best_mask)
            candidates.remove(best_mask)
    
    best_score = count_comparable_fast(family)
    best_family = list(family)
    
    # Local search: single-move and double-move
    # Single move: replace family[i] with new_mask
    # Double move: replace family[i] and family[j] with new_mask1 and new_mask2
    
    def try_single_replacement_inplace(family, family_bits, family_set):
        """Try to improve by replacing one mask in-place. Returns True if improved."""
        k = len(family)
        for i in range(k):
            old_mask = family[i]
            old_degree = popcount_int(comparable_bits[old_mask] & (family_bits ^ (1 << old_mask)))
            
            best_new_mask = None
            best_new_degree = old_degree
            
            for c in range(total_masks):
                if c in family_set:
                    continue
                new_degree = popcount_int(comparable_bits[c] & (family_bits ^ (1 << old_mask)))
                if new_degree > best_new_degree:
                    best_new_degree = new_degree
                    best_new_mask = c
            
            if best_new_mask is not None:
                family[i] = best_new_mask
                family_set.remove(old_mask)
                family_set.add(best_new_mask)
                family_bits = family_bits ^ (1 << old_mask) | (1 << best_new_mask)
                return True
        
        return False
    
    # Multi-start local search
    restarts = 0
    no_improve_count = 0
    max_no_improve = 15
    
    while time.time() < deadline:
        restarts += 1
        no_improve_count = 0
        
        # Perturb the best solution
        if restarts == 1:
            family = list(best_family)
            family_set = set(family)
            family_bits = 0
            for mask in family:
                family_bits |= (1 << mask)
        else:
            family = list(best_family)
            family_set = set(family)
            family_bits = 0
            for mask in family:
                family_bits |= (1 << mask)
            
            # Randomly replace 2-6 masks
            num_replace = random.randint(2, 6)
            indices = random.sample(range(m), num_replace)
            for idx in indices:
                old_mask = family[idx]
                family_set.remove(old_mask)
                family_bits ^= (1 << old_mask)
                while True:
                    new_mask = random.randint(0, total_masks - 1)
                    if new_mask not in family_set:
                        break
                family[idx] = new_mask
                family_set.add(new_mask)
                family_bits |= (1 << new_mask)
        
        # Local search: repeatedly try single replacements until no improvement
        while time.time() < deadline - 1.0:
            improved = try_single_replacement_inplace(family, family_bits, family_set)
            if not improved:
                break
        
        score = count_comparable_fast(family)
        if score > best_score:
            best_score = score
            best_family = list(family)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve and restarts > 10:
                break
    
    # Final polish
    family = list(best_family)
    family_set = set(family)
    family_bits = 0
    for mask in family:
        family_bits |= (1 << mask)
    
    while time.time() < deadline - 0.5:
        improved = try_single_replacement_inplace(family, family_bits, family_set)
        if not improved:
            break
    
    score = count_comparable_fast(family)
    if score > best_score:
        best_score = score
        best_family = list(family)
    
    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
