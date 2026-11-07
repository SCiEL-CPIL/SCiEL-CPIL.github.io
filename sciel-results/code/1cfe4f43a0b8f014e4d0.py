import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute popcount
    popcount = [0] * total_masks
    for mask in range(1, total_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute subset/superset relationships
    # For each mask, precompute which other masks it's comparable with
    # comparable[i][j] = True if i is strict subset of j or j is strict subset of i
    # This is a 4096x4096 boolean matrix - too large for dense storage
    # Instead, for each mask, store a bitset of masks it's comparable with
    
    # Actually, let's precompute for each mask, the list of masks that are strict supersets
    # and the list of masks that are strict subsets
    # But 4096*4096 is 16M entries - manageable with bitsets
    
    # Better approach: for each mask i, precompute a bitset (as Python int) where bit j is set
    # if i and j are comparable (one is strict subset of the other)
    # This uses 4096 big integers, each with 4096 bits = 4096 * 512 bytes = 2MB - fine
    
    # Precompute comparability bitsets
    # comp_bits[i] = bitset where bit j is 1 if i and j are comparable and i != j
    comp_bits = [0] * total_masks
    
    for i in range(total_masks):
        bits = 0
        # For each j, check if comparable
        # i is strict subset of j: (i & j) == i and i != j
        # j is strict subset of i: (j & i) == j and j != i
        # Combined: (i & j) == i and i != j) or ((j & i) == j and j != i)
        # Note: (i & j) == i means i is subset of j
        # (j & i) == j means j is subset of i
        # They can't both be true unless i == j
        for j in range(total_masks):
            if i == j:
                continue
            if (i & j) == i or (j & i) == j:
                bits |= (1 << j)
        comp_bits[i] = bits
    
    # Helper to count comparable pairs in a family
    def count_comparable(family):
        count = 0
        k = len(family)
        for i in range(k):
            for j in range(i + 1, k):
                if comp_bits[family[i]] & (1 << family[j]):
                    count += 1
        return count
    
    # Helper to compute gain of adding mask to family
    def gain_of_adding(mask, family):
        """Number of existing family members comparable to mask"""
        bits = comp_bits[mask]
        count = 0
        for f in family:
            if bits & (1 << f):
                count += 1
        return count
    
    # Helper to compute loss of removing mask from family
    def loss_of_removing(mask, family):
        """Number of other family members comparable to mask"""
        bits = comp_bits[mask]
        count = 0
        for f in family:
            if f != mask and (bits & (1 << f)):
                count += 1
        return count
    
    # Build initial solution using greedy approach
    # Start with a chain, then add greedily
    family = []
    in_family = set()
    
    # Add the main chain: 0, {0}, {0,1}, ..., {0..11}
    for size in range(13):
        mask = (1 << size) - 1
        family.append(mask)
        in_family.add(mask)
    
    # Greedily add remaining 27 sets
    while len(family) < m:
        best_mask = -1
        best_gain = -1
        for mask in range(total_masks):
            if mask in in_family:
                continue
            gain = gain_of_adding(mask, family)
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        if best_mask == -1:
            break
        family.append(best_mask)
        in_family.add(best_mask)
    
    best_score = count_comparable(family)
    best_family = list(family)
    
    # Exhaustive local search: try all single swaps
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        family = list(best_family)
        in_family = set(family)
        
        for i in range(m):
            if time.time() > deadline - 5.0:
                break
            
            old_mask = family[i]
            loss = loss_of_removing(old_mask, family)
            
            best_new_mask = -1
            best_delta = 0
            
            # Try all possible new masks
            for new_mask in range(total_masks):
                if new_mask in in_family:
                    continue
                gain = gain_of_adding(new_mask, family)
                # But we need to exclude the old mask from the family when computing gain
                # Actually, gain_of_adding counts comparable with ALL family members including old_mask
                # We need to subtract the comparability with old_mask if new_mask is comparable to old_mask
                # Let me recompute: gain should be with family minus old_mask
                # Let me just recompute properly
                
                # Recompute gain excluding old_mask
                gain_excl = 0
                bits = comp_bits[new_mask]
                for j in range(m):
                    if j == i:
                        continue
                    if bits & (1 << family[j]):
                        gain_excl += 1
                
                delta = gain_excl - loss
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_new_mask != -1 and best_delta > 0:
                in_family.remove(old_mask)
                family[i] = best_new_mask
                in_family.add(best_new_mask)
                best_family = list(family)
                best_score += best_delta
                improved = True
                break  # Restart the outer loop
    
    # Second pass: try 2-swaps if time allows
    if time.time() < deadline - 10.0:
        family = list(best_family)
        in_family = set(family)
        improved2 = True
        while improved2 and time.time() < deadline - 5.0:
            improved2 = False
            for i in range(m):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i + 1, m):
                    if time.time() > deadline - 5.0:
                        break
                    old1 = family[i]
                    old2 = family[j]
                    
                    # Loss of removing both
                    loss = 0
                    for k in range(m):
                        if k == i or k == j:
                            continue
                        if comp_bits[old1] & (1 << family[k]):
                            loss += 1
                        if comp_bits[old2] & (1 << family[k]):
                            loss += 1
                    # Also account for comparability between old1 and old2
                    if comp_bits[old1] & (1 << old2):
                        loss += 1
                    
                    best_new1 = -1
                    best_new2 = -1
                    best_delta = 0
                    
                    # Try pairs of new masks - this is expensive, so limit
                    # Sample some candidates
                    candidates = []
                    for _ in range(200):
                        c1 = random.randint(0, total_masks - 1)
                        c2 = random.randint(0, total_masks - 1)
                        if c1 not in in_family and c2 not in in_family and c1 != c2:
                            candidates.append((c1, c2))
                    
                    for c1, c2 in candidates:
                        gain = 0
                        for k in range(m):
                            if k == i or k == j:
                                continue
                            if comp_bits[c1] & (1 << family[k]):
                                gain += 1
                            if comp_bits[c2] & (1 << family[k]):
                                gain += 1
                        if comp_bits[c1] & (1 << c2):
                            gain += 1
                        
                        delta = gain - loss
                        if delta > best_delta:
                            best_delta = delta
                            best_new1 = c1
                            best_new2 = c2
                    
                    if best_new1 != -1 and best_delta > 0:
                        in_family.remove(old1)
                        in_family.remove(old2)
                        family[i] = best_new1
                        family[j] = best_new2
                        in_family.add(best_new1)
                        in_family.add(best_new2)
                        best_family = list(family)
                        best_score += best_delta
                        improved2 = True
                        break
                if improved2:
                    break
    
    # Convert to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    # Ensure types are correct
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
    
    return result
