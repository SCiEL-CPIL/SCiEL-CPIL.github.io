"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _sunflower_core(s1, s2):
    """
    Returns the list of sets t such that {s1, s2, t} is a sunflower.
    s1, s2 are ints (bitmasks).
    """
    inter = s1 & s2
    inter_size = inter.bit_count()
    if inter_size > 4:
        return []
    # t must contain 'inter'
    # t must be disjoint from (s1 | s2) \ inter
    # s1 | s2 contains inter and the unique parts.
    # The available pool for the rest of t is [1..16] \ (s1 | s2)
    union = s1 | s2
    available_mask = 0xFFFF ^ union # 16 bits, 0xFFFF is 1..16 mask
    # We need to pick (4 - inter_size) elements from available_mask
    k_needed = 4 - inter_size
    if k_needed < 0:
        return []
    
    # Extract bits from available_mask
    avail_bits = []
    mask = available_mask
    while mask:
        lsb = mask & -mask
        # Get index (1-based)
        idx = lsb.bit_length()
        avail_bits.append(1 << idx) # Store as bitmask for the element? No, just the bit value
        mask ^= lsb
        
    # Actually, let's just store the bit values.
    # If we pick a set of bits J, t = inter | J.
    # We need to generate all combinations of avail_bits of size k_needed.
    
    results = []
    if k_needed == 0:
        results.append(inter)
    elif k_needed == 1:
        for b in avail_bits:
            results.append(inter | b)
    elif k_needed == 2:
        for i in range(len(avail_bits)):
            for j in range(i+1, len(avail_bits)):
                results.append(inter | avail_bits[i] | avail_bits[j])
    elif k_needed == 3:
        for i in range(len(avail_bits)):
            for j in range(i+1, len(avail_bits)):
                for k in range(j+1, len(avail_bits)):
                    results.append(inter | avail_bits[i] | avail_bits[j] | avail_bits[k])
    elif k_needed == 4:
        for i in range(len(avail_bits)):
            for j in range(i+1, len(avail_bits)):
                for k in range(j+1, len(avail_bits)):
                    for l in range(k+1, len(avail_bits)):
                        results.append(inter | avail_bits[i] | avail_bits[j] | avail_bits[k] | avail_bits[l])
                        
    return results

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << x)
        all_sets.append(mask)
    
    N = len(all_sets)
    # Map mask to index for quick lookup?
    # We need to update Count[t]. t is a mask.
    # We can use a dictionary or an array if we map masks to indices.
    # Since N is 1820, we can just iterate.
    # But updating Count[t] requires finding the index of t.
    # Let's create a map: mask_to_idx
    mask_to_idx = {m: i for i, m in enumerate(all_sets)}
    
    best_family = []
    best_size = 0
    
    # Run multiple greedy passes with different random orders
    for iteration in range(50): # Limit iterations to ensure time
        if time.time() > deadline - 5.0:
            break
            
        # Shuffle the order of consideration
        indices = list(range(N))
        random.shuffle(indices)
        
        family = []
        in_family = [False] * N
        # Count[i] is the number of pairs in family that form a sunflower with all_sets[i]
        count = [0] * N
        
        # We maintain a list of "available" indices where count[i] == 0
        # Initially all are 0.
        # As we add sets, counts increase.
        
        # To speed up finding a candidate, we can just iterate through indices
        # and check count.
        
        # Greedy Selection:
        # At each step, pick an available set t (count[t]==0) that minimizes the "cost"
        # Cost = number of new conflicts created.
        # New conflicts = sum over u in family of (number of t' such that {t, u, t'} is sunflower)
        # This is expensive to compute for all candidates.
        # Approximation: Pick the one with smallest current "degree" in the conflict graph?
        # Or just pick the first one available.
        
        # Let's try: Pick the first available one in the shuffled order.
        # This is fast.
        
        for idx in indices:
            if time.time() > deadline - 2.0:
                break
                
            if count[idx] > 0:
                continue
                
            s = all_sets[idx]
            # Add s to family
            family.append(s)
            in_family[idx] = True
            
            # Update counts for all other sets
            # For each u in family (excluding s itself, but s is the last one added)
            # We need to update counts for pairs (s, u) for all u in family[:-1]
            # Wait, the pair (s, u) creates conflicts for t.
            # The conflict sets are determined by s and u.
            
            for u in family[:-1]:
                # Get all t such that {s, u, t} is sunflower
                # These t's will have their count incremented.
                # Note: t could be s or u?
                # {s, u, t} distinct.
                # If t == s, {s, u, s} is not distinct.
                # Our _sunflower_core returns t such that t contains inter and is disjoint from union\inter.
                # If t == s, does s contain inter? Yes. Is s disjoint from (s|u)\inter?
                # (s|u)\inter = (s\inter) U (u\inter).
                # s is NOT disjoint from s\inter.
                # So s will not be generated by _sunflower_core(s, u) unless s\inter is empty?
                # If s\inter is empty, s = inter. Then s is subset of inter.
                # But _sunflower_core picks from available_mask which is disjoint from s|u.
                # So s is never in the result of _sunflower_core(s, u) because s intersects s|u.
                # Wait, available_mask = 0xFFFF ^ (s|u).
                # So any t generated is disjoint from s|u.
                # Thus t is distinct from s and u.
                
                conflicts = _sunflower_core(s, u)
                for t_mask in conflicts:
                    if t_mask in mask_to_idx:
                        t_idx = mask_to_idx[t_mask]
                        count[t_idx] += 1
                        # If count becomes 1, it is no longer available.
                        # We don't need to remove from a list if we just check count[idx]==0.
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # Optional: Local Search / Swap
        # Given time constraints, maybe just more restarts is better.
        # But let's try a simple swap if we have time.
        
    # Convert bitmasks back to lists of ints
    result = []
    for mask in best_family:
        elems = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length()
            elems.append(idx)
            m ^= lsb
        result.append(sorted(elems))
        
    return result
# EVOLVE-BLOCK-END
