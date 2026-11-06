import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 1..16, so bits 1..16
    all_masks = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
        
    n_candidates = len(all_masks)
    max_size = 0
    best_family = []
    
    # Precompute bit counts? Python 3.10+ has int.bit_count()
    # If older, use bin(x).count("1")
    
    # Cache for submasks of size k for a given mask R
    # Key: (R, k), Value: list of masks S such that S subset R and popcount(S)=k
    submask_cache = {}
    
    def get_submasks(R, k):
        if k == 0:
            return [0]
        if k < 0:
            return []
        key = (R, k)
        if key in submask_cache:
            return submask_cache[key]
        
        # Extract bits of R
        bits = []
        r = R
        i = 0
        while r:
            if r & 1:
                bits.append(1 << i)
            i += 1
            r >>= 1
            
        res = []
        for c in combinations(bits, k):
            m = 0
            for b in c:
                m |= b
            res.append(m)
        submask_cache[key] = res
        return res

    # Run multiple randomized greedy attempts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.shuffle(all_masks)
        family = []
        family_set = set()
        
        for mask in all_masks:
            valid = True
            # Check against all pairs in current family
            # Optimization: Iterate through family, for each A, check if there exists B in family
            # such that (A, B, mask) is a sunflower.
            for A in family:
                C = A & mask
                c_size = C.bit_count()
                t_size = 4 - c_size
                
                # If t_size is 0, B must be C. But B must be distinct from A and mask.
                # If C == A, then A subset mask? No, A and mask are 4-sets.
                # If C == A, then A == mask? No, A and mask are distinct in the list?
                # The list contains distinct masks.
                # If A == mask, we wouldn't be checking.
                # So C != A and C != mask.
                
                R = ((1 << 17) - 2) & ~(A | mask) # Mask for bits 1..16 not in A or mask
                # Note: (1<<17)-2 is bits 1..16 set.
                
                # We need B such that B = C | S, S subset R, popcount(S) = t_size
                # And B in family_set.
                
                submasks = get_submasks(R, t_size)
                for S in submasks:
                    B = C | S
                    if B in family_set:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family.append(mask)
                family_set.add(mask)
                
        if len(family) > max_size:
            max_size = len(family)
            best_family = family.copy()
            
    # Convert best_family masks back to sorted lists of integers
    result = []
    for m in best_family:
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))
        
    return result
