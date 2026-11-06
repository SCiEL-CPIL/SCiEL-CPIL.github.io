import time
import random
from collections import defaultdict

def construct():
    # Precompute all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    all_sets.append(mask)
    
    # Precompute submasks of size 0, 1, 2, 3 for each set
    # We need to check K such that K is subset of C and |K| < 4
    # Actually, K = A ∩ B. Since |A|=|B|=4, |K| can be 0,1,2,3.
    # If |K|=4, A=B, which is not allowed as distinct.
    submasks_cache = {}
    for mask in all_sets:
        subs = []
        # Iterate through all submasks of mask
        sub = mask
        while sub:
            if bin(sub).count('1') < 4:
                subs.append(sub)
            sub = (sub - 1) & mask
        subs.append(0) # Empty set
        submasks_cache[mask] = subs

    def greedy_run(random_seed):
        random.seed(random_seed)
        candidates = all_sets[:]
        random.shuffle(candidates)
        
        F = []
        # Pairs[K] stores list of (A_mask, B_mask)
        Pairs = defaultdict(list)
        
        for C in candidates:
            valid = True
            # Check against all existing pairs
            for K in submasks_cache[C]:
                if K in Pairs:
                    for A, B in Pairs[K]:
                        # Check if C forms a sunflower with A, B given core K
                        # Conditions:
                        # 1. K is subset of C (guaranteed by loop)
                        # 2. C is disjoint from A\K
                        # 3. C is disjoint from B\K
                        if (C & (A ^ K)) == 0 and (C & (B ^ K)) == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                F.append(C)
                # Update Pairs with new pairs involving C
                for A in F[:-1]:
                    K = A & C
                    Pairs[K].append((A, C))
        
        return F

    deadline = time.time() + 50.0
    best_family = []
    
    # Run multiple random greedy instances
    while time.time() < deadline:
        seed = random.randint(0, 10**9)
        current = greedy_run(seed)
        if len(current) > len(best_family):
            best_family = current
            # Optional: break if we hit a known high bound or max possible?
            # Max possible is 1820. If we get close, we can stop.
            if len(best_family) > 1500:
                break

    # Convert bitmasks back to sorted lists of 1-based indices
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
