import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total_masks = len(all_masks)
    
    # Precompute submasks for each mask to speed up compatibility check?
    # Submasks of a 4-bit mask are few (16). We can just generate them on the fly.
    
    def get_submasks(mask):
        sub = mask
        while True:
            yield sub
            if sub == 0: break
            sub = (sub - 1) & mask
            
    def is_compatible(S, family):
        # Group family members by their intersection with S
        # We need to check for each K (subset of S)
        # If there are A, B in family with A&S=K and B&S=K
        # such that (A\K) & (B\K) == 0, then conflict.
        
        # Optimization: 
        # Instead of iterating submasks of S and scanning family,
        # we can iterate family members, compute K = A & S, and store A in a dict grouped by K.
        groups = {}
        for A in family:
            K = A & S
            if K not in groups:
                groups[K] = []
            groups[K].append(A)
            
        # Now for each group, check for disjoint outside parts
        for K, group in groups.items():
            # If group size < 2, no pair
            if len(group) < 2:
                continue
            # Check pairs in group
            # (A & ~K) & (B & ~K) == 0
            # Note: ~K in Python is infinite bits, so we should mask with ALL_MASK or just use (A ^ K) & (B ^ K)?
            # A \ K is A & ~K. Since A only has bits in [0,15], A & ~K is safe if we consider ~K as mask of 1s where K is 0.
            # Actually, (A & ~K) is equivalent to (A ^ (A & K)) which is A ^ K since K is subset of A?
            # No, K is subset of S, but not necessarily A.
            # Wait, K = A & S. So K is subset of A.
            # So A \ K = A ^ K.
            # Let's verify: A=1100, S=1000, K=1000. A^K = 0100. A&~K = 1100 & ...0111 = 0100. Correct.
            # So we check (A ^ K) & (B ^ K) == 0.
            
            L = len(group)
            for i in range(L):
                Ai = group[i]
                Ai_out = Ai ^ K
                for j in range(i+1, L):
                    Bj = group[j]
                    Bj_out = Bj ^ K
                    if (Ai_out & Bj_out) == 0:
                        return False
        return True

    best_family = []
    rng = random.Random(42)
    
    # Run multiple random restarts
    while time.time() < deadline:
        shuffled = all_masks[:]
        rng.shuffle(shuffled)
        
        current_family = []
        for S in shuffled:
            if is_compatible(S, current_family):
                current_family.append(S)
        
        if len(current_family) > len(best_family):
            best_family = current_family[:]
            # If we reach the max target, we can stop? 
            # Max is 250. Total possible is 1820.
            # It's unlikely to hit 250 immediately, but possible.
            if len(best_family) >= 250:
                break

    # Convert best_family to list of sorted 1-indexed lists
    result = []
    for mask in best_family:
        elements = [i+1 for i in range(n) if (mask & (1 << i))]
        result.append(elements)
        
    return result
