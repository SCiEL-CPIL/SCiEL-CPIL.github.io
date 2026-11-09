import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total = len(all_masks)
    
    # Precompute intersection table: inter[i][j] = all_masks[i] & all_masks[j]
    # This is a 1820x1820 table, which is about 3.3M entries.
    # Each entry is an integer (up to 2^16), so ~26MB. Acceptable.
    # But we can be smarter: only store what we need.
    
    # Actually, let's precompute for each mask C, and for each possible intersection value I,
    # which family members have that intersection with C.
    # Better approach: for each pair (i,j), precompute inter[i][j]
    
    # Let's use a different strategy: maintain a "conflict" structure.
    # For a sunflower (A,B,C) with common core I, we need A&I = B&I = C&I = I,
    # and A&B = I, A&C = I, B&C = I.
    # So for a fixed C, and a fixed intersection I = A&C = B&C,
    # we need A&B = I.
    
    # Key insight: for each C, group existing family members by their intersection with C.
    # Within each group (same intersection I), check if any pair has mutual intersection = I.
    
    # Precompute: for each mask index i, and for each possible 4-bit pattern (intersection),
    # this doesn't quite work because intersections can be any subset of C.
    
    # Let's just optimize the existing approach:
    # 1. Use a set for the family for O(1) lookup
    # 2. Precompute pairwise intersections in a dict or 2D array
    
    # Precompute all pairwise intersections
    # inter[i][j] = all_masks[i] & all_masks[j]
    # Store as a list of lists
    inter = [[0] * total for _ in range(total)]
    for i in range(total):
        mi = all_masks[i]
        for j in range(i + 1, total):
            ij = mi & all_masks[j]
            inter[i][j] = ij
            inter[j][i] = ij
    
    # Mapping from mask value to index
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}
    
    def check_compatible(family_list, C, C_idx):
        """
        Check if C (at index C_idx) can be added to family_list without forming a sunflower.
        Returns True if compatible.
        """
        if len(family_list) < 2:
            return True
        
        # Bucket family members by their intersection with C
        buckets = {}
        for A in family_list:
            I = inter[C_idx][mask_to_idx[A]]
            if I not in buckets:
                buckets[I] = []
            buckets[I].append(A)
        
        # For each bucket, check pairs
        for I, bl in buckets.items():
            if len(bl) < 2:
                continue
            for ia in range(len(bl)):
                A = bl[ia]
                Ai = mask_to_idx[A]
                for ib in range(ia + 1, len(bl)):
                    B = bl[ib]
                    Bi = mask_to_idx[B]
                    if inter[Ai][Bi] == I:
                        return False
        return True

    def greedy_solve(order):
        family_list = []
        family_set = set()
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            C = all_masks[idx]
            if C in family_set:
                continue
            
            if check_compatible(family_list, C, idx):
                family_list.append(C)
                family_set.add(C)
        return family_list

    best_family = []
    best_size = 0
    
    # 1. Lexicographic Order
    order_lex = list(range(total))
    fam = greedy_solve(order_lex)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam
    
    # 2. Reversed Lexicographic Order
    order_rev = list(range(total))
    order_rev.reverse()
    fam = greedy_solve(order_rev)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam
    
    # 3. Multiple random orders
    max_restarts = 500
    rng = random.Random(42)
    
    for i in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        order = list(range(total))
        rng.shuffle(order)
        
        fam = greedy_solve(order)
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam
            if best_size >= 250:
                break
    
    # 4. Local Search / Kick-starts
    if time.time() < deadline - 5.0 and best_size > 0:
        for ks_round in range(100):
            if time.time() > deadline - 2.0:
                break
            
            if len(best_family) < 2:
                break
                
            remove_indices = rng.sample(range(len(best_family)), 2)
            
            new_family = [x for i, x in enumerate(best_family) if i not in remove_indices]
            new_family_set = set(new_family)
            
            # Build candidate list efficiently
            candidates = [m for m in all_masks if m not in new_family_set]
            rng.shuffle(candidates)
            
            improved_family = new_family[:]
            improved_set = new_family_set.copy()
            
            for C in candidates:
                if time.time() > deadline - 2.0:
                    break
                if C in improved_set:
                    continue
                
                C_idx = mask_to_idx[C]
                if check_compatible(improved_family, C, C_idx):
                    improved_family.append(C)
                    improved_set.add(C)
            
            if len(improved_family) > best_size:
                best_size = len(improved_family)
                best_family = improved_family
                if best_size >= 250:
                    break
    
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
    
    return result
