import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 128
    
    # Key structural insight:
    # If S contains elements from [65,128], any two such elements have sum >= 131 > 128,
    # so they can never both be in a forbidden triple.
    # Forbidden triples can only be:
    #   Type A: three from [1,64]
    #   Type B: two from [1,64], one from [65,128]
    
    # Strategy: Use bit-parallel conflict masks for O(1) validity checks
    
    # Precompute for each x in [1,64]:
    # conflict_mask_L[x] = bitmask of y in [1,64] such that x+y is in [2,128]
    #   (i.e., if x is added, y cannot also be in L if x+y is in S)
    # Actually, we need to track which pairs (a,b) in L have a+b in S.
    
    # Let's use a different approach:
    # For each x in [1,64], precompute:
    #   small_partners[x] = set of y in [1,64] where y > x (to avoid duplicates)
    #   For Type B: for each x in [1,64] and c in [65,128], x+c is in [66,192]
    #     If x+c <= 128, then x+c is in U range
    
    # Bit-parallel approach:
    # Represent L as a 64-bit integer (bits 0-63 for values 1-64)
    # Represent U as a 64-bit integer (bits 0-63 for values 65-128)
    
    # For checking if adding x to L is valid:
    # Type A: For any a,b in L (including x), if a+b in S, then no c in L with a+c in S and b+c in S
    # Type B: For any a,b in L (including x) and c in U, if a+b in S and a+c in S and b+c in S, invalid
    
    # Precompute for each pair (i,j) with 1 <= i < j <= 64:
    #   pair_sum[i][j] = i+j (in range [3, 128])
    
    # For Type B, for each pair (i,j) in L and each c in U:
    #   Need i+j in S, i+c in S, j+c in S
    #   i+c in [66, 192], so i+c in S means i+c in U (if <= 128)
    
    # Let's precompute:
    # For each c in [65,128], which pairs (i,j) in [1,64] have i+c in U and j+c in U?
    # i+c in U means 65 <= i+c <= 128, so 65-c <= i <= 128-c
    # Since i >= 1, we need i >= max(1, 65-c) and i <= min(64, 128-c)
    
    # For fixed c, the valid i range is [max(1, 65-c), min(64, 128-c)]
    
    # Let's precompute for each c in [65,128]:
    #   c_valid_range[c] = (lo, hi) where lo = max(1, 65-c), hi = min(64, 128-c)
    #   If lo > hi, no valid i for this c
    
    # For the bit-parallel check when adding x to L:
    # We need to check all pairs involving x.
    # For Type A: pairs (x, a) where a in L, a != x
    #   If x+a in S, then for any b in L, b != x, b != a, we need NOT (x+b in S and a+b in S)
    # For Type B: pairs (x, a) where a in L, a != x, and c in U
    #   If x+a in S and x+c in S and a+c in S, invalid
    
    # This is still complex. Let's use a simpler but efficient approach:
    # Maintain L as a set, U as a set, S = L ∪ U
    # For check_add(x, L, U, S):
    #   Check all pairs (x, a) for a in L:
    #     Type A: if x+a in S, check if there exists b in L, b != x, b != a, with x+b in S and a+b in S
    #     Type B: if x+a in S, check if there exists c in U with x+c in S and a+c in S
    #   Also check pairs (a, b) for a, b in L, a < b (already checked when they were added,
    #   but adding x doesn't affect these unless x is involved)
    #   Actually, adding x only creates new forbidden triples involving x.
    #   So we only need to check triples where x is one of the three elements.
    
    # Triples involving x:
    #   (x, a, b) where a, b in L, a < b, a != x, b != x
    #     Need: x+a in S, x+b in S, a+b in S
    #   (x, a, c) where a in L, a != x, c in U
    #     Need: x+a in S, x+c in S, a+c in S
    
    # Precompute for efficiency:
    # For each x in [1,64] and each a in [1,64], a != x:
    #   x_plus_a = x + a
    # For each x in [1,64] and each c in [65,128]:
    #   x_plus_c = x + c (if <= 128, it's in U range)
    
    # Let's implement with sets and early termination
    
    def check_add(x, L_set, U_set, S_set):
        """Check if adding x to L creates any forbidden triple."""
        # Triples of form (x, a, b) with a, b in L, a < b
        for a in L_set:
            if a == x:
                continue
            xa = x + a
            if xa not in S_set:
                continue
            for b in L_set:
                if b == x or b == a:
                    continue
                if b > a:  # avoid duplicates
                    xb = x + b
                    ab = a + b
                    if xb in S_set and ab in S_set:
                        return False
        
        # Triples of form (x, a, c) with a in L, a != x, c in U
        for a in L_set:
            if a == x:
                continue
            xa = x + a
            if xa not in S_set:
                continue
            for c in U_set:
                xc = x + c
                ac = a + c
                if xc in S_set and ac in S_set:
                    return False
        
        return True
    
    best_size = 64
    best_result = set(range(65, 129))
    
    candidates = list(range(1, 65))
    
    # Multiple random greedy trials
    for trial in range(500):
        if time.time() - start_time > DEADLINE:
            break
        
        random.shuffle(candidates)
        L_set = set()
        U_set = set(range(65, 129))
        S_set = set(range(65, 129))
        
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if check_add(x, L_set, U_set, S_set):
                L_set.add(x)
                S_set.add(x)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_result = set(S_set)
    
    # Try with reduced U sets
    for trial in range(150):
        if time.time() - start_time > DEADLINE:
            break
        
        n_remove = random.randint(1, 8)
        U_remove = set(random.sample(range(65, 129), n_remove))
        U_set = set(range(65, 129)) - U_remove
        S_set = set(U_set)
        L_set = set()
        
        random.shuffle(candidates)
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if check_add(x, L_set, U_set, S_set):
                L_set.add(x)
                S_set.add(x)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_result = set(S_set)
    
    # Local search: try to add more L elements or swap
    if time.time() - start_time < DEADLINE:
        current = set(best_result)
        L_set = current & set(range(1, 65))
        U_set = current & set(range(65, 129))
        
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for x in range(1, 65):
                if x in current:
                    continue
                if check_add(x, L_set, U_set, current):
                    current.add(x)
                    L_set.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try removing one U element and adding L elements
                for u in sorted(U_set):
                    if time.time() - start_time > DEADLINE:
                        break
                    new_U = U_set - {u}
                    new_L = set(L_set)
                    new_S = current - {u}
                    added = True
                    while added and time.time() - start_time < DEADLINE:
                        added = False
                        for x in range(1, 65):
                            if x in new_S:
                                continue
                            if check_add(x, new_L, new_U, new_S):
                                new_L.add(x)
                                new_S.add(x)
                                added = True
                                break
                    if len(new_S) > len(current):
                        current = new_S
                        L_set = new_L
                        U_set = new_U
                        improved = True
                        break
    
    result = sorted(current)
    result = [int(x) for x in result]
    return result
