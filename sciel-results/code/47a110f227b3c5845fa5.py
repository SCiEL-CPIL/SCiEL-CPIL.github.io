import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    
    # Precompute conflict masks for O(1) validity checks
    # For each element x, precompute:
    # 1. pair_sums[x]: bitmask of all elements that are pair sums involving x
    # 2. partner_mask[x]: for each other element y, the bitmask of z such that {x,y,z} is a forbidden triple
    # 
    # A triple (a,b,c) is forbidden if a+b, a+c, b+c are all in the set.
    # When adding x to set S, we need to check:
    # - x as a vertex: exists a,b in S\{x} with a+x in S, b+x in S, and a+b in S (or a+b==x)
    # - x as a sum: x=a+b for a,b in S, and exists c in S\{a,b} with a+c in S and b+c in S
    #
    # Precomputation:
    # For each pair (a,b) with a<b, compute the set of c values that would make {a,b,c} forbidden.
    # c is forbidden if: (a+c in [1,128] and b+c in [1,128] and a+b in [1,128])
    #   AND (a+c != c, b+c != c, a+b != c) -- but these are automatically satisfied for positive ints
    # Actually the constraint is: a+b, a+c, b+c must ALL be in the selected set.
    # The six values (a,b,c,a+b,a+c,b+c) need not be mutually distinct.
    # 
    # For a fixed pair (a,b), c is a "conflict" if:
    #   a+c <= 128 AND b+c <= 128 AND a+b <= 128
    #   AND a+c != a, b+c != b, a+b != a, a+b != b (always true for positive)
    #   AND c != a, c != b (distinct members)
    # So for pair (a,b), the conflicting c's are those where a+c<=128 and b+c<=128 and a+b<=128.
    # i.e., c <= 128 - max(a,b) and a+b <= 128.
    
    # Precompute: for each x, a bitmask conflict_with[x] such that:
    # If we have current set S (as bitmask), then x can be added iff:
    #   There do NOT exist two elements a,b in S (a != b, a != x, b != x) such that
    #   a+b in S, a+c in S, b+c in S where c=x... 
    # 
    # Let me re-derive. Forbidden condition: three DISTINCT members a,b,c in S
    # such that a+b in S, a+c in S, b+c in S.
    # 
    # When adding x to S, a violation occurs iff:
    # Case A: x is one of {a,b,c}. WLOG x=c. Then need a,b in S, a!=b, a!=x, b!=x,
    #         with a+x in S, b+x in S, and a+b in S.
    # Case B: x is one of {a+b, a+c, b+c}. WLOG x=a+b. Then need a,b in S, a!=b,
    #         a!=x, b!=x, and exists c in S, c!=a, c!=b, c!=x, with a+c in S and b+c in S.
    # 
    # For Case A with x as c:
    #   Let C_x = {a in S : a+x in S} (i.e., a in S and a+x in S)
    #   We need two distinct a,b in C_x (a!=b, both != x) such that a+b in S or a+b==x.
    #   C_x as bitmask: for a from 1 to 128-x, bit (a-1) is set in C_x iff bits (a-1) and (a+x-1) are set in S.
    #   C_x_mask = S & (S >> x)  [shifted right by x positions]
    #   But we need a+x <= 128, so a <= 128-x. The mask S >> x naturally handles this.
    #   Also exclude a=x: clear bit (x-1) in C_x_mask.
    #   Then check if there exist two distinct a,b in C_x_mask with a+b in S or a+b==x.
    # 
    # For Case B with x as a+b:
    #   For each pair (a,b) with a<b, a+b==x, a,b in S:
    #     C_a = {c in S : a+c in S} = S & (S >> a)
    #     C_b = {c in S : b+c in S} = S & (S >> b)
    #     potential_c = C_a & C_b, excluding a, b, x
    #     If potential_c != 0, violation.
    
    # For speed, let's precompute for each x:
    # caseA_candidates[x]: list of (a, b, a+b) triples where adding x as c creates conflict
    #   i.e., a+x<=128, b+x<=128, a!=b, a!=x, b!=x, and a+b<=128
    #   Actually we need to check against current S, so we can't fully precompute.
    # 
    # Alternative approach: precompute for each x, the list of "witness pairs" (a,b) 
    # such that if a,b,x are all in S, then a violation might occur (need also a+x, b+x, a+b in S).
    # But that's still complex.
    # 
    # Let me use a different strategy: precompute for each x, a 128-bit mask 
    # partner_conflict[x][y] = bitmask of z such that {x,y,z} would be a forbidden triple
    # if all of x,y,z,x+y,x+z,y+z are in the set.
    # 
    # For fixed x,y (x!=y), z is a conflict partner if:
    #   z != x, z != y
    #   x+z <= 128, y+z <= 128, x+y <= 128
    #   (the values x+y, x+z, y+z just need to be in [1,128] to potentially be in S)
    # 
    # So for fixed x,y: z must satisfy z <= 128 - max(x,y) and x+y <= 128 and z != x and z != y.
    # 
    # Precompute conflict_partner[x][y] as a bitmask of all valid z.
    # Then when checking if we can add x to S:
    #   For each y in S (y != x):
    #     If x+y <= 128:
    #       potential_z = conflict_partner[x][y] & S  (z in S that conflict with x,y)
    #       If potential_z has any bit set, adding x causes violation.
    #   Also need to check: for each y in S, if x+y in S, then for each z in S:
    #     if x+z in S and y+z in S and x+y in S -> violation (this is the same check)
    # 
    # Wait, I need to be more careful. The forbidden condition is:
    # a,b,c distinct in S, AND a+b in S, AND a+c in S, AND b+c in S.
    # 
    # When adding x to S, violation iff there exist distinct a,b,c all in S∪{x}
    # (with x being one of them or one of the sums) satisfying the condition.
    # 
    # Sub-case 1: x is a vertex. x ∈ {a,b,c}. WLOG c=x.
    #   Need: a,b ∈ S, a≠b, a≠x, b≠x, a+x∈S, b+x∈S, a+b∈S.
    #   For each a ∈ S with a+x≤128 and a+x∈S:
    #     For each b ∈ S with b≠a, b≠x, b+x≤128, b+x∈S:
    #       Check if a+b ∈ S or a+b == x.
    #   This is O(|S|²) per addition, too slow.
    # 
    # Let me think about precomputation differently.
    # 
    # For each x, precompute:
    #   neighbors[x] = {y : x+y ≤ 128}  -- elements that can form a sum with x
    # 
    # For the validity check when adding x to S:
    #   C = {y ∈ S : y+x ∈ S}  (y and y+x both in S, y ≠ x)
    #   As bitmask: C_mask = S & (S >> x), then clear bit (x-1) if set.
    #   Then check: do there exist distinct a,b ∈ C with a+b ∈ S or a+b == x?
    #   This requires checking pairs in C, which is O(|C|²).
    # 
    # To make this O(1), I can precompute for each x:
    #   sum_check[x] = bitmask of all s such that there exist a,b with a+b=s, a+x≤128, b+x≤128, a≠b, a≠x, b≠x
    #   Then: violation in Case A iff (C_mask has ≥2 bits) AND (S & sum_check[x] ≠ 0) OR (x in C_mask pairs...)
    #   Hmm, this doesn't quite work because we need a,b to BOTH be in C.
    # 
    # OK let me just optimize the existing approach with better precomputation and faster loops.
    
    # Precompute for each x: list of y such that x+y <= 128
    # And for each x: the mask of valid y's
    valid_partner_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        for y in range(1, n + 1):
            if y != x and x + y <= n:
                mask |= (1 << (y - 1))
        valid_partner_mask[x] = mask
    
    # Precompute for each pair (x,y) with x<y, x+y<=128: the mask of z such that
    # {x,y,z} is a potential forbidden triple (z distinct from x,y, x+z<=128, y+z<=128)
    # z_mask[x][y] = bitmask of valid z's
    z_conflict = {}
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            s = x + y
            if s > n:
                continue
            mask = 0
            max_z = n - max(x, y)
            for z in range(1, max_z + 1):
                if z != x and z != y:
                    mask |= (1 << (z - 1))
            if mask:
                z_conflict[(x, y)] = mask
    
    def popcount(mask):
        return bin(mask).count('1')
    
    def is_valid_addition(mask, x):
        """Check if x can be added to set S (represented by mask). O(|S|) or better."""
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Case A: x is a vertex (c=x). Need a,b in S, a!=b, a!=x, b!=x,
        #         a+x in S, b+x in S, a+b in S or a+b==x.
        # C = {a in S : a+x in S, a != x}
        C_mask = mask & (mask >> x)
        C_mask &= ~bit_x  # exclude x from C
        
        # Get candidates from C_mask
        candidates = []
        temp = C_mask
        while temp:
            low_bit = temp & (-temp)
            idx = low_bit.bit_length() - 1
            a = idx + 1
            candidates.append(a)
            temp ^= low_bit
        
        k = len(candidates)
        if k >= 2:
            for i in range(k):
                a = candidates[i]
                for j in range(i + 1, k):
                    b = candidates[j]
                    s = a + b
                    if s == x:
                        return False
                    if s <= n and (mask >> (s - 1)) & 1:
                        return False
        
        # Case B: x is a sum. x = a+b for some a,b in S.
        for a in range(1, x // 2 + 1):
            b = x - a
            if b <= a:
                continue
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                # Need c in S\{a,b,x} such that a+c in S and b+c in S
                C_a = mask & (mask >> a)
                C_b = mask & (mask >> b)
                potential_c = C_a & C_b
                potential_c &= ~(1 << (a - 1))
                potential_c &= ~(1 << (b - 1))
                potential_c &= ~bit_x
                if potential_c != 0:
                    return False
        
        return True
    
    def greedy_fill(mask):
        """Greedily add elements in order."""
        for x in range(1, n + 1):
            if not (mask >> (x - 1)) & 1:
                if is_valid_addition(mask, x):
                    mask |= (1 << (x - 1))
        return mask
    
    def local_search(initial_mask, time_limit):
        """Hill climbing with random restarts and simulated annealing."""
        current = initial_mask
        best = initial_mask
        best_size = popcount(best)
        current_size = popcount(current)
        
        T = 0.5
        cooling = 0.999
        min_T = 0.01
        end_time = time.time() + time_limit
        no_improve_count = 0
        max_no_improve = 50000
        
        while time.time() < end_time and T > min_T:
            # Random move
            x = random.randint(1, n)
            bit_x = 1 << (x - 1)
            
            if (current >> (x - 1)) & 1:
                # Remove x
                new_mask = current & ~bit_x
                new_size = current_size - 1
                delta = -1
            else:
                # Try to add x
                if is_valid_addition(current, x):
                    new_mask = current | bit_x
                    new_size = current_size + 1
                    delta = 1
                else:
                    # Try removing a random element
                    idx = random.randint(0, n - 1)
                    if (current >> idx) & 1:
                        x = idx + 1
                        new_mask = current & ~(1 << idx)
                        new_size = current_size - 1
                        delta = -1
                    else:
                        continue
            
            # Accept with probability
            if delta > 0 or random.random() < (T / max(0.5, abs(delta))):
                current = new_mask
                current_size = new_size
                if current_size > best_size:
                    best = current
                    best_size = current_size
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            T *= cooling
        
        return best, best_size
    
    # Strategy: multiple greedy fills with different orderings + local search
    best_mask = 0
    best_size = 0
    
    # Initial: upper half
    initial_mask = 0
    for i in range(65, 129):
        initial_mask |= (1 << (i - 1))
    
    greedy_mask = greedy_fill(initial_mask)
    best_mask = greedy_mask
    best_size = popcount(best_mask)
    
    # Local search from greedy result
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    if remaining > 10:
        sa_mask, sa_size = local_search(greedy_mask, remaining - 10)
        if sa_size > best_size:
            best_mask = sa_mask
            best_size = sa_size
    
    # More restarts
    elapsed = time.time() - start_time
    if deadline - start_time - elapsed > 15:
        for trial in range(5):
            if time.time() > deadline - 8:
                break
            # Random initial set
            rand_mask = 0
            for i in range(1, n + 1):
                if random.random() < 0.3:
                    rand_mask |= (1 << (i - 1))
            g_mask = greedy_fill(rand_mask)
            g_size = popcount(g_mask)
            if g_size > best_size:
                best_mask = g_mask
                best_size = g_size
            if deadline - start_time - time.time() > 5:
                sa_mask, sa_size = local_search(g_mask, 4)
                if sa_size > best_size:
                    best_mask = sa_mask
                    best_size = sa_size
    
    # Convert to list of Python ints
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(int(i + 1))
    
    return result
