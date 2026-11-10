"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128
    # Bit 0 is unused, bits 1..128 represent elements 1..128
    # full_mask has bits 1..128 set
    full_mask = (1 << (N + 1)) - 1  # bits 0..128, we use 1..128

    # Precompute: for each x in 1..128, and each y in 1..128,
    # the pair-sum x+y. We need to check if x+y is in the set.
    # For a set S (as bitmask), adding candidate c is valid iff
    # for all pairs (a,b) in S with a<b, NOT (a+b in S and a+c in S and b+c in S)
    # Equivalently: no pair (a,b) in S such that (a+b) in S and (a+c) in S and (b+c) in S

    # Key optimization: precompute for each candidate c, and for each element a,
    # the set of b such that if a and b are both in S, and c is added,
    # then a+b, a+c, b+c are all in S.
    # This depends on S, so we can't fully precompute. But we can precompute
    # the sums.

    # Let's use a different approach: for a given set S (bitmask) and candidate c,
    # the conflict exists if there exist a < b in S such that:
    #   (a+b) in S AND (a+c) in S AND (b+c) in S
    # Since a+c and b+c must be in S (not just S∪{c}), and a+c > a, b+c > b,
    # these are elements of S that are larger than a and b respectively.

    # For speed, represent S as a Python int bitmask.
    # bit[i] = 1 iff i is in S.

    def bit_set(mask, x):
        return (mask >> x) & 1

    def add_bit(mask, x):
        return mask | (1 << x)

    def remove_bit(mask, x):
        return mask & ~(1 << x)

    def count_bits(mask):
        return bin(mask).count('1')

    def is_valid_addition(mask, c):
        """Check if adding c to set represented by mask keeps validity.
        Need: no a<b in mask such that (a+b) in mask and (a+c) in mask and (b+c) in mask.
        """
        # Iterate over pairs in mask
        # For efficiency, iterate a from 1 to 128, then b from a+1 to 128
        # But we need to check membership, which is O(1) with bitmask
        a = 1
        while a <= N:
            if bit_set(mask, a):
                b = a + 1
                while b <= N:
                    if bit_set(mask, b):
                        if bit_set(mask, a + b) and bit_set(mask, a + c) and bit_set(mask, b + c):
                            return False
                    b += 1
            a += 1
        return True

    # Faster: precompute for each c, the list of (a, b, a+b, a+c, b+c) tuples
    # that would form a conflict. But this depends on which are in S.
    
    # Alternative: for a given mask and c, the conflict condition is:
    # exists a<b in mask: (a+b) in mask AND (a+c) in mask AND (b+c) in mask
    # We can rewrite: for each a in mask, for each b in mask with b>a:
    #   check if (a+b) in mask, (a+c) in mask, (b+c) in mask
    
    # Even faster: precompute for each c, an array conflict[c][a] = bitmask of b values
    # such that if a and b are in S, and (a+b) in S, then (a+c) and (b+c) being in S
    # would create a conflict. But (a+c) in S and (b+c) in S depend on S.
    
    # Let me try a different angle. For a fixed set S, adding c creates a conflict
    # iff there exist a<b in S with a+b in S, a+c in S, b+c in S.
    # Note a+c in S means a+c is an element of S. Similarly b+c in S.
    # So for each a in S, the values a+c must also be in S.
    # For each b in S, b+c must also be in S.
    # And a+b must be in S.
    
    # So: let S_c = {x in S : x+c in S}. Then we need a<b in S_c with a+b in S.
    # This is: among pairs in S_c, is there any pair whose sum is in S?
    
    # Precompute S_c as a bitmask: S_c_mask = 0
    # for x in S: if x+c <= N and bit_set(S, x+c): S_c_mask |= (1<<x)
    # Then check if any pair in S_c_mask has sum in S.
    
    def is_valid_addition_v2(mask, c):
        """Faster validity check using the S_c insight."""
        # Build S_c: elements x in mask such that x+c is also in mask
        S_c_mask = 0
        x = 1
        while x <= N:
            if bit_set(mask, x):
                if x + c <= N and bit_set(mask, x + c):
                    S_c_mask |= (1 << x)
            x += 1
        
        # Now check if any pair a<b in S_c_mask has a+b in mask
        a = 1
        while a <= N:
            if bit_set(S_c_mask, a):
                b = a + 1
                while b <= N:
                    if bit_set(S_c_mask, b):
                        if bit_set(mask, a + b):
                            return False
                    b += 1
            a += 1
        return True

    # Even faster: precompute for each c, and for each possible "pattern" of which
    # elements are in S, but that's 2^128 states, too many.
    
    # Let me try yet another approach: precompute all forbidden triples.
    # A forbidden triple is {a,b,c} with a<b<c such that a+b, a+c, b+c are all in [1,128].
    # Wait, the constraint is that a+b, a+c, b+c are all IN THE SET, not just in [1,128].
    # So the forbidden triples depend on the set.
    
    # But we can precompute: for each triple {a,b,c} with a<b<c, the three sums
    # a+b, a+c, b+c. If all three sums are in [1,128], then this triple is "potentially
    # forbidden" - it becomes forbidden if all three sums happen to be in S.
    
    # Precompute all such triples and their sum-triples.
    triples = []  # list of (a, b, c, s1, s2, s3) where s1=a+b, s2=a+c, s3=b+c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 <= N and s2 <= N and s3 <= N:
                    triples.append((a, b, c, s1, s2, s3))
    
    # For a set S, a triple (a,b,c) is forbidden if a,b,c,s1,s2,s3 are all in S.
    # When adding candidate x to S, new forbidden triples are those where x is one of
    # a,b,c or one of s1,s2,s3.
    
    # Actually, the condition is: a,b,c are in S, AND a+b, a+c, b+c are in S.
    # So all six values (a,b,c,a+b,a+c,b+c) must be in S.
    # When we add x, the only new triples that can become forbidden are those where
    # x is one of the six values.
    
    # For efficiency, let's precompute for each x, the list of triples where x appears
    # as one of the six values.
    
    triples_by_element = [[] for _ in range(N + 1)]
    for (a, b, c, s1, s2, s3) in triples:
        for val in (a, b, c, s1, s2, s3):
            triples_by_element[val].append((a, b, c, s1, s2, s3))
    
    def is_valid_addition_v3(mask, c):
        """Check validity by examining only triples involving c."""
        for (a, b, c2, s1, s2, s3) in triples_by_element[c]:
            # Check if all six values are in mask ∪ {c}
            # Since c is being added, it's in the set.
            # We need a, b, c2, s1, s2, s3 all in mask ∪ {c}
            # But a,b,c2 are the members, s1,s2,s3 are the sums.
            # All must be in the final set = mask ∪ {c}
            vals = (a, b, c2, s1, s2, s3)
            all_in = True
            for v in vals:
                if v == c:
                    continue  # c is being added, so it's in the set
                if not bit_set(mask, v):
                    all_in = False
                    break
            if all_in:
                return False
        return True

    # This should be much faster since we only check triples involving c.
    # Let's count how many triples involve each element.
    # For the base set, this should be very fast.

    # Now let's set up the base set
    base_A = set(range(65, 129))  # 64 elements
    base_B = set(range(17, 33))   # 16 elements
    base_set = base_A | base_B
    base_mask = 0
    for x in base_set:
        base_mask |= (1 << x)
    
    remaining = [x for x in range(1, 129) if x not in base_set]
    
    best_mask = base_mask
    best_size = len(base_set)

    # Multiple greedy strategies
    rng = random.Random(42)
    
    strategies = []
    strategies.append(sorted(remaining))
    strategies.append(sorted(remaining, reverse=True))
    mid = len(remaining) // 2
    small = sorted(remaining[:mid])
    large = sorted(remaining[mid:], reverse=True)
    strategies.append(small + large)
    strategies.append(large + small)
    for _ in range(80):
        c = remaining[:]
        rng.shuffle(c)
        strategies.append(c)

    for strat in strategies:
        if time.time() > deadline - 5:
            break
        mask = base_mask
        for x in strat:
            if time.time() > deadline - 5:
                break
            if bit_set(mask, x):
                continue
            if is_valid_addition_v3(mask, x):
                mask |= (1 << x)
        size = count_bits(mask)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Local improvement: try 1-for-2 swaps (remove x, add y and z)
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            mask = best_mask
            # Get list of elements in mask
            in_mask = [x for x in range(1, N + 1) if bit_set(mask, x)]
            not_in_mask = [x for x in range(1, N + 1) if not bit_set(mask, x)]
            
            for x in in_mask:
                if time.time() > deadline - 3:
                    break
                mask_remove = mask & ~(1 << x)
                
                for y in not_in_mask:
                    if time.time() > deadline - 3:
                        break
                    if not is_valid_addition_v3(mask_remove, y):
                        continue
                    mask_temp = mask_remove | (1 << y)
                    
                    for z in not_in_mask:
                        if z == y:
                            continue
                        if bit_set(mask_temp, z):
                            continue
                        if is_valid_addition_v3(mask_temp, z):
                            mask_new = mask_temp | (1 << z)
                            size_new = count_bits(mask_new)
                            if size_new > best_size:
                                best_mask = mask_new
                                best_size = size_new
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Try 1-for-1 swaps that enable growth
            if not improved and time.time() < deadline - 3:
                for x in in_mask:
                    if time.time() > deadline - 3:
                        break
                    mask_remove = mask & ~(1 << x)
                    for y in not_in_mask:
                        if time.time() > deadline - 3:
                            break
                        if not is_valid_addition_v3(mask_remove, y):
                            continue
                        mask_new = mask_remove | (1 << y)
                        # Try to add more
                        mask_grow = mask_new
                        for w in not_in_mask:
                            if w == y or bit_set(mask_grow, w):
                                continue
                            if is_valid_addition_v3(mask_grow, w):
                                mask_grow |= (1 << w)
                        if count_bits(mask_grow) > best_size:
                            best_mask = mask_grow
                            best_size = count_bits(mask_grow)
                            improved = True
                            break
                    if improved:
                        break

    # Convert bitmask to list of ints
    result = [int(x) for x in range(1, N + 1) if bit_set(best_mask, x)]
    return result
# EVOLVE-BLOCK-END
