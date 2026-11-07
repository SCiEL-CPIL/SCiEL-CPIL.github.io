import random
import time

def construct():
    # Max possible sum for 30 elements of max 256
    MAX_SUM = 256 * 30 + 10
    
    # Precompute forbidden masks: for each sum s, mask[s] has bits set at all multiples of s (2s, 3s, ...)
    # When we add a new sum t, we need to check:
    # 1. Does any existing sum s divide t? -> Check if (bits & mask_divisors[t]) != 0
    # 2. Does t divide any existing sum? -> Check if (bits & mask_multiples[t]) != 0
    # 3. Do new sums conflict with each other?
    
    # Actually, let's precompute for each value v (1..MAX_SUM):
    # multiples_mask[v]: bitset with bits at 2v, 3v, ... <= MAX_SUM
    # This allows O(1) check: if (bits & multiples_mask[v]) != 0, then v divides some existing sum
    
    # For the "v is divisible by some existing sum" check, we need:
    # divisors_mask[v]: bitset with bits at all proper divisors of v
    # Then if (bits & divisors_mask[v]) != 0, then some existing sum divides v
    
    # But wait - when we add a new element x, we create MANY new sums (all old sums + x).
    # We need to check all new sums against each other and against old sums.
    
    # Let's think differently. When we have current set with sum-bitset `bits`,
    # and we add element x, the new sums are: {old_sum + x for old_sum in sums} ∪ {x}
    # where sums = {s : bit s in bits}
    
    # new_bits = bits | (bits << x) | (1 << x)
    # The NEW bits (not in original bits) are: (bits << x) | (1 << x) minus bits
    # Actually: new_sums = (bits << x) | (1 << x)
    # But some of these might already be in bits (collisions!)
    # Collision check: (bits & (bits << x)) != 0 means there's a collision
    
    # If no collision, the new sums are exactly (bits << x) | (1 << x)
    # We need to check:
    # a) No two new sums divide each other
    # b) No new sum divides an old sum  
    # c) No old sum divides a new sum
    
    # Let's precompute multiples_mask for each value
    # multiples_mask[v] = sum of (1 << k*v) for k = 2, 3, ... while k*v <= MAX_SUM
    
    # For checking "does new sum s divide any old sum":
    #   (bits & multiples_mask[s]) != 0
    
    # For checking "does any old sum divide new sum s":
    #   We need divisors of s that are in bits
    #   divisors_mask[s] = sum of (1 << d) for d | s, d < s
    #   (bits & divisors_mask[s]) != 0
    
    # For checking new sums against each other:
    #   Let new_sums_bits = (bits << x) | (1 << x)
    #   For each new sum s in new_sums_bits:
    #     Check if any other new sum t is a multiple of s: (new_sums_bits & multiples_mask[s]) != 0
    #     Check if any other new sum t is a divisor of s: (new_sums_bits & divisors_mask[s]) != 0
    #   But this is still O(number_of_new_sums) per check.
    
    # Alternative: just check the full new_bits for any divisibility conflict.
    # But we can optimize by only checking the new parts.
    
    # Let me precompute the masks.
    
    # Precompute multiples_mask: for each v, bitmask of multiples
    # This is a list of Python ints (arbitrary precision bitsets)
    
    # Actually, MAX_SUM is ~7700, so we can use Python ints as bitsets.
    
    # Let's precompute:
    # multiples_mask[v] for v in 1..MAX_SUM
    # divisors_mask[v] for v in 1..MAX_SUM
    
    # But storing 7700 big ints might be memory-heavy. Let's see.
    # Each is ~7700 bits = ~1KB. 7700 * 1KB = ~7.7MB. That's fine.
    
    # Actually, let's just precompute multiples_mask.
    # For the divisor check, we can iterate through divisors of s.
    
    # Let's try a simpler approach first: precompute multiples_mask only.
    
    # Precompute multiples_mask
    multiples_mask = [0] * (MAX_SUM + 1)
    for v in range(1, MAX_SUM + 1):
        mask = 0
        for k in range(2, MAX_SUM // v + 1):
            mask |= (1 << (k * v))
        multiples_mask[v] = mask
    
    # Now, is_valid can be:
    # For each set bit s in bits:
    #   if (bits & multiples_mask[s]) != 0: return False
    # But this is still O(number_of_set_bits * cost_of_AND)
    
    # Better: iterate through set bits and check.
    
    def is_valid(bits):
        # Extract set bits and check
        b = bits
        while b:
            # Get lowest set bit
            lb = b & (-b)
            s = lb.bit_length() - 1
            if (bits & multiples_mask[s]) != 0:
                return False
            b ^= lb
        return True
    
    # But this iterates through all set bits. For dense sets, this could be slow.
    
    # Alternative: iterate s from 1 to MAX_SUM//2, check if bit s is set, then check multiples.
    # This is what the original did. Let's keep that but use the precomputed mask.
    
    def is_valid_v2(bits):
        for s in range(1, MAX_SUM // 2 + 1):
            if (bits >> s) & 1:
                if (bits & multiples_mask[s]) != 0:
                    return False
        return True
    
    # Even better: we only need to check NEW sums when adding an element.
    # When we add x to current set with bits:
    # new_bits = bits | (bits << x) | (1 << x)
    # First check collision: if (bits & (bits << x)) != 0: invalid
    # Then the new sums are: new_sums = (bits << x) | (1 << x)
    # But some of new_sums might overlap with bits (that's the collision case)
    # If no collision, new_sums & bits == 0
    # We need to check:
    # 1. For each new sum s in new_sums:
    #    a. (bits & multiples_mask[s]) == 0  (no old sum is multiple of s)
    #    b. For each old sum d that divides s: d should not be in bits... wait
    #       Actually: if old sum d divides new sum s, that's a conflict.
    #       So we need: no old sum d such that d | s.
    #       This is: (bits & divisors_mask[s]) == 0 where divisors_mask[s] has bits at proper divisors of s.
    #    c. New sums don't divide each other.
    
    # Let's precompute divisors_mask too.
    
    divisors_mask = [0] * (MAX_SUM + 1)
    for v in range(1, MAX_SUM + 1):
        mask = 0
        for d in range(1, v):
            if v % d == 0:
                mask |= (1 << d)
        divisors_mask[v] = mask
    
    # Now, when adding x:
    # 1. Check collision: (bits & (bits << x)) != 0 -> invalid
    # 2. new_sums = (bits << x) | (1 << x)
    # 3. Check new_sums vs bits:
    #    For each s in new_sums (set bits):
    #      if (bits & multiples_mask[s]) != 0: invalid  (s divides some old sum)
    #      if (bits & divisors_mask[s]) != 0: invalid  (some old sum divides s)
    # 4. Check new_sums vs new_sums:
    #    For each s in new_sums:
    #      if (new_sums & multiples_mask[s]) != 0: invalid  (but this includes s itself? No, multiples_mask[s] starts at 2s)
    #      if (new_sums & divisors_mask[s]) != 0: invalid
    
    # But iterating through set bits of new_sums is still potentially expensive.
    
    # Let's just use the full is_valid check on new_bits. It's simpler and might be fast enough.
    
    # Actually, let's optimize is_valid by using the precomputed masks and only checking up to MAX_SUM//2.
    
    def check_valid(bits):
        # Check if any pair (s, t) with s < t and t % s == 0 exists in bits
        # Iterate s from 1 to MAX_SUM//2
        for s in range(1, MAX_SUM // 2 + 1):
            if (bits >> s) & 1:
                if (bits & multiples_mask[s]) != 0:
                    return False
        return True
    
    # Now let's do the search.
    
    start_time = time.time()
    timeout = 55
    
    best_set = []
    best_size = 0
    
    candidates = list(range(1, 257))
    
    # Try multiple random restarts with different orderings
    attempt = 0
    while time.time() - start_time < timeout:
        attempt += 1
        
        # Different strategies for ordering
        if attempt % 3 == 0:
            # Descending
            cand = sorted(candidates, reverse=True)
        elif attempt % 3 == 1:
            # Random
            cand = candidates[:]
            random.shuffle(cand)
        else:
            # Sort by number of divisors (fewer divisors = safer to include?)
            # Or sort by value * something
            cand = candidates[:]
            random.shuffle(cand)
            # Bias towards larger numbers
            cand.sort(key=lambda x: -x)
            # But randomize within groups?
            # Just do a partial shuffle
            for i in range(len(cand) - 1, 0, -1):
                j = random.randint(0, i)
                cand[i], cand[j] = cand[j], cand[i]
        
        current_set = []
        bits = 0
        
        for x in cand:
            # Check if adding x is valid
            # First check collision
            shifted = bits << x
            if (bits & shifted) != 0:
                continue  # Collision, skip
            
            new_bits = bits | shifted | (1 << x)
            
            # Check validity
            if check_valid(new_bits):
                bits = new_bits
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return [int(x) for x in best_set]
