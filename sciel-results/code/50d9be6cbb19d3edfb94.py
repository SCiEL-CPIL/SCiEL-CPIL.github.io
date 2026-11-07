import time
from math import isqrt

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def get_sf(n):
        sf = 1
        x = n
        p = 2
        while p * p <= x:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt ^= 1
            if cnt:
                sf *= p
            p += 1
        if x > 1:
            sf *= x
        return sf
    
    # Precompute square-free parts for a range
    max_n = 500
    sfs = {}
    for i in range(211, max_n + 1):
        sfs[i] = get_sf(i)
    
    # Target: sf(210) = 2*3*5*7 = 210
    target_sf = 210
    
    # For each M from 211 upward, check if there exists a non-empty subset
    # of {211, ..., M} whose product of sfs has square-free part = target_sf.
    # 
    # This is equivalent to: the XOR (in the multiplicative group of square-free ints)
    # of the selected sfs equals target_sf.
    #
    # Since square-free integers form a group under the operation: multiply then reduce to sf,
    # this is a linear algebra problem over GF(2) where the "basis" is the set of primes.
    #
    # Represent each sf as a bit vector over primes. Then we need a subset whose XOR = target.
    
    # Collect all primes up to max_n
    primes = []
    for p in range(2, max_n + 1):
        is_p = True
        for q in primes:
            if q * q > p:
                break
            if p % q == 0:
                is_p = False
                break
        if is_p:
            primes.append(p)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)
    
    def sf_to_bits(sf):
        """Convert square-free integer to bit vector (as Python int bitmask)."""
        bits = 0
        x = sf
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                bits |= (1 << prime_to_idx[p])
                x //= p
        if x > 1:
            if x in prime_to_idx:
                bits |= (1 << prime_to_idx[x])
            else:
                # Should not happen for our range
                pass
        return bits
    
    target_bits = sf_to_bits(target_sf)
    
    # For each M, build the set of bit vectors for numbers 211..M
    # and check if target_bits is in the span (XOR space) of these vectors.
    # Also need to ensure the subset is non-empty and we track which elements are used.
    
    # Use Gaussian elimination over GF(2) to find if target is in the span,
    # and if so, recover the subset.
    
    def find_subset(num_list, bits_list, target):
        """
        Find a subset of bits_list (corresponding to num_list) whose XOR equals target.
        Returns list of indices or None.
        Uses Gaussian elimination with tracking.
        """
        n = len(bits_list)
        if n == 0:
            return None
        
        # Build augmented matrix: [bits_list | identity]
        # We want to find x such that A*x = target
        # where A is the matrix with rows bits_list (as columns actually... let me think)
        
        # Actually, let's think of it as: we have vectors v_1, ..., v_n in GF(2)^d.
        # We want to find coefficients c_1, ..., c_n in {0,1} such that XOR of c_i * v_i = target.
        
        # Gaussian elimination: build a basis and express target in terms of it.
        
        # Represent each vector as a Python int (bitmask).
        # We'll do row reduction where each "row" is a vector.
        
        # Build list of (vector, index)
        vecs = [(bits_list[i], i) for i in range(n)]
        
        # Gaussian elimination
        basis = {}  # leading_bit -> (vector, index_set)
        # We'll maintain: for each pivot position, the vector that has that as its highest set bit
        # and the set of original indices used to form it.
        
        # Sort by highest bit descending
        vecs_sorted = sorted(vecs, key=lambda x: x[0].bit_length(), reverse=True)
        
        basis = {}  # pivot_bit -> (reduced_vector, frozenset_of_indices)
        
        for v, idx in vecs_sorted:
            if v == 0:
                continue
            # Reduce v by existing basis vectors
            cur = v
            used = {idx}
            # Find highest set bit
            hb = cur.bit_length() - 1
            # Try to reduce
            changed = True
            while changed:
                changed = False
                hb = cur.bit_length() - 1 if cur != 0 else -1
                if hb < 0:
                    break
                if hb in basis:
                    bv, bused = basis[hb]
                    cur ^= bv
                    used = used.symmetric_difference(bused)
                    changed = True
            if cur == 0:
                # This vector is in the span, skip (or we could use it but it doesn't add to span)
                continue
            hb = cur.bit_length() - 1
            basis[hb] = (cur, frozenset(used))
        
        # Now try to express target in terms of basis
        if target == 0:
            return []  # empty subset works, but we need non-empty
        
        cur = target
        used = set()
        changed = True
        while changed:
            changed = False
            hb = cur.bit_length() - 1 if cur != 0 else -1
            if hb < 0:
                break
            if hb in basis:
                bv, bused = basis[hb]
                cur ^= bv
                used = used.symmetric_difference(set(bused))
                changed = True
        
        if cur != 0:
            return None  # target not in span
        
        if not used:
            return None  # empty set, need non-empty
        
        return sorted(list(used))
    
    # Search for minimal M
    best_solution = None
    best_M = max_n + 1
    
    for M in range(211, max_n + 1):
        if time.time() > deadline - 5.0:
            break
        
        nums = list(range(211, M + 1))
        bits = [sf_to_bits(sfs[n]) for n in nums]
        
        # Check if target is in the span
        # Use the find_subset function
        idxs = find_subset(nums, bits, target_bits)
        
        if idxs is not None and len(idxs) > 0:
            # Verify
            selected = [nums[i] for i in idxs]
            prod = 210
            for x in selected:
                prod *= x
            if _is_perfect_square(prod):
                best_solution = selected
                best_M = M
                break  # Found minimal M
    
    if best_solution is None:
        # Fallback
        best_solution = [224, 240]
    
    # Ensure types are correct
    result = [int(x) for x in best_solution]
    return result
