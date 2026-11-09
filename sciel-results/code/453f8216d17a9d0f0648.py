import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 210
    max_val = 10210

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Precompute parity masks using a sieve-like approach
    # mask[x] = XOR of masks of its prime factors (with multiplicity)
    masks = [0] * (max_val + 1)
    for p in primes:
        p_bit = 1 << prime_index[p]
        for multiple in range(p, max_val + 1, p):
            # Count how many times p divides multiple
            cnt = 0
            t = multiple
            while t % p == 0:
                cnt += 1
                t //= p
            if cnt % 2 == 1:
                masks[multiple] ^= p_bit

    # Parity mask of 210
    target = masks[210]

    def verify_solution(sol):
        """Verify that product of sol * 210 is a perfect square."""
        prod = 210
        for x in sol:
            prod *= x
        s = math.isqrt(prod)
        return s * s == prod

    # Incremental basis building with history tracking
    basis_vec = {}   # pivot -> vector
    basis_hist = {}  # pivot -> history mask
    # Track which elements have been added to the basis
    added_to_basis = set()

    def add_to_basis(M2):
        """Add element M2 to the basis if it's independent."""
        v = masks[M2]
        h = 1 << (M2 - 211)
        tv = v
        th = h
        while tv:
            p = tv.bit_length() - 1
            if p in basis_vec:
                tv ^= basis_vec[p]
                th ^= basis_hist[p]
            else:
                basis_vec[p] = tv
                basis_hist[p] = th
                added_to_basis.add(M2)
                return True
        # Dependent element - store its representation
        # th now represents M2 as XOR of basis elements
        return th

    def check_target_in_span():
        """Check if target is in the span of the current basis.
        Returns the history mask if yes, None otherwise."""
        tv = target
        th = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis_vec:
                tv ^= basis_vec[p]
                th ^= basis_hist[p]
            else:
                return None
        return th

    def reduce_solution(sol_set, available_range):
        """Try to reduce the solution size by finding kernel vectors.
        available_range is (start, end) of elements that can be used."""
        # First pass: remove zero-mask elements and duplicate masks
        mask_to_elements = {}
        for e in sol_set:
            m = masks[e]
            if m not in mask_to_elements:
                mask_to_elements[m] = []
            mask_to_elements[m].append(e)
        
        removed = set()
        for m, elems in mask_to_elements.items():
            if m == 0:
                for e in elems:
                    removed.add(e)
            elif len(elems) >= 2:
                keep = elems[0]
                for e in elems[1:]:
                    removed.add(e)
        
        if removed:
            sol_set = sol_set - removed
        
        # Second pass: find small kernel vectors (size 2, 3, 4)
        # For size 2: already handled by duplicate masks
        # For size 3: find a,b,c with mask[a]^mask[b]^mask[c] = 0
        if len(sol_set) > 100:
            sol_list = list(sol_set)
            # Build a map from mask to list of elements with that mask
            mask_to_elems = {}
            for e in sol_list:
                m = masks[e]
                if m not in mask_to_elems:
                    mask_to_elems[m] = []
                mask_to_elems[m].append(e)
            
            # Also include elements from available_range not in sol_set for replacement
            new_removed = set()
            
            # Try to find triples in sol_set that sum to zero
            for i in range(len(sol_list)):
                if sol_list[i] in new_removed:
                    continue
                mi = masks[sol_list[i]]
                for j in range(i + 1, len(sol_list)):
                    if sol_list[j] in new_removed:
                        continue
                    mj = masks[sol_list[j]]
                    xor_ij = mi ^ mj
                    # Check if xor_ij is in mask_to_elems (some element with that mask)
                    if xor_ij in mask_to_elems:
                        for k in mask_to_elems[xor_ij]:
                            if k != sol_list[i] and k != sol_list[j] and k not in new_removed:
                                new_removed.add(sol_list[i])
                                new_removed.add(sol_list[j])
                                new_removed.add(k)
                                break
            
            if new_removed:
                sol_set = sol_set - new_removed
        
        # Third pass: try to replace elements with smaller ones from available_range
        # For each element in sol_set, see if there's a subset of (sol_set \ {e}) ∪ (available \ sol_set)
        # whose XOR equals mask[e]. If the subset is smaller, replace.
        if len(sol_set) > 100 and available_range:
            lo, hi = available_range
            sol_list = list(sol_set)
            
            # Build mask lookup for all available elements
            all_avail = set(range(lo, hi + 1))
            mask_lookup = {}
            for e in all_avail:
                m = masks[e]
                if m not in mask_lookup:
                    mask_lookup[m] = []
                mask_lookup[m].append(e)
            
            improved = True
            while improved and len(sol_set) > 100:
                improved = False
                sol_list = list(sol_set)
                for e in sol_list:
                    me = masks[e]
                    if me == 0:
                        sol_set.discard(e)
                        improved = True
                        continue
                    # Try to find a subset of other elements whose XOR = me
                    # Start with size 1: is there another element with same mask?
                    candidates = mask_lookup.get(me, [])
                    alt = None
                    for c in candidates:
                        if c != e and c not in sol_set:
                            alt = c
                            break
                    if alt is not None:
                        sol_set.discard(e)
                        sol_set.add(alt)
                        improved = True
                        continue
                    # Try size 2: find a,b not in sol_set with mask[a]^mask[b] = me
                    # This is expensive, so limit to small searches
                    if len(sol_set) > 100:
                        found_pair = False
                        # Sample some candidates
                        sample_size = min(50, len(all_avail))
                        import random
                        sample = random.sample(list(all_avail), sample_size)
                        for a in sample:
                            if a in sol_set or a == e:
                                continue
                            ma = masks[a]
                            need = me ^ ma
                            if need in mask_lookup:
                                for b in mask_lookup[need]:
                                    if b != a and b != e and b not in sol_set:
                                        sol_set.discard(e)
                                        sol_set.add(a)
                                        sol_set.add(b)
                                        found_pair = True
                                        improved = True
                                        break
                            if found_pair:
                                break
                        if found_pair:
                            continue
        
        return sol_set

    # Main search: find minimum M where a valid solution exists
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        
        # Add M to basis incrementally
        result = add_to_basis(M)
        
        # Check if target is now in the span
        th = check_target_in_span()
        if th is None:
            continue
        
        # Extract initial solution from history mask
        sol_set = set()
        hh = th
        while hh:
            lsb = hh & -hh
            idx = lsb.bit_length() - 1
            sol_set.add(211 + idx)
            hh ^= lsb
        
        # Ensure all elements in solution are <= M
        # (they should be, since we only added up to M)
        
        # Try to reduce solution
        sol_set = reduce_solution(sol_set, (211, M))
        
        if len(sol_set) <= 100:
            sol = list(sol_set)
            if verify_solution(sol):
                return [int(x) for x in sol]

    # Fallback (should not be reached)
    return [840]
