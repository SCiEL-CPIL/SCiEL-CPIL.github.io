"""Improved program for large sets with distinct subset products."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 40, 'maximum_size': 18}

# Precompute factorizations of all numbers 2..40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

def _factorize(n):
    """Return tuple of prime exponents for n."""
    exponents = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            exponents[i] += 1
            n //= p
    return tuple(exponents)

# Precompute factorizations for all numbers 2..40
_FACTORS = {}
for _i in range(2, 41):
    _FACTORS[_i] = _factorize(_i)

# Also precompute 2x, 3x, 4x, 5x, 6x of small primes for candidate generation
# These are numbers in 2..40 that are multiples of small primes
_MULTIPLES = {}
for _base in range(2, 41):
    _MULTIPLES[_base] = _FACTORS[_base]

def _check_validity_fast(current, candidate):
    """Check if adding candidate to current set keeps all subset products distinct.
    
    Uses the property: two subset products are equal iff they have the same prime exponent vector.
    Instead of checking all 2^(k+1) subsets, we can use the existing product set and check
    if any new subset product (old_subset * candidate) equals an existing one.
    
    Actually, the cleanest way: we maintain the set of all subset product exponent vectors.
    When we add a new element with factorization f, the new subset products are:
    - All old subset products (unchanged)
    - All old subset products multiplied by f
    
    So we need to check that no old_subset_product * f equals any old_subset_product,
    AND no two old_subset_product_i * f equals old_subset_product_j * f (which is equivalent
    to old_subset_product_i != old_subset_product_j, already guaranteed).
    
    Wait, that's not quite right. Let me think again.
    
    Old set S has 2^k subset products: {p_1, p_2, ..., p_{2^k}}
    New set S' = S ∪ {c} has 2^(k+1) subset products: {p_1, ..., p_{2^k}, p_1*c, ..., p_{2^k}*c}
    
    We need all of these to be distinct. Since the old ones are already distinct, we need:
    1. p_i * c != p_j for all i, j (i.e., p_i * c is not in the old set)
    2. p_i * c != p_j * c for all i != j (which reduces to p_i != p_j, already true)
    
    So we just need to check that p_i * c is not in {p_1, ..., p_{2^k}} for all i.
    
    In terms of exponent vectors: if old_subset_exponents = {(e_1), ..., (e_{2^k})} and
    c has factorization f, then new subsets have exponents {e_1, ..., e_{2^k}, e_1+f, ..., e_{2^k}+f}.
    We need e_i + f != e_j for all i, j.
    """
    k = len(current)
    if k == 0:
        return True
    
    f = _FACTORS[candidate]
    
    # Compute all 2^k subset exponent vectors
    # Use DP: exp[mask] = exp[mask without lsb] + factors[idx]
    num_subsets = 1 << k
    
    # Store as list of tuples
    exps = [None] * num_subsets
    exps[0] = tuple([0] * _NUM_PRIMES)
    
    # Precompute factor tuples for the current set
    cur_factors = [_FACTORS[x] for x in current]
    
    for mask in range(1, num_subsets):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        prev_exp = exps[prev_mask]
        cf = cur_factors[idx]
        exps[mask] = tuple(prev_exp[i] + cf[i] for i in range(_NUM_PRIMES))
    
    # Build set of existing exponent vectors
    seen = set(exps)
    
    # Check if any exp + f is in seen
    for exp in exps:
        new_exp = tuple(exp[i] + f[i] for i in range(_NUM_PRIMES))
        if new_exp in seen:
            return False
    
    return True

def _full_check(nums):
    """Full check that all 2^k subset products are distinct."""
    k = len(nums)
    if k == 0:
        return True
    if k == 1:
        return True
    
    num_subsets = 1 << k
    cur_factors = [_FACTORS[x] for x in nums]
    
    exps = [None] * num_subsets
    exps[0] = tuple([0] * _NUM_PRIMES)
    seen = {exps[0]}
    
    for mask in range(1, num_subsets):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        prev_exp = exps[prev_mask]
        cf = cur_factors[idx]
        exps[mask] = tuple(prev_exp[i] + cf[i] for i in range(_NUM_PRIMES))
        if exps[mask] in seen:
            return False
        seen.add(exps[mask])
    
    return True

def _greedy_extend(current, deadline):
    """Greedily extend the current set by adding valid candidates."""
    current = list(current)
    k = len(current)
    if k >= P['maximum_size']:
        return current
    
    # Try adding candidates in increasing order
    while len(current) < P['maximum_size']:
        if time.time() > deadline:
            break
        added = False
        for candidate in range(2, P['n'] + 1):
            if candidate not in current:
                if _check_validity_fast(current, candidate):
                    current.append(candidate)
                    added = True
                    break
        if not added:
            break
    
    return current

def _local_search_improved(start_set, deadline):
    """Local search: try swaps and additions to maximize set size."""
    best = list(start_set)
    if not _full_check(best):
        return None
    
    best_size = len(best)
    no_improve_count = 0
    
    while time.time() < deadline - 1.0 and len(best) < P['maximum_size']:
        improved = False
        
        # Try adding a new element
        for candidate in range(2, P['n'] + 1):
            if candidate not in best:
                if _check_validity_fast(best, candidate):
                    new_set = best + [candidate]
                    best = new_set
                    best_size = len(best)
                    improved = True
                    no_improve_count = 0
                    break
        
        if improved:
            continue
        
        # Try single swaps: remove one, add one
        swap_found = False
        for i in range(len(best)):
            if time.time() > deadline - 1.0:
                break
            for candidate in range(2, P['n'] + 1):
                if candidate == best[i] or candidate in best:
                    continue
                new_set = [x for j, x in enumerate(best) if j != i]
                new_set.append(candidate)
                if _full_check(new_set):
                    # Check if this swap leads to a larger set eventually
                    extended = _greedy_extend(new_set, deadline)
                    if len(extended) > best_size:
                        best = extended
                        best_size = len(best)
                        improved = True
                        swap_found = True
                        break
            if swap_found:
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # Try double swaps: remove two, add two
        if len(best) >= 2:
            ds_found = False
            n = len(best)
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                for j in range(i + 1, n):
                    for c1 in range(2, P['n'] + 1):
                        if c1 in best:
                            continue
                        for c2 in range(c1 + 1, P['n'] + 1):
                            if c2 in best:
                                continue
                            new_set = [x for k_idx, x in enumerate(best) if k_idx != i and k_idx != j]
                            new_set.extend([c1, c2])
                            if _full_check(new_set):
                                extended = _greedy_extend(new_set, deadline)
                                if len(extended) > best_size:
                                    best = extended
                                    best_size = len(best)
                                    improved = True
                                    ds_found = True
                                    break
                        if ds_found:
                            break
                    if ds_found:
                        break
                if ds_found:
                    break
            
            if improved:
                no_improve_count = 0
                continue
        
        no_improve_count += 1
        if no_improve_count > 5:
            break
    
    return best

def _random_start(deadline):
    """Generate a random valid starting set."""
    # Try to build a random set of size 4-8 that is valid
    for attempt in range(100):
        if time.time() > deadline:
            break
        size = random.randint(4, 8)
        candidates = random.sample(range(2, P['n'] + 1), size)
        if _full_check(candidates):
            return candidates
    # Fallback
    return [2, 3, 5, 7]

def construct():
    """Construct a set of distinct integers in 2..40 with distinct subset products.
    Maximize the number of selected integers."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best_set = None
    best_size = 0
    
    # Phase 1: Deterministic greedy from multiple orderings
    orderings = []
    # Natural order
    orderings.append(list(range(2, 41)))
    # Prime-heavy order
    primes_first = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites = [x for x in range(2, 41) if x not in primes_first]
    orderings.append(primes_first + composites)
    # Large-first
    orderings.append(list(range(40, 1, -1)))
    # Small-first
    orderings.append(list(range(2, 41)))
    
    for order in orderings:
        if time.time() > deadline:
            break
        current = []
        for candidate in order:
            if len(current) >= P['maximum_size']:
                break
            if _check_validity_fast(current, candidate):
                current.append(candidate)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Try to extend further
        extended = _greedy_extend(current, deadline)
        if len(extended) > best_size:
            best_size = len(extended)
            best_set = extended[:]
    
    # Phase 2: Local search from greedy results
    if best_set:
        result = _local_search_improved(best_set, deadline)
        if result and len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Phase 3: Random restarts with local search
    if time.time() < deadline - 5.0:
        num_restarts = 0
        while time.time() < deadline - 3.0 and num_restarts < 30:
            num_restarts += 1
            start_set = _random_start(deadline)
            if start_set is None:
                break
            
            # Extend greedily first
            extended = _greedy_extend(start_set, deadline)
            
            if len(extended) > best_size:
                best_size = len(extended)
                best_set = extended[:]
            
            # Local search
            if time.time() < deadline - 2.0:
                result = _local_search_improved(extended, deadline)
                if result and len(result) > best_size:
                    best_size = len(result)
                    best_set = result
            
            if best_size >= P['maximum_size']:
                break
    
    # Phase 4: If we have time, try more targeted approaches
    if best_set is not None and best_size < P['maximum_size'] and time.time() < deadline - 2.0:
        # Try to find what's blocking us
        # For each number in 2..40 not in best_set, see if removing one element allows it
        for candidate in range(2, P['n'] + 1):
            if candidate in best_set:
                continue
            for i in range(len(best_set)):
                if time.time() > deadline - 1.0:
                    break
                new_set = [x for j, x in enumerate(best_set) if j != i]
                if _check_validity_fast(new_set, candidate):
                    new_set.append(candidate)
                    extended = _greedy_extend(new_set, deadline)
                    if len(extended) > best_size:
                        best_size = len(extended)
                        best_set = extended[:]
                        break
                if best_size >= P['maximum_size']:
                    break
            if best_size >= P['maximum_size']:
                break
    
    # Final validation
    if best_set is None:
        best_set = [2]
    elif not _full_check(best_set):
        best_set = [2]
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
