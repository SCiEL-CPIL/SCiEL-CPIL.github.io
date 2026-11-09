"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

# --- Core data: primes and exponent vectors ---
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NPRIMES = len(PRIMES)

def _sieve_primes(n):
    if n < 2:
        return []
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

# Exponent vectors for numbers 2..40
EXP_VECTORS = {}
for _i in range(2, 41):
    _vec = [0] * NPRIMES
    _n = _i
    for _j, _p in enumerate(PRIMES):
        while _n % _p == 0:
            _vec[_j] += 1
            _n //= _p
    EXP_VECTORS[_i] = tuple(_vec)

# Precompute: for each number, which primes divide it
def _prime_mask(x):
    mask = 0
    for j, p in enumerate(PRIMES):
        if EXP_VECTORS[x][j] > 0:
            mask |= (1 << j)
    return mask

PRIME_MASKS = {x: _prime_mask(x) for x in range(2, 41)}

# --- Fast validity check using difference set ---
# A set S is valid iff all 2^|S| subset products are distinct.
# Equivalent: no non-trivial integer combination of exponent vectors sums to zero.
# We maintain the set of all "difference" vectors: {sum(A) - sum(B) : A, B subsets of S}
# Adding v is valid iff v is not in the difference set.
# When we add v, new differences = old_differences ∪ {v + d : d in old_differences} ∪ {-v - d : d in old_differences}
# Actually simpler: new_differences = old_differences ∪ {v + d : d in old_differences}
# But we also need to check that v itself doesn't collide, which is covered since 0 is in old_differences.

def _valid_with_diff_set(diff_set, vec):
    """Check if vec can be added given the set of existing difference vectors."""
    if vec in diff_set:
        return False, None
    new_diff = set(diff_set)
    for d in diff_set:
        nd = tuple(d[i] + vec[i] for i in range(NPRIMES))
        if nd in new_diff:
            return False, None
        new_diff.add(nd)
    return True, new_diff

def _construct():
    t0 = time.time()
    deadline = t0 + 55.0
    
    zero_vec = tuple([0] * NPRIMES)
    
    best_set = []
    best_size = 0
    
    all_nums = list(range(2, 41))
    
    def _constrain_score(x):
        v = EXP_VECTORS[x]
        num_distinct = sum(1 for e in v if e > 0)
        total_exp = sum(v)
        return num_distinct * 2 + total_exp
    
    # Generate multiple orderings for restarts
    import random
    rng = random.Random(42)
    
    orderings = []
    # Ordering 1: least constraining first
    orderings.append(sorted(all_nums, key=lambda x: _constrain_score(x)))
    # Ordering 2: most constraining first
    orderings.append(sorted(all_nums, key=lambda x: -_constrain_score(x)))
    # Ordering 3: primes first, then by value
    primes_in_range = [x for x in all_nums if all(x % d != 0 for d in range(2, isqrt(x) + 1))]
    composites_in_range = [x for x in all_nums if x not in primes_in_range]
    orderings.append(primes_in_range + composites_in_range)
    # Ordering 4: random
    for _ in range(5):
        rand_order = list(all_nums)
        rng.shuffle(rand_order)
        orderings.append(rand_order)
    
    def _backtrack(ordered, idx, diff_set, selected, deadline):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
        
        if idx >= len(ordered):
            return
        
        # Upper bound
        if len(selected) + (len(ordered) - idx) <= best_size:
            return
        
        num = ordered[idx]
        vec = EXP_VECTORS[num]
        
        # Try including num
        valid, new_diff = _valid_with_diff_set(diff_set, vec)
        if valid:
            selected.append(num)
            _backtrack(ordered, idx + 1, new_diff, selected, deadline)
            selected.pop()
        
        # Try excluding num
        _backtrack(ordered, idx + 1, diff_set, selected, deadline)
    
    # Try each ordering with a time budget
    num_orderings = len(orderings)
    time_per_ordering = (deadline - 5.0) / num_orderings
    
    for oi, ordered in enumerate(orderings):
        elapsed = time.time() - t0
        remaining_time = deadline - 5.0 - elapsed
        if remaining_time < 1.0:
            break
        bt_deadline = t0 + elapsed + time_per_ordering
        if bt_deadline > deadline - 5.0:
            bt_deadline = deadline - 5.0
        _backtrack(ordered, 0, {zero_vec}, [], bt_deadline)
    
    # Local search: try swapping one element for another
    if time.time() < deadline - 10 and best_size < 18:
        current = list(best_set)
        current_set = set(current)
        
        # Rebuild difference set for current set
        ds = {zero_vec}
        for x in current:
            v = EXP_VECTORS[x]
            new_ds = set(ds)
            for d in ds:
                nd = tuple(d[i] + v[i] for i in range(NPRIMES))
                new_ds.add(nd)
            ds = new_ds
        
        # Local search: try swapping one element for another
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for ri in range(len(current)):
                if time.time() > deadline - 3:
                    break
                removed = current[ri]
                remaining = [x for i, x in enumerate(current) if i != ri]
                
                # Rebuild difference set for remaining
                ds_rem = {zero_vec}
                for x in remaining:
                    v = EXP_VECTORS[x]
                    new_ds = set(ds_rem)
                    for d in ds_rem:
                        nd = tuple(d[i] + v[i] for i in range(NPRIMES))
                        new_ds.add(nd)
                    ds_rem = new_ds
                
                for c in range(2, 41):
                    if c in current_set:
                        continue
                    valid, _ = _valid_with_diff_set(ds_rem, EXP_VECTORS[c])
                    if valid:
                        new_set = remaining + [c]
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(new_set)
                            current = new_set
                            current_set = set(current)
                            improved = True
                            break
                if improved:
                    break
    
    # Final verification with direct product enumeration
    def _direct_verify(nums):
        k = len(nums)
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= nums[i]
            if prod in products:
                return False
            products.add(prod)
        return True
    
    if not _direct_verify(best_set):
        # Fallback: use primes
        best_set = [x for x in range(2, 41) if all(x % d != 0 for d in range(2, isqrt(x) + 1))]
    
    # Ensure return type is list of Python ints
    best_set = [int(x) for x in best_set]
    
    assert len(best_set) >= 12
    assert all(isinstance(x, int) for x in best_set)
    assert all(2 <= x <= 40 for x in best_set)
    assert len(set(best_set)) == len(best_set)
    
    return best_set

def construct():
    return _construct()
# EVOLVE-BLOCK-END
