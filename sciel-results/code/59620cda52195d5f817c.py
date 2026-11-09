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

# --- Bitset-based validity check using subset sum collision detection ---
# A set is valid iff no non-trivial integer linear combination of exponent vectors = 0
# Equivalently: all 2^k subset product exponent sums are distinct.
# For efficiency, we use the following insight:
# If we maintain the set of all achievable "difference" vectors (subset A - subset B),
# adding a new number v is valid iff v is not in the existing difference set
# and v + existing differences don't create collisions.
# 
# Alternative: maintain all subset sums of the current set. Adding v is valid iff
# no existing subset sum equals v + another existing subset sum (i.e., v is not a
# difference of two existing subset sums).

def _valid_with_subset_sums(subset_sums, vec):
    """Check if vec can be added given the set of existing subset sums."""
    # vec can be added iff for all s1, s2 in subset_sums, s1 + vec != s2
    # i.e., vec is not in {s1 - s2 : s1, s2 in subset_sums}
    # Equivalently: for all s in subset_sums, s + vec is not in subset_sums
    # and also we need: no two elements of (subset_sums ∪ {s+vec for s in subset_sums}) collide
    # Actually simpler: new subset sums = subset_sums ∪ {s + vec for s in subset_sums}
    # Valid iff all these 2*(2^k) sums are distinct.
    new_sums = set()
    for s in subset_sums:
        if s in new_sums:
            return False, None
        new_sums.add(s)
    for s in subset_sums:
        ns = tuple(s[i] + vec[i] for i in range(NPRIMES))
        if ns in new_sums:
            return False, None
        new_sums.add(ns)
    return True, new_sums

def _construct():
    t0 = time.time()
    deadline = t0 + 55.0
    
    # Initial subset sums: just {0-vector}
    zero_vec = tuple([0] * NPRIMES)
    
    best_set = []
    best_size = 0
    
    # All candidate numbers 2..40
    all_nums = list(range(2, 41))
    
    # Bitset-based backtracking: try to build a maximal independent set
    # State: (remaining candidates bitmask, current subset_sums set, current selected list)
    # Use iterative deepening or simple backtracking with good ordering.
    
    # Key insight: order candidates so that "hard" numbers (those with many prime factors
    # or high exponents) are tried later, giving more pruning opportunities.
    
    # Score each number by how "constraining" it is
    def _constrain_score(x):
        v = EXP_VECTORS[x]
        # Higher score = more constraining = try later
        return sum(v) + sum(1 for e in v if e > 0)
    
    # Sort: least constraining first (try to include them early)
    ordered = sorted(all_nums, key=lambda x: _constrain_score(x))
    
    # Backtracking with subset-sum-based pruning
    def _backtrack(idx, subset_sums, selected, deadline):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
        
        if idx >= len(ordered):
            return
        
        # Upper bound: can we beat best_size?
        if len(selected) + (len(ordered) - idx) <= best_size:
            return
        
        num = ordered[idx]
        vec = EXP_VECTORS[num]
        
        # Try including num
        valid, new_sums = _valid_with_subset_sums(subset_sums, vec)
        if valid:
            selected.append(num)
            _backtrack(idx + 1, new_sums, selected, deadline)
            selected.pop()
        
        # Try excluding num
        _backtrack(idx + 1, subset_sums, selected, deadline)
    
    _backtrack(0, {zero_vec}, [], deadline - 5.0)
    
    # If we found a good set, try to improve with local search
    if time.time() < deadline - 10 and best_size < 18:
        current = list(best_set)
        current_set = set(current)
        
        # Rebuild subset sums for current set
        ss = {zero_vec}
        for x in current:
            v = EXP_VECTORS[x]
            new_ss = set(ss)
            for s in ss:
                ns = tuple(s[i] + v[i] for i in range(NPRIMES))
                new_ss.add(ns)
            ss = new_ss
        
        # Local search: try swapping one element for another
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for ri in range(len(current)):
                if time.time() > deadline - 3:
                    break
                removed = current[ri]
                remaining = [x for i, x in enumerate(current) if i != ri]
                
                # Rebuild subset sums for remaining
                ss_rem = {zero_vec}
                for x in remaining:
                    v = EXP_VECTORS[x]
                    new_ss = set(ss_rem)
                    for s in ss_rem:
                        ns = tuple(s[i] + v[i] for i in range(NPRIMES))
                        new_ss.add(ns)
                    ss_rem = new_ss
                
                for c in range(2, 41):
                    if c in current_set:
                        continue
                    valid, _ = _valid_with_subset_sums(ss_rem, EXP_VECTORS[c])
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
