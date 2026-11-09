"""Direct construction for 'Large sets with distinct subset products'."""

import time
import itertools

DEADLINE = 82.0
T_START = time.time()

def _is_valid(numbers):
    """Check if all subset products are distinct."""
    n = len(numbers)
    if n > 25:
        # Too many to enumerate 2^n, use prime exponent check
        from math import gcd
        # Check multiplicative independence via prime exponents
        # For each prime, the exponents must be subset-sum-free
        # This is complex; fall back to sampling
        return None  # unknown
    prods = set()
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= numbers[i]
        if prod in prods:
            return False
        prods.add(prod)
    return True

def _exponent_vector(n, primes):
    """Return tuple of exponents of n w.r.t. primes."""
    vec = []
    for p in primes:
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        vec.append(e)
    return tuple(vec)

def _is_subset_sum_free(exponents):
    """Check if multiset of exponents is subset-sum-free."""
    # Remove zeros
    pos = [e for e in exponents if e > 0]
    n = len(pos)
    for i in range(n):
        target = pos[i]
        others = [pos[j] for j in range(n) if j != i]
        # Check if target = sum of any subset of others
        m = len(others)
        for mask in range(1, 1 << m):
            s = 0
            for j in range(m):
                if mask & (1 << j):
                    s += others[j]
            if s == target:
                return False
    return True

def _check_validity_exponents(numbers, primes):
    """Check validity using prime exponent vectors."""
    for p_idx, p in enumerate(primes):
        exps = []
        for num in numbers:
            e = 0
            x = num
            while x % p == 0:
                x //= p
                e += 1
            exps.append(e)
        if not _is_subset_sum_free(exps):
            return False
    # Also need to check that numbers with primes not in our list don't cause issues
    # But since we only use primes in [2,40], and numbers are in [2,40],
    # any prime factor of a number <= 40 is in our prime list
    return True

def construct():
    global DEADLINE, T_START
    deadline = DEADLINE
    if deadline < 0:
        deadline = time.time() + 82.0

    def time_left():
        return deadline - time.time()

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Generate initial set: prime powers with sum-free exponent sets
    initial = []
    for p in primes:
        m = 1
        while p ** (m + 1) <= 40:
            m += 1
        exps_avail = list(range(1, m + 1))
        best = []
        for k in range(len(exps_avail), -1, -1):
            for combo in itertools.combinations(exps_avail, k):
                cset = list(combo)
                ok = _is_subset_sum_free(cset)
                if ok:
                    best = cset
                    break
            if best:
                break
        for e in best:
            v = p ** e
            if 2 <= v <= 40 and v not in initial:
                initial.append(v)

    best_set = sorted(initial)
    best_size = len(best_set)
    
    all_candidates = list(range(2, 41))
    
    # Greedy add: try to add each candidate if valid
    for cand in all_candidates:
        if cand in best_set:
            continue
        if time_left() < 5.0:
            break
        new_set = best_set + [cand]
        if _check_validity_exponents(new_set, primes):
            best_set = new_set
            best_size += 1
    
    # Local search: try swaps
    improved = True
    while improved and time_left() > 10.0:
        improved = False
        for i, num in enumerate(list(best_set)):
            for cand in all_candidates:
                if cand in best_set:
                    continue
                new_set = best_set[:i] + best_set[i+1:] + [cand]
                if _check_validity_exponents(new_set, primes):
                    # Only swap if it allows us to add more later, or keep size same
                    # For now, just track best
                    pass
        # Try to add more after swaps
        for cand in all_candidates:
            if cand in best_set:
                continue
            if time_left() < 5.0:
                break
            new_set = best_set + [cand]
            if _check_validity_exponents(new_set, primes):
                best_set = new_set
                best_size += 1
                improved = True
                break
    
    # Final verification
    n = len(best_set)
    if n <= 25:
        valid = _is_valid(best_set)
        if not valid:
            # Fallback
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    else:
        valid = _check_validity_exponents(best_set, primes)
        if not valid:
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    res = [int(x) for x in best_set]
    assert all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 for x in res)
    assert len(res) == len(set(res))
    
    return res
