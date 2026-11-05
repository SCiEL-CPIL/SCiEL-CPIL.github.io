"""Rewritten program: systematic + randomized search for large pairwise-coprime sumsets.

Key structural insights:
1. Pairwise-coprime sumset => at most one even number in A+B.
2. For a 3x3 grid a_i+b_j, the number of even sums is k*m + (3-k)*(3-m)
   where k = #even in A, m = #even in B. This equals 1 for NO (k,m) in [0,3]^2.
   Therefore we need ZERO even sums, which forces:
     - A all odd & B all even, OR
     - A all even & B all odd.
   This means every sum is ODD, so factor 2 never appears in the gcd check.
3. We only need to manage odd prime factors.

Strategy:
- Precompute smallest-prime-factor (SPF) table for numbers up to 256.
- For each odd number n in [3, 255], compute the set of odd prime factors.
- Search over combinations of 3 odd numbers for A and 3 even numbers for B
  (and vice versa), checking if all distinct sums are pairwise coprime.
- Use bitmask representation of prime factors for fast intersection checks.
- Systematic search with pruning, plus randomized restarts if time permits.
- Deadline guard: track elapsed time, return best found before 90s limit.
"""

import itertools
import math
import time

# Deadline: 90 seconds wall-clock, use 85s as safety margin
_DEADLINE = 85.0


def _build_spf(n_max):
    """Smallest prime factor table."""
    spf = list(range(n_max + 1))
    for i in range(2, int(n_max ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n_max + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def _odd_prime_factors(n, spf):
    """Return frozenset of odd prime factors of n."""
    factors = set()
    while n > 1:
        p = spf[n]
        if p != 2:
            factors.add(p)
        while n % p == 0:
            n //= p
    return frozenset(factors)


def _is_pairwise_coprime(values, spf):
    """Check if all distinct values in `values` are pairwise coprime."""
    distinct = list(set(values))
    if len(distinct) < 2:
        return True
    # Precompute prime factor sets
    pfs = [_odd_prime_factors(v, spf) for v in distinct]
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if pfs[i] & pfs[j]:
                return False
    return True


def _count_distinct_sums(A, B):
    """Count distinct values in A+B."""
    s = set()
    for a in A:
        for b in B:
            s.add(a + b)
    return len(s)


def _validate(A, B, spf):
    """Check validity: distinct sums are pairwise coprime. Return count or 0."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    if len(distinct) < 2:
        return len(distinct)
    # Check pairwise coprime
    dlist = list(distinct)
    pfs = [_odd_prime_factors(v, spf) for v in dlist]
    for i in range(len(dlist)):
        for j in range(i + 1, len(dlist)):
            if pfs[i] & pfs[j]:
                return 0
    return len(distinct)


def construct():
    t0 = time.time()
    
    # Build SPF up to 256 (max sum is 128+128=256)
    spf = _build_spf(256)
    
    # Odd numbers in [1, 128]
    odds = [i for i in range(1, 129) if i % 2 == 1]
    # Even numbers in [1, 128]
    evens = [i for i in range(2, 129) if i % 2 == 2]
    
    # Precompute prime factor frozensets for all odd numbers in [3, 255]
    # (sums of one odd + one even = odd, range [3, 255])
    odd_pf = {}
    for n in range(3, 256, 2):
        odd_pf[n] = _odd_prime_factors(n, spf)
    
    best_A = None
    best_B = None
    best_count = 0
    
    def check_update(A, B):
        nonlocal best_A, best_B, best_count
        # Quick check: all sums must be odd (guaranteed by parity)
        sums = [a + b for a in A for b in B]
        distinct = set(sums)
        cnt = len(distinct)
        if cnt <= best_count:
            return
        # Check pairwise coprime using precomputed pf
        dlist = list(distinct)
        pfs = [odd_pf[v] for v in dlist]
        ok = True
        for i in range(len(dlist)):
            for j in range(i + 1, len(dlist)):
                if pfs[i] & pfs[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best_A = list(A)
            best_B = list(B)
            best_count = cnt
    
    # Systematic search: A from odds, B from evens (and vice versa)
    # To make this tractable, we iterate over combinations but prune.
    # 3C from 64 odds * 3C from 63 evens = 41664 * 39711 ~ too many.
    # Instead, use a smarter approach: iterate over A combinations, and for each,
    # find compatible B combinations.
    
    # Let's do a more targeted search:
    # For each pair of A triples and B triples, check.
    # But that's still huge. Let's use random sampling + local improvement.
    
    import random
    random.seed(42)
    
    # First, try some structured candidates
    # Small primes tend to have few factors
    small_odds = [i for i in odds if i <= 32]
    small_evens = [i for i in evens if i <= 32]
    
    # Try all combos of small sets first
    for A in itertools.combinations(small_odds, 3):
        if time.time() - t0 > _DEADLINE:
            break
        for B in itertools.combinations(small_evens, 3):
            check_update(A, B)
    
    # Now do broader random search
    for iteration in range(500000):
        if time.time() - t0 > _DEADLINE:
            break
        # Random A from odds, B from evens
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        check_update(A, B)
        
        # Also try the reverse: A even, B odd
        A2 = random.sample(evens, 3)
        B2 = random.sample(odds, 3)
        check_update(A2, B2)
    
    # Local improvement: mutate best solution
    if best_A is not None:
        for _ in range(200000):
            if time.time() - t0 > _DEADLINE:
                break
            # Mutate one element of A or B
            if random.random() < 0.5:
                new_A = list(best_A)
                idx = random.randint(0, 2)
                new_A[idx] = random.choice(odds)
                if len(set(new_A)) < 3:
                    continue
                check_update(tuple(new_A), tuple(best_B))
            else:
                new_B = list(best_B)
                idx = random.randint(0, 2)
                new_B[idx] = random.choice(evens)
                if len(set(new_B)) < 3:
                    continue
                check_update(tuple(best_A), tuple(new_B))
    
    # If we found a solution, return it; otherwise return a valid fallback
    if best_A is not None:
        # Ensure distinct elements
        return [best_A, best_B]
    else:
        # Fallback: a known valid configuration
        # A=[1,3,5], B=[2,4,6]: sums = {3,4,5, 5,6,7, 7,8,9}
        # Distinct: {3,4,5,6,7,8,9} - check pairwise coprime
        # 3&4=1, 3&5=1, 3&6=3 NO. So this is invalid.
        # Let's use the original seed: A=[1,7,13], B=[4,10,16]
        # Sums: 5,11,17, 11,17,23, 17,23,29 -> distinct {5,11,17,23,29}
        # All primes, so pairwise coprime. Count=5.
        return [[1, 7, 13], [4, 10, 16]]


# Quick self-test
if __name__ == "__main__":
    A, B = construct()
    print(f"A={A}, B={B}")
    sums = [a+b for a in A for b in B]
    distinct = sorted(set(sums))
    print(f"Distinct sums: {distinct}, count={len(distinct)}")
    # Verify pairwise coprime
    import math
    ok = True
    for i in range(len(distinct)):
        for j in range(i+1, len(distinct)):
            if math.gcd(distinct[i], distinct[j]) != 1:
                ok = False
                print(f"  gcd({distinct[i]},{distinct[j]})={math.gcd(distinct[i],distinct[j])}")
    print(f"Pairwise coprime: {ok}")
