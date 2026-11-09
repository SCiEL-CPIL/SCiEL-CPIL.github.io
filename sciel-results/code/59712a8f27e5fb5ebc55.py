"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_sieve(max_val):
    """Precompute smallest prime factor for fast factorization up to max_val."""
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _prime_factors_spf(n, spf):
    """Get distinct prime factors of n using precomputed smallest prime factor."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _count_distinct_primes(numbers, spf):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    n = len(numbers)
    prime_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            s = numbers[i] + numbers[j]
            prime_set |= _prime_factors_spf(s, spf)
    return len(prime_set), prime_set

def _try_residue_class(mod, residue, max_val, k=12, spf=None):
    """Try to find k distinct numbers all congruent to residue mod mod.
    If all numbers ≡ r (mod m), then all pairwise sums ≡ 2r (mod m).
    This constrains the prime factors of the sums."""
    candidates = [x for x in range(1, max_val + 1) if x % mod == residue]
    if len(candidates) < k:
        return None, None
    
    # Try greedy: pick k numbers that minimize distinct prime factors
    best_count = float('inf')
    best_set = None
    best_primes = None
    
    # Try different starting points in the candidate list
    for start in range(0, min(len(candidates) - k + 1, 50)):
        subset = candidates[start:start + k]
        count, primes = _count_distinct_primes(subset, spf)
        if count < best_count:
            best_count = count
            best_set = subset[:]
            best_primes = primes
    
    # Also try spaced-out selections
    step = max(1, (max_val // k))
    for base in range(1, min(mod * 3, 100)):
        subset = []
        for i in range(k):
            val = base + i * step
            if val <= max_val and val % mod == residue:
                subset.append(val)
            else:
                # Find next valid value
                next_val = val
                while next_val <= max_val and next_val % mod != residue:
                    next_val += 1
                if next_val <= max_val:
                    subset.append(next_val)
                else:
                    break
        if len(subset) == k and len(set(subset)) == k:
            count, primes = _count_distinct_primes(subset, spf)
            if count < best_count:
                best_count = count
                best_set = subset[:]
                best_primes = primes
    
    if best_set is None:
        return None, None
    return best_count, best_set

def _local_search(numbers, spf, max_val, time_limit):
    """Improve a starting set via local search (swap one element)."""
    n = len(numbers)
    current = sorted(numbers)
    current_count, current_primes = _count_distinct_primes(current, spf)
    
    start_time = time.time()
    improved = True
    iteration = 0
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        iteration += 1
        for i in range(n):
            # Try replacing current[i] with other values
            for new_val in range(1, max_val + 1):
                if new_val in current:
                    continue
                candidate = current[:i] + [new_val] + current[i+1:]
                candidate = sorted(candidate)
                count, _ = _count_distinct_primes(candidate, spf)
                if count < current_count:
                    current = candidate
                    current_count = count
                    current_primes = _prime_factors_spf(0, spf)  # reset
                    improved = True
                    break
        if time.time() - start_time > time_limit * 0.8:
            break
    
    return current, current_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    max_val = 10000
    k = 12
    
    # Build sieve once
    spf = _build_sieve(2 * max_val)
    
    best_count = float('inf')
    best_numbers = list(range(1, k + 1))
    
    # Strategy 1: Try residue classes with small moduli
    # If all numbers ≡ r (mod m), sums ≡ 2r (mod m)
    # We want 2r to have few prime factors
    for mod in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
        for residue in range(mod):
            if time.time() > deadline:
                break
            count, numbers = _try_residue_class(mod, residue, max_val, k, spf)
            if numbers is not None and count < best_count:
                best_count = count
                best_numbers = numbers
    
    # Strategy 2: Try arithmetic progressions
    for a in range(1, 100):
        for d in range(1, 100):
            if time.time() > deadline:
                break
            seq = [a + i * d for i in range(k)]
            if max(seq) > max_val:
                continue
            if len(set(seq)) != k:
                continue
            count, _ = _count_distinct_primes(seq, spf)
            if count < best_count:
                best_count = count
                best_numbers = seq[:]
    
    # Strategy 3: Local search from best found so far
    if time.time() < deadline - 10:
        local_start = time.time()
        local_time_limit = min(20, deadline - time.time() - 5)
        improved_numbers, improved_count = _local_search(best_numbers, spf, max_val, local_time_limit)
        if improved_count < best_count:
            best_count = improved_count
            best_numbers = improved_numbers
    
    # Final verification
    final_count, _ = _count_distinct_primes(best_numbers, spf)
    return best_numbers

# EVOLVE-BLOCK-END
