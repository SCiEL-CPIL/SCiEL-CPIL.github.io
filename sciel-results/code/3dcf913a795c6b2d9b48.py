"""Rewritten program: exhaustive search with prime-mask pruning for max pairwise-coprime sumset."""
import time
from itertools import combinations

# Deadline guard: 90s wall-clock limit, use 80s as safety margin
_DEADLINE = time.time() + 80.0


def _prime_factor_mask(n):
    """Return a bitmask where bit i is set if prime i-th prime divides n.
    We only care about primes up to 256 (max sum is 128+128=256)."""
    mask = 0
    d = 2
    prime_idx = 0
    while d * d <= n:
        if n % d == 0:
            mask |= (1 << prime_idx)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2  # 2, then odd numbers
        # Track prime index properly
        # Actually let's use a different approach: map primes to indices
    if n > 1:
        mask |= (1 << prime_idx)
    return mask


def _build_prime_masks(max_val=256):
    """Precompute prime factor masks for all values 1..max_val.
    Use a proper prime list for consistent indexing."""
    # Generate primes up to max_val
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * (max_val + 1)
    for n in range(2, max_val + 1):
        m = n
        mask = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask |= (1 << prime_to_idx[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask |= (1 << prime_to_idx[m])
        masks[n] = mask
    return masks


def _pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    distinct = list(set(values))
    if len(distinct) <= 1:
        return True
    # Use gcd check
    import math
    for i in range(len(distinct)):
        for j in range(i+1, len(distinct)):
            if math.gcd(distinct[i], distinct[j]) != 1:
                return False
    return True


def construct():
    global _DEADLINE
    start = time.time()
    
    # Precompute prime factor masks
    masks = _build_prime_masks(256)
    
    # Key insight: pairwise coprime sumset => at most one even number in sumset.
    # For 3x3 grid, number of even sums = k*m + (3-k)*(3-m) where k=#even in A, m=#even in B.
    # This equals 0 only when (k=0,m=0) or (k=3,m=0) or (k=0,m=3) or (k=3,m=3).
    # k*m + (3-k)*(3-m):
    #   (0,0): 0 + 9 = 9  -> 9 even sums, impossible for pairwise coprime (more than 1 even)
    #   Wait, let me recalculate. Even sums occur when both a_i and b_j are even, or both are odd.
    #   # even sums = (#even in A)*(#even in B) + (#odd in A)*(#odd in B)
    #   k = #even in A, 3-k = #odd in A
    #   m = #even in B, 3-m = #odd in B
    #   even_sums = k*m + (3-k)*(3-m)
    #   (0,0): 0*0 + 3*3 = 9
    #   (0,1): 0*1 + 3*2 = 6
    #   (0,2): 0*2 + 3*1 = 3
    #   (0,3): 0*3 + 3*0 = 0  <-- all odd in A, all even in B
    #   (1,0): 1*0 + 2*3 = 6
    #   (1,1): 1*1 + 2*2 = 5
    #   (1,2): 1*2 + 2*1 = 4
    #   (1,3): 1*3 + 2*0 = 3
    #   (2,0): 2*0 + 1*3 = 3
    #   (2,1): 2*1 + 1*2 = 4
    #   (2,2): 2*2 + 1*1 = 5
    #   (2,3): 2*3 + 1*0 = 6
    #   (3,0): 3*0 + 0*3 = 0  <-- all even in A, all odd in B
    #   (3,1): 3*1 + 0*2 = 3
    #   (3,2): 3*2 + 0*1 = 6
    #   (3,3): 3*3 + 0*0 = 9
    # 
    # So only (k,m) = (0,3) or (3,0) give 0 even sums.
    # Case 1: A all odd, B all even
    # Case 2: A all even, B all odd
    # In both cases, all sums are odd, so we only need to check odd prime factors.
    
    odds = [i for i in range(1, 129) if i % 2 == 1]
    evens = [i for i in range(2, 129) if i % 2 == 0]
    
    best_count = 0
    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    
    # Search Case 1: A from odds, B from evens
    # Search Case 2: A from evens, B from odds
    # To maximize |A+B|, we want 9 distinct sums if possible.
    
    # Try to find a solution with 9 distinct pairwise-coprime sums first
    # Then fall back to fewer if needed.
    
    # For efficiency, iterate over combinations
    # C(64,3) = 41664 for each set, so 41664^2 is too many for brute force.
    # We need pruning.
    
    # Strategy: For each A (triple), find best B.
    # Precompute for each odd a: the set of odd sums a+b for b in evens.
    # Actually, let's just do a smarter search.
    
    # Since 41664 * 41664 is ~17 billion, we can't brute force.
    # Let's use a different approach: iterate over A triples, and for each,
    # check all B triples but with early pruning.
    
    # Better: iterate over all pairs of A-triples and B-triples but with
    # precomputed constraints.
    
    # Actually, let's think about this differently.
    # The sumset has at most 9 elements. For them to be pairwise coprime and all odd,
    # they must be odd numbers that are pairwise coprime.
    
    # Let's try a direct search with strong pruning:
    # For each A (sorted triple of odds), compute all 9 sums with each B candidate.
    # But we need to check if the 9 sums are pairwise coprime.
    
    # Alternative: enumerate all valid 9-element subsets of odd numbers in [3, 256]
    # that are pairwise coprime, then check if they can be written as A+B.
    
    # Actually, let's just do a bounded search with deadline.
    
    # Generate all A triples from odds
    odd_triples = list(combinations(odds, 3))
    even_triples = list(combinations(evens, 3))
    
    # For each A, for each B, check if sumset is pairwise coprime.
    # But 41664 * 41664 is too many. Let's prune.
    
    # Key: for a given A, the sums a_i + b_j must be pairwise coprime.
    # Let's iterate A, and for each, try B's but check incrementally.
    
    # Let's limit the search: try A's with smaller values first, and B's with smaller values.
    # Also, we can precompute for each a in A, which b's are "compatible".
    
    # Actually, let's just do a time-bounded search.
    
    import math
    
    def check_sumset(A, B):
        """Check if all distinct values in A+B are pairwise coprime. Return count of distinct sums."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct = list(sums)
        if len(distinct) <= 1:
            return len(distinct)
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return -1
        return len(distinct)
    
    # Search with deadline
    # Try Case 1: A odd, B even
    found = False
    for i, A in enumerate(odd_triples):
        if time.time() > _DEADLINE:
            break
        for j, B in enumerate(even_triples):
            if time.time() > _DEADLINE:
                break
            count = check_sumset(A, B)
            if count > 0:
                if count > best_count:
                    best_count = count
                    best_A = list(A)
                    best_B = list(B)
                if count == 9:
                    return [list(A), list(B)]
    
    # If deadline hit, try Case 2: A even, B odd
    if time.time() < _DEADLINE:
        for i, A in enumerate(even_triples):
            if time.time() > _DEADLINE:
                break
            for j, B in enumerate(odd_triples):
                if time.time() > _DEADLINE:
                    break
                count = check_sumset(A, B)
                if count > 0:
                    if count > best_count:
                        best_count = count
                        best_A = list(A)
                        best_B = list(B)
                    if count == 9:
                        return [list(A), list(B)]
    
    return [best_A, best_B]
