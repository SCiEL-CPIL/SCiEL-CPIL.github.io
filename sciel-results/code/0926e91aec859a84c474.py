"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            s = nums[i] + nums[j]
            all_primes |= _prime_factors(s)
    return len(all_primes)

def _count_distinct_primes_fast(nums):
    """Faster count using precomputed factorization for small numbers."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            s = nums[i] + nums[j]
            # Factorize s
            d = 2
            while d * d <= s:
                while s % d == 0:
                    all_primes.add(d)
                    s //= d
                d += 1 if d == 2 else 2
            if s > 1:
                all_primes.add(s)
    return len(all_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin
    
    n = 12
    maximum = 10000
    
    # Strategy: Use multiples of 6 to guarantee primes 2 and 3 in all sums.
    # Then local search on the k-values (where num = 6*k) to minimize additional primes.
    # If num_i = 6*k_i, then sum = 6*(k_i + k_j), so primes dividing sum are {2, 3} union primes dividing (k_i + k_j).
    # So we need to minimize distinct primes dividing (k_i + k_j) for all pairs.
    
    # First, try a systematic approach: pick k values that make k_i + k_j have few distinct prime factors.
    # If all k_i are multiples of some M, then k_i + k_j is multiple of M, reducing prime variety.
    
    # Let's try: all numbers are multiples of 6. So nums[i] = 6 * k[i] where k[i] are distinct positive integers.
    # 6*k <= 10000 => k <= 1666.
    
    # To minimize primes in sums k_i + k_j, we can try to make all k_i congruent modulo some number,
    # or use a structure where sums have few prime factors.
    
    # Approach: Start with a good initial set, then do local search.
    
    # Initial guess: use small k values that are all even (so k_i + k_j is even, adding prime 2 to the k-sums,
    # but 2 is already in the overall sum primes since 6 contributes 2 and 3).
    # Actually, the primes in the final sums are {2, 3} ∪ {primes dividing (k_i + k_j) for all i≠j}.
    # So we want to minimize |{primes dividing (k_i + k_j)}| for all pairs.
    
    # Let's try to make all k_i + k_j have only a small set of prime factors.
    # If we pick k_i all to be multiples of 2, then k_i + k_j is even, so 2 is always a factor.
    # If we also make them multiples of 3, then k_i + k_j is divisible by 3.
    # So if k_i are multiples of 6, then k_i + k_j is divisible by 6, and the primes in the final sums are {2,3} ∪ {primes in (k_i+k_j)/6 ... }
    # Wait, no. Let me re-think.
    
    # nums[i] = 6 * k[i]. Sum = 6*(k[i]+k[j]). Prime factors of 6*(k[i]+k[j]) = {2,3} ∪ prime_factors(k[i]+k[j]).
    # So total distinct primes = |{2,3} ∪ (∪_pairs prime_factors(k[i]+k[j]))|.
    
    # To minimize this, we want the union of prime_factors(k[i]+k[j]) to be as small as possible, 
    # and ideally a subset of {2,3} or a small set.
    
    # If we make all k[i] equal, that's not allowed (distinct). 
    # If we make k[i] all even, then k[i]+k[j] is even, so 2 is always in the union.
    # If we make k[i] all ≡ 0 mod 3, then k[i]+k[j] ≡ 0 mod 3, so 3 is always in the union.
    # So if k[i] are all multiples of 6, then k[i]+k[j] is always divisible by 6, so {2,3} ⊆ union of prime_factors(k[i]+k[j]).
    # Then total primes = |{2,3} ∪ (∪ prime_factors(k[i]+k[j]))| = |∪ prime_factors(k[i]+k[j])| since 2,3 are already included.
    
    # So the problem reduces to: pick 12 distinct multiples of 6 (i.e., k values that are multiples of 6) 
    # to minimize the number of distinct prime factors of all pairwise sums k[i]+k[j].
    
    # But wait, k[i] don't have to be multiples of 6. Let me reconsider.
    
    # Actually, let's just do a local search on the k values directly.
    
    # Start with k values: 1, 2, 3, ..., 12 (so nums = 6, 12, 18, ..., 72)
    best_k = list(range(1, 13))
    best_score = _count_distinct_primes_fast([6*k for k in best_k])
    
    # Local search: try swapping k[i] with k[j] or changing one k value.
    max_k = 1666  # since 6*1666 = 9996 <= 10000
    
    random.seed(42)
    
    def _score_k(k_list):
        return _count_distinct_primes_fast([6*k for k in k_list])
    
    # Try to improve via local search
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Try random perturbations
        for _ in range(5000):
            if time.time() > deadline - 10.0:
                break
            new_k = list(best_k)
            # Pick a random index and change its value
            idx = random.randint(0, n-1)
            old_val = new_k[idx]
            # Try a nearby value or random value
            if random.random() < 0.5:
                new_val = old_val + random.randint(-50, 50)
            else:
                new_val = random.randint(1, max_k)
            
            if new_val <= 0 or new_val > max_k:
                continue
            if new_val in new_k:  # must be distinct
                continue
            new_k[idx] = new_val
            
            new_score = _score_k(new_k)
            if new_score < best_score:
                best_score = new_score
                best_k = new_k
                improved = True
    
    # Also try: make all k values multiples of 2 (even) to ensure 2 is in all k-sums
    # and multiples of 3 to ensure 3 is in all k-sums.
    # Try k values that are multiples of 6.
    candidate_k = list(range(6, 6*n+1, 6))  # 6, 12, 18, ..., 72
    candidate_score = _score_k(candidate_k)
    if candidate_score < best_score:
        best_score = candidate_score
        best_k = candidate_k
    
    # Try another structure: k values all ≡ 1 mod 6 or some other residue
    for residue in range(6):
        candidate_k = []
        k = residue
        while len(candidate_k) < n:
            if k > 0 and k <= max_k:
                candidate_k.append(k)
            k += 6
        if len(candidate_k) >= n:
            candidate_k = candidate_k[:n]
            candidate_score = _score_k(candidate_k)
            if candidate_score < best_score:
                best_score = candidate_score
                best_k = candidate_k
    
    # Final local search with more iterations
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for _ in range(10000):
            if time.time() > deadline - 5.0:
                break
            new_k = list(best_k)
            idx = random.randint(0, n-1)
            old_val = new_k[idx]
            new_val = old_val + random.randint(-20, 20)
            
            if new_val <= 0 or new_val > max_k:
                continue
            if new_val in new_k:
                continue
            new_k[idx] = new_val
            
            new_score = _score_k(new_k)
            if new_score < best_score:
                best_score = new_score
                best_k = new_k
                improved = True
    
    # Convert back to nums
    result = [6*k for k in best_k]
    
    # Verify distinctness and bounds
    assert len(set(result)) == n
    assert all(0 < x <= maximum for x in result)
    
    return result
# EVOLVE-BLOCK-END
