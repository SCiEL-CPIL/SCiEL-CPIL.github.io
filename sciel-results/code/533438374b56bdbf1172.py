"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum is 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[i] == i:
        for j in range(i * i, _MAX_SUM + 1, i):
            if _sieve[j] == j:
                _sieve[j] = i

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed sieve."""
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

# Precompute prime factors for all sums 2..20000
_sum_pf = [None] * (_MAX_SUM + 1)
for s in range(2, _MAX_SUM + 1):
    _sum_pf[s] = _prime_factors_fast(s)

def _compute_score(nums):
    """Compute the number of distinct primes across all pairwise sums."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= _sum_pf[nums[i] + nums[j]]
    return len(all_primes)

def _compute_prime_counts(nums):
    """For each prime, count how many pairs contain it. Return dict prime->count."""
    from collections import defaultdict
    counts = defaultdict(int)
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            for p in _sum_pf[nums[i] + nums[j]]:
                counts[p] += 1
    return counts

def construct():
    random.seed(42)
    
    # Strategy: find 12 numbers whose pairwise sums use as few distinct primes as possible.
    # Key insight: if all numbers are multiples of a small number m, sums are multiples of m,
    # reducing prime variety. Also, using numbers close together keeps sums in a narrow range.
    
    # Good initial candidates
    candidates = []
    
    # Multiples of small highly-composite numbers
    for m in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24]:
        # Pick 12 distinct multiples of m
        base = m
        cand = []
        for k in range(1, 200):
            v = m * k
            if v <= 10000 and v not in cand:
                cand.append(v)
            if len(cand) == 12:
                break
        if len(cand) == 12:
            candidates.append(cand)
    
    # Consecutive multiples
    for m in range(2, 30):
        for start in range(1, 50):
            cand = [m * (start + i) for i in range(12)]
            if all(1 <= x <= 10000 for x in cand) and len(set(cand)) == 12:
                candidates.append(cand)
    
    # Small numbers
    candidates.append(list(range(1, 13)))
    candidates.append(list(range(2, 25, 2)))
    candidates.append(list(range(1, 24, 2)))
    
    # Numbers that are products of small primes
    small_primes = [2, 3, 5, 7]
    for a in range(1, 20):
        for b in range(1, 20):
            cand = []
            for x in range(1, 100):
                v = a * x
                if v <= 10000 and v not in cand:
                    cand.append(v)
                if len(cand) == 12:
                    break
            if len(cand) == 12:
                candidates.append(cand)
    
    # Evaluate all candidates
    best_candidate = None
    best_score = float('inf')
    
    for cand in candidates:
        if _deadline_hit():
            break
        if len(set(cand)) != 12:
            continue
        if not all(1 <= x <= 10000 for x in cand):
            continue
        score = _compute_score(cand)
        if score < best_score:
            best_score = score
            best_candidate = cand[:]
    
    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_score = _compute_score(best_candidate)
    
    # Local search with better candidate generation
    current = best_candidate[:]
    current_score = best_score
    n = 12
    
    # Track which primes are present and their counts
    prime_counts = _compute_prime_counts(current)
    
    def _recompute_prime_counts(nums):
        from collections import defaultdict
        counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                for p in _sum_pf[nums[i] + nums[j]]:
                    counts[p] += 1
        return counts
    
    # Simulated annealing / hill climbing
    temp = 100.0
    cooling = 0.995
    
    for step in range(20000):
        if _deadline_hit():
            break
        
        # Pick a random index to change
        idx = random.randint(0, n - 1)
        old_val = current[idx]
        
        # Generate candidate values
        new_val = None
        r = random.random()
        if r < 0.3:
            # Small perturbation
            delta = random.randint(-20, 20)
            new_val = old_val + delta
        elif r < 0.5:
            # Medium perturbation
            delta = random.randint(-100, 100)
            new_val = old_val + delta
        elif r < 0.7:
            # Try nearby multiple of a small number
            m = random.choice([2, 3, 4, 5, 6, 7, 8, 10, 12])
            base = (old_val // m) * m
            new_val = base + random.choice([-m, m, 2*m, -2*m])
        else:
            # Random value
            new_val = random.randint(1, 10000)
        
        if new_val is None or new_val < 1 or new_val > 10000:
            continue
        if new_val == old_val:
            continue
        if new_val in current:
            continue
        
        # Compute new score
        new_primes = set()
        for j in range(n):
            if j == idx:
                continue
            s = new_val + current[j]
            new_primes |= _sum_pf[s]
        
        # Primes from pairs not involving idx
        other_primes = set()
        for i in range(n):
            if i == idx:
                continue
            for j in range(i + 1, n):
                if j == idx:
                    continue
                other_primes |= _sum_pf[current[i] + current[j]]
        
        total_primes = new_primes | other_primes
        new_score = len(total_primes)
        
        # Accept if better, or with probability exp(-delta/temp)
        delta = new_score - current_score
        if delta < 0:
            current[idx] = new_val
            current_score = new_score
        elif temp > 0.1:
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                current[idx] = new_val
                current_score = new_score
        
        temp *= cooling
    
    # Final greedy improvement pass
    improved = True
    while improved and not _deadline_hit():
        improved = False
        for idx in range(n):
            if _deadline_hit():
                break
            old_val = current[idx]
            best_new_val = None
            best_new_score = current_score
            
            # Try values in a range around old_val
            for new_val in range(max(1, old_val - 100), min(10000, old_val + 100) + 1):
                if new_val == old_val:
                    continue
                if new_val in current:
                    continue
                
                new_primes = set()
                for j in range(n):
                    if j == idx:
                        continue
                    s = new_val + current[j]
                    new_primes |= _sum_pf[s]
                
                other_primes = set()
                for i in range(n):
                    if i == idx:
                        continue
                    for j in range(i + 1, n):
                        if j == idx:
                            continue
                        other_primes |= _sum_pf[current[i] + current[j]]
                
                total_primes = new_primes | other_primes
                new_score = len(total_primes)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
            
            if best_new_val is not None and best_new_score < current_score:
                current[idx] = best_new_val
                current_score = best_new_score
                improved = True
    
    # Final verification
    result = current
    assert len(set(result)) == 12
    assert all(1 <= x <= 10000 for x in result)
    
    return result
