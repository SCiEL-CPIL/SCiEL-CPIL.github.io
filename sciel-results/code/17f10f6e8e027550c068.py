"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 80.0

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

def _prime_factors_fast_tuple(n):
    """Return tuple of prime factors of n (for faster set operations)."""
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _compute_factor_sets(nums):
    """For each pair (i,j), compute the set of prime factors of nums[i]+nums[j].
    Return a list of 12 sets, where set[i] contains all prime factors of sums involving index i."""
    n = len(nums)
    pair_factors = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            pf = _prime_factors_fast(s)
            pair_factors[i][j] = pf
            pair_factors[j][i] = pf
    return pair_factors

def _total_factor_count(pair_factors):
    """Count distinct primes across all pair factor sets."""
    all_primes = set()
    n = len(pair_factors)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= pair_factors[i][j]
    return len(all_primes)

def _build_prime_set(nums):
    """Return the set of all distinct primes dividing any pair sum."""
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            all_primes |= _prime_factors_fast(nums[i] + nums[j])
    return all_primes

# Precompute smooth numbers (2,3,5,7-smooth) up to 20000 for faster candidate generation
_smooth_numbers = []
for _a in range(15):
    _p2 = 2**_a
    if _p2 > 20000:
        break
    for _b in range(12):
        _p3 = 3**_b
        if _p2 * _p3 > 20000:
            break
        for _c in range(9):
            _p5 = 5**_c
            if _p2 * _p3 * _p5 > 20000:
                break
            for _d in range(7):
                _p7 = 7**_d
                _smooth = _p2 * _p3 * _p5 * _p7
                if _smooth > 20000:
                    break
                _smooth_numbers.append(_smooth)
_smooth_numbers = sorted(set(_smooth_numbers))

def construct():
    random.seed(42)
    
    # Generate initial candidates
    candidates = []
    
    # Strategy 1: Powers of 2
    candidates.append([2**i for i in range(1, 13)])
    
    # Strategy 2: Multiples of highly composite numbers
    for base in [30, 60, 120, 210, 360, 840, 2520]:
        candidates.append([base * (i+1) for i in range(12)])
    
    # Strategy 3: Small consecutive numbers
    candidates.append(list(range(1, 13)))
    candidates.append(list(range(2, 14)))
    candidates.append(list(range(10, 22)))
    
    # Strategy 4: Even numbers
    candidates.append(list(range(2, 25, 2)))
    candidates.append(list(range(10, 33, 2)))
    
    # Strategy 5: Numbers of form 2^a * 3^b
    cand_23 = []
    for a in range(15):
        for b in range(12):
            val = 2**a * 3**b
            if 1 <= val <= 10000 and val not in cand_23:
                cand_23.append(val)
            if len(cand_23) >= 12:
                break
        if len(cand_23) >= 12:
            break
    if len(cand_23) >= 12:
        candidates.append(cand_23[:12])
    
    # Strategy 6: Arithmetic progressions with lcm-based differences
    for d in [30, 60, 120, 210, 420]:
        for a in [1, 2, 3, 5, 10, 15, 30]:
            cand = [a + d * i for i in range(12)]
            if all(1 <= x <= 10000 for x in cand) and len(set(cand)) == 12:
                candidates.append(cand)
    
    # Strategy 7: Use smooth numbers as candidates
    if len(_smooth_numbers) >= 12:
        candidates.append(_smooth_numbers[:12])
        candidates.append(_smooth_numbers[5:17])
        candidates.append(_smooth_numbers[10:22])
    
    # Strategy 8: Numbers that are multiples of small primes
    for p in [2, 3, 5, 6, 10, 12, 15, 20, 30]:
        cand = []
        for k in range(1, 100):
            val = p * k
            if 1 <= val <= 10000 and val not in cand:
                cand.append(val)
            if len(cand) >= 12:
                break
        if len(cand) == 12:
            candidates.append(cand)
    
    # Evaluate all candidates
    best_candidate = None
    best_score = float('inf')
    best_pf = None
    
    for cand in candidates:
        if _deadline_hit():
            break
        if len(set(cand)) != 12:
            continue
        if not all(1 <= x <= 10000 for x in cand):
            continue
        pf = _compute_factor_sets(cand)
        score = _total_factor_count(pf)
        if score < best_score:
            best_score = score
            best_candidate = cand[:]
            best_pf = pf
    
    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_pf = _compute_factor_sets(best_candidate)
        best_score = _total_factor_count(best_pf)
    
    # Local search with efficient candidate generation
    current = best_candidate[:]
    current_pf = best_pf
    current_score = best_score
    n = 12
    
    # Precompute current prime set
    current_primes = set()
    for i in range(n):
        for j in range(i+1, n):
            current_primes |= current_pf[i][j]
    
    def _base_primes_excluding(pf, exclude_idx):
        """Get union of all pair factors excluding pairs involving exclude_idx."""
        base = set()
        for i in range(n):
            if i == exclude_idx:
                continue
            for j in range(i+1, n):
                if j == exclude_idx:
                    continue
                base |= pf[i][j]
        return base
    
    # Precompute base primes for each index
    base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
    
    # Precompute a fixed set of candidate values for faster search
    # Use smooth numbers and values near current elements
    fixed_cand_pool = set()
    for smooth in _smooth_numbers[:300]:
        fixed_cand_pool.add(smooth)
    for v in range(1, 200):
        fixed_cand_pool.add(v)
    # Add values that are multiples of small primes
    for p in [2, 3, 5, 6, 10, 12, 15, 20, 30]:
        for k in range(1, 50):
            fixed_cand_pool.add(p * k)
    fixed_cand_pool = list(fixed_cand_pool)
    
    for restart in range(200):
        if _deadline_hit():
            break
        
        improved = True
        while improved and not _deadline_hit():
            improved = False
            
            for idx in range(n):
                if _deadline_hit():
                    break
                
                old_val = current[idx]
                base_primes = base_primes_cache[idx]
                
                best_new_val = None
                best_new_score = current_score
                best_new_pf = None
                best_new_primes = None
                
                # Generate candidate values efficiently
                cand_vals = set()
                
                # Values near old value
                for d in range(-30, 31):
                    v = old_val + d
                    if 1 <= v <= 10000:
                        cand_vals.add(v)
                
                # Smooth-based candidates: smooth - current[j] for each j
                for j in range(n):
                    if j == idx:
                        continue
                    for smooth in _smooth_numbers[:100]:
                        v = smooth - current[j]
                        if 1 <= v <= 10000:
                            cand_vals.add(v)
                
                # Values that make sums divisible by small primes
                for target_prime in [2, 3, 5, 7]:
                    for j in range(n):
                        if j == idx:
                            continue
                        rem = (-current[j]) % target_prime
                        # Try a few values with this remainder
                        for k in range(max(0, rem - 20), min(10000, rem + 200), target_prime):
                            v = k + rem
                            if 1 <= v <= 10000:
                                cand_vals.add(v)
                        if len(cand_vals) > 500:
                            break
                    if len(cand_vals) > 500:
                        break
                
                # Add some from fixed pool
                for v in fixed_cand_pool[:100]:
                    if 1 <= v <= 10000:
                        cand_vals.add(v)
                
                # Random values
                for _ in range(20):
                    cand_vals.add(random.randint(1, 10000))
                
                # Remove values already in current
                current_set = set(current)
                cand_vals -= current_set
                
                # Limit candidates
                if len(cand_vals) > 400:
                    cand_vals = set(random.sample(list(cand_vals), 400))
                
                # Evaluate candidates
                for new_val in cand_vals:
                    if new_val in current:
                        continue
                    
                    new_primes = set(base_primes)
                    for j in range(n):
                        if j == idx:
                            continue
                        s = new_val + current[j]
                        pf = _prime_factors_fast(s)
                        new_primes |= pf
                    
                    new_score = len(new_primes)
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                        best_new_primes = new_primes
                        best_new_pf = [row[:] for row in current_pf]
                        for j in range(n):
                            if j != idx:
                                s = new_val + current[j]
                                pf = _prime_factors_fast(s)
                                best_new_pf[idx][j] = pf
                                best_new_pf[j][idx] = pf
                
                if best_new_val is not None and best_new_score < current_score:
                    current[idx] = best_new_val
                    current_pf = best_new_pf
                    current_score = best_new_score
                    current_primes = best_new_primes
                    base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
                    improved = True
                    break
            
            # If no improvement, try a random perturbation
            if not improved and not _deadline_hit():
                for _ in range(3):
                    if _deadline_hit():
                        break
                    new_cand = current[:]
                    num_changes = random.randint(1, 2)
                    for _ in range(num_changes):
                        idx = random.randint(0, 11)
                        new_cand[idx] = random.choice(fixed_cand_pool) if fixed_cand_pool else random.randint(1, 10000)
                    
                    if len(set(new_cand)) == 12 and all(1 <= x <= 10000 for x in new_cand):
                        new_pf = _compute_factor_sets(new_cand)
                        new_score = _total_factor_count(new_pf)
                        if new_score < current_score:
                            current = new_cand
                            current_pf = new_pf
                            current_score = new_score
                            base_primes_cache = [_base_primes_excluding(current_pf, i) for i in range(n)]
                            break
    
    # Final verification
    result = current
    if len(set(result)) != 12 or not all(1 <= x <= 10000 for x in result):
        result = [360 * (i+1) for i in range(12)]
    
    return result
