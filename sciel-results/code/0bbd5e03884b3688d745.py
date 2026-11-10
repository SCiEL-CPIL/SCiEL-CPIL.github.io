"""Combined approach: structural constraint via modular arithmetic + fast incremental local search.

Key structural idea: if all numbers are multiples of M, then every pairwise sum is a multiple of M,
so all prime factors of M are 'free' (always present). We want M to have few prime factors, and we
want the quotients (sum / M) to introduce as few additional primes as possible.

The best structural choice is to pick M such that:
  - M has few prime factors (e.g., M = 2, 3, 6, 12, 30, 60, ...)
  - The 12 numbers are M * a_i where a_i are small distinct positive integers
  - The sums a_i + a_j (for i != j) have prime factors from a small set

We then do fast local search to refine the a_i values (and potentially the base M).

The incremental scoring uses a precomputed sieve for speed.
"""
import time
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum = 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _sieve[_j] == _j:
                _sieve[_j] = _i

def _prime_factors_fast(n):
    """Return set of prime factors of n using precomputed sieve."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _score_full(nums):
    """Compute number of distinct prime factors across all pairwise sums."""
    n = len(nums)
    all_primes = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= _prime_factors_fast(nums[i] + nums[j])
    return len(all_primes)

def _pair_factor_matrix(nums):
    """Build 12x12 matrix of prime factor sets for each pair."""
    n = len(nums)
    pf = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            f = _prime_factors_fast(s)
            pf[i][j] = f
            pf[j][i] = f
    return pf

def _total_from_matrix(pf):
    """Count distinct primes from a 12x12 pair factor matrix."""
    n = len(pf)
    all_primes = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= pf[i][j]
    return len(all_primes)

def _try_improve_local(nums, pf, current_score):
    """Hill-climb: try changing each element to a nearby value. Returns (new_nums, new_pf, new_score, improved)."""
    n = len(nums)
    improved = True
    while improved and not _deadline_hit():
        improved = False
        for idx in range(n):
            if _deadline_hit():
                break
            old_val = nums[idx]
            # Try a set of candidate replacements
            candidates = set()
            # Small perturbations
            for d in range(-30, 31):
                v = old_val + d
                if 1 <= v <= 10000 and v != old_val:
                    candidates.add(v)
            # Some random candidates
            for _ in range(15):
                candidates.add(random.randint(1, 10000))
            
            best_new_val = None
            best_new_score = current_score
            best_new_pf = None
            
            for new_val in candidates:
                if new_val in nums:
                    continue
                # Compute new score: only pairs involving idx change
                new_primes_involving = set()
                for j in range(n):
                    if j == idx:
                        continue
                    s = new_val + nums[j]
                    new_primes_involving |= _prime_factors_fast(s)
                
                # Primes from pairs NOT involving idx
                other_primes = set()
                for i in range(n):
                    if i == idx:
                        continue
                    for j in range(i + 1, n):
                        if j == idx:
                            continue
                        other_primes |= pf[i][j]
                
                total = new_primes_involving | other_primes
                new_score = len(total)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
            
            if best_new_val is not None:
                # Apply the change
                for j in range(n):
                    if j != idx:
                        s = best_new_val + nums[j]
                        f = _prime_factors_fast(s)
                        pf[idx][j] = f
                        pf[j][idx] = f
                nums[idx] = best_new_val
                current_score = best_new_score
                improved = True
                break  # restart the outer loop
    
    return nums, pf, current_score

def construct():
    random.seed(42)
    
    n = 12
    best_nums = None
    best_score = float('inf')
    
    # Structural insight: try different multipliers M.
    # If nums = M * a_i, then sums = M * (a_i + a_j).
    # Prime factors = primes(M) union primes(a_i + a_j for all i<j).
    # We want primes(M) to be small in count, and primes of the quotients to be small.
    
    # Candidate multipliers: numbers with few prime factors
    multipliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]
    
    # For each multiplier, try a base set of small a_i and then local search
    for M in multipliers:
        if _deadline_hit():
            break
        
        # Base a_i: 1, 2, ..., 12 (gives nums = M, 2M, ..., 12M)
        # But we need nums <= 10000, so max a_i <= 10000 // M
        max_a = 10000 // M
        if max_a < 12:
            continue
        
        base_a = list(range(1, 13))
        nums = [M * a for a in base_a]
        
        # Also try shifted/scaled variants
        variants = [nums[:]]
        
        # Try a_i = 1..12 but with some randomization
        for _ in range(3):
            a_set = random.sample(range(1, min(max_a + 1, 100)), 12)
            variants.append([M * a for a in sorted(a_set)])
        
        # Try a_i centered around a small value
        for center in [10, 20, 50]:
            if center + 6 > max_a:
                continue
            a_set = [center + i for i in range(-6, 7)]
            if all(a >= 1 for a in a_set):
                variants.append([M * a for a in a_set])
        
        for cand in variants:
            if _deadline_hit():
                break
            if len(set(cand)) != 12:
                continue
            if not all(1 <= x <= 10000 for x in cand):
                continue
            
            score = _score_full(cand)
            if score < best_score:
                best_score = score
                best_nums = cand[:]
        
        # Now do local search on the best variant for this M
        if best_nums and best_score < float('inf'):
            # Find the best starting point for this M
            # Re-evaluate all variants to find the best
            pass
    
    # Now do a global local search from the best found so far
    if best_nums is None:
        best_nums = list(range(2, 25, 2))
        best_score = _score_full(best_nums)
    
    # Multiple restarts with local search
    for restart in range(300):
        if _deadline_hit():
            break
        
        if restart == 0:
            # Start from the best found
            nums = best_nums[:]
            pf = _pair_factor_matrix(nums)
            current_score = best_score
        else:
            # Random restart with structural bias
            M = random.choice(multipliers)
            max_a = 10000 // M
            if max_a < 12:
                continue
            a_set = random.sample(range(1, min(max_a + 1, 200)), 12)
            nums = [M * a for a in sorted(a_set)]
            if len(set(nums)) != 12 or not all(1 <= x <= 10000 for x in nums):
                continue
            pf = _pair_factor_matrix(nums)
            current_score = _total_from_matrix(pf)
        
        # Local search
        nums, pf, current_score = _try_improve_local(nums, pf, current_score)
        
        if current_score < best_score:
            best_score = current_score
            best_nums = nums[:]
        
        # Early stop if we've found a very good solution
        if best_score <= 4:
            break
    
    # Final validation
    assert len(set(best_nums)) == 12
    assert all(1 <= x <= 10000 for x in best_nums)
    
    return best_nums

# EVOLVE-BLOCK-END
