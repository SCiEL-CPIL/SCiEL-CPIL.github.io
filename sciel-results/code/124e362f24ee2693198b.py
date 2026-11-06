"""Optimized program: search for 12 distinct positive integers <=10000 minimizing distinct primes in pair sums."""
import time
import random
from itertools import combinations
from functools import lru_cache

# Precompute smallest-prime-factor table up to 20000 (max sum = 10000+9999=19999)
_MAX_SUM = 20000
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM**0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

def _prime_factors(n):
    """Return set of prime factors of n using SPF table."""
    factors = set()
    while n > 1:
        p = _spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

# Cache for prime factors of sums - this is the key speedup
_pf_cache = {}

def _prime_factors_cached(n):
    """Return set of prime factors of n using cache + SPF table."""
    if n in _pf_cache:
        return _pf_cache[n]
    factors = _prime_factors(n)
    _pf_cache[n] = factors
    return factors

def _distinct_primes_in_sums(arr):
    """Count distinct primes dividing any sum of two distinct elements."""
    primes_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            primes_set.update(_prime_factors_cached(s))
    return primes_set

def _objective(arr):
    return len(_distinct_primes_in_sums(arr))

def _generate_structured_candidates():
    """Generate diverse structured candidate arrays with high probability of low prime union."""
    structured = []
    
    # 1. Arithmetic Progressions: a + i*d
    # Sums are 2a + (i+j)d. If d is small and a is chosen well, factors might be limited.
    for d in range(1, 200):
        for a in range(1, 200):
            arr = [a + i * d for i in range(12)]
            if max(arr) <= 10000 and len(set(arr)) == 12:
                structured.append(arr)
                
    # 2. Multiples of small bases
    # If all numbers are multiples of B, sums are multiples of B.
    # The prime factors of the sums will be Primes(B) U Primes(sums/B).
    for base in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 40, 48, 60]:
        max_mult = 10000 // base
        if max_mult < 12:
            continue
        # Regular multiples
        for start in range(1, 50):
            if start + 11 > max_mult:
                break
            arr = [base * (start + i) for i in range(12)]
            if len(set(arr)) == 12:
                structured.append(arr)
        
        # Random multiples
        for _ in range(20):
            mults = random.sample(range(1, max_mult + 1), 12)
            arr = [base * m for m in mults]
            if len(set(arr)) == 12:
                structured.append(arr)

    # 3. Numbers with specific modular properties
    # E.g., all numbers congruent to 1 mod 3 -> sums are 2 mod 3 (not divisible by 3)
    # All numbers odd -> sums even (divisible by 2)
    
    # Try to force sums to be divisible by only 2, 3, 5
    # If we pick numbers such that sums are multiples of 30, the prime factors of the quotients matter.
    
    # 4. Small integers
    # Often the optimal solution uses small numbers because large numbers introduce larger prime factors.
    for _ in range(100):
        arr = random.sample(range(1, 100), 12)
        structured.append(arr)
        
    for _ in range(100):
        arr = random.sample(range(2, 200, 2), 12) # All even
        structured.append(arr)

    for _ in range(100):
        arr = random.sample(range(1, 200, 3), 12) # All div by 3
        structured.append(arr)

    for _ in range(100):
        # Mix of small even and small multiples of 3
        candidates = list(range(2, 100, 2)) + list(range(3, 100, 3))
        candidates = list(set(candidates))
        arr = random.sample(candidates, 12)
        structured.append(arr)

    return structured

def _local_search(current, current_score, deadline, best_score, best_arr):
    """Local search with smart perturbations. Returns (best_score, best_arr)."""
    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        if no_improve_count > 20:
            break
            
        # Try single-element changes
        for i in range(12):
            # Strategy 1: Random small value (bias towards small numbers)
            if random.random() < 0.7:
                new_val = random.randint(1, 500)
            else:
                new_val = random.randint(1, 10000)
                
            if new_val in current and new_val != current[i]:
                continue
            cand = list(current)
            cand[i] = new_val
            if len(set(cand)) != 12 or max(cand) > 10000:
                continue
            score = _objective(cand)
            if score < current_score:
                current = cand
                current_score = score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_arr = list(current)
                break
            elif score == current_score and random.random() < 0.1:
                # Accept neutral moves to escape local minima (simulated annealing-like)
                # But be cautious, we want to minimize, so only accept if it helps diversity
                pass

            # Strategy 2: Multiple of a small number
            if not improved:
                base = random.choice([2, 3, 4, 5, 6, 10, 12, 15, 20, 30])
                m = random.randint(1, min(10000 // base, 500))
                new_val = base * m
                if new_val in current and new_val != current[i]:
                    continue
                cand = list(current)
                cand[i] = new_val
                if len(set(cand)) != 12 or max(cand) > 10000:
                    continue
                score = _objective(cand)
                if score < current_score:
                    current = cand
                    current_score = score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = list(current)
                    break

        # Try swapping two elements with random candidates
        if not improved:
            for _ in range(10):
                i, j = random.sample(range(12), 2)
                cand = list(current)
                # Generate new values for i and j
                if random.random() < 0.5:
                    new_vi = random.randint(1, 200)
                else:
                    new_vi = random.choice([2,3,4,5,6,10,12,15,20,30]) * random.randint(1, 100)
                    
                if random.random() < 0.5:
                    new_vj = random.randint(1, 200)
                else:
                    new_vj = random.choice([2,3,4,5,6,10,12,15,20,30]) * random.randint(1, 100)
                
                # Check validity
                if new_vi == new_vj: continue
                if (new_vi in current and new_vi != current[i]) or (new_vj in current and new_vj != current[j]):
                     # Handle duplicates carefully
                     pass # Simplified: just skip if obvious conflict, local search handles the rest via scoring
                     
                cand[i] = new_vi
                cand[j] = new_vj
                if len(set(cand)) != 12 or max(cand) > 10000:
                    continue
                score = _objective(cand)
                if score < current_score:
                    current = cand
                    current_score = score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = list(current)
                    break

    return best_score, best_arr

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # --- Phase 1: Structured Constructions ---
    best_arr = None
    best_score = float('inf')
    
    structured = _generate_structured_candidates()
    
    # Shuffle to avoid bias
    random.shuffle(structured)
    
    for arr in structured:
        if time.time() > deadline - 15:
            break
        arr = list(dict.fromkeys(arr))
        if len(arr) != 12:
            continue
        if any(x < 1 or x > 10000 for x in arr):
            continue
        score = _objective(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
            # If we find a very good score (e.g., <= 4), we can stop early
            if best_score <= 4:
                break
    
    if best_arr is None:
        # Fallback
        best_arr = list(range(1, 13))
        best_score = _objective(best_arr)
    
    # --- Phase 2: Local Search from Best ---
    if time.time() < deadline - 10:
        current = list(best_arr)
        current_score = best_score
        # Run local search for a fixed amount of time
        local_deadline = min(deadline - 5, start_time + 30)
        best_score, best_arr = _local_search(current, current_score, local_deadline, best_score, best_arr)
    
    # --- Phase 3: Random Restarts ---
    if time.time() < deadline - 5:
        restarts_done = 0
        while time.time() < deadline - 3 and restarts_done < 500:
            restarts_done += 1
            
            # Generate diverse random candidates
            r = random.random()
            if r < 0.2:
                # Small numbers
                arr = random.sample(range(1, 100), 12)
            elif r < 0.4:
                # Small even numbers
                arr = random.sample(range(2, 300, 2), 12)
            elif r < 0.6:
                # Multiples of 3
                arr = random.sample(range(3, 300, 3), 12)
            elif r < 0.8:
                # Multiples of 6 (LCM of 2,3)
                arr = random.sample(range(6, 500, 6), 12)
            else:
                # Random multiples of a random small base
                base = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
                max_m = 10000 // base
                if max_m < 12:
                    base = 2
                    max_m = 5000
                mults = random.sample(range(1, max_m + 1), 12)
                arr = [base * m for m in mults]
            
            if len(set(arr)) != 12 or any(x < 1 or x > 10000 for x in arr):
                continue
            
            score = _objective(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            
            # Quick local search from this candidate
            if time.time() < deadline - 3:
                cur = list(arr)
                cur_score = score
                # Short local search
                search_deadline = min(deadline - 3, time.time() + 1.5)
                best_score, best_arr = _local_search(cur, cur_score, search_deadline, best_score, best_arr)

    return best_arr

# EVOLVE-BLOCK-END
