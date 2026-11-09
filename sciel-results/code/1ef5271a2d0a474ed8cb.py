import time
import random
from math import isqrt

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= _prime_factors(s)
    return len(all_primes)

def _evaluate(arr):
    return _count_distinct_primes(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Initial candidate: small numbers
    best = list(range(1, 13))
    best_score = _evaluate(best)
    
    # Try known good structure: numbers of form 2^a * 3^b
    # Generate candidates
    candidates = []
    for a in range(0, 14):
        for b in range(0, 10):
            v = (1 << a) * (3 ** b)
            if 1 <= v <= 10000:
                candidates.append(v)
    candidates = sorted(set(candidates))
    
    # Greedy: pick numbers that minimize added primes
    # Start with a random subset from candidates
    random.seed(42)
    
    # Try multiple random initializations
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initial set from candidates
        if len(candidates) >= 12:
            init = sorted(random.sample(candidates, 12))
        else:
            init = list(range(1, 13))
        
        score = _evaluate(init)
        if score < best_score:
            best = init[:]
            best_score = score
        
        # Local search: try swapping one element
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current = best[:]
            cur_score = best_score
            
            # Try replacing each element
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                old_val = current[i]
                # Try candidate values
                for new_val in candidates:
                    if new_val == old_val:
                        continue
                    if any(new_val == current[j] for j in range(12) if j != i):
                        continue
                    trial_arr = current[:]
                    trial_arr[i] = new_val
                    trial_arr.sort()
                    s = _evaluate(trial_arr)
                    if s < cur_score:
                        current = trial_arr
                        cur_score = s
                        improved = True
                        break
                if improved:
                    break
            
            if cur_score < best_score:
                best = current[:]
                best_score = cur_score
    
    # Final polish: try all pairs of candidates for each position
    if time.time() < deadline - 3:
        current = best[:]
        cur_score = best_score
        for i in range(12):
            if time.time() > deadline - 3:
                break
            old_val = current[i]
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if any(new_val == current[j] for j in range(12) if j != i):
                    continue
                trial_arr = current[:]
                trial_arr[i] = new_val
                s = _evaluate(trial_arr)
                if s < cur_score:
                    current = trial_arr
                    cur_score = s
    
    if cur_score < best_score:
        best = current[:]
        best_score = cur_score
    
    return best
