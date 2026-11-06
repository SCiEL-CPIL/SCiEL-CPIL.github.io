"""Program to minimize distinct prime factors across pairwise sums."""
import random
import math
import time
from itertools import combinations

def _sieve_smallest_prime(limit):
    """Return list spf where spf[i] is smallest prime factor of i."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _prime_factors(n, spf):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _count_distinct_primes(arr, spf):
    """Count distinct primes dividing all pairwise sums of arr."""
    prime_set = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                prime_set.add(p)
                while n % p == 0:
                    n //= p
    return len(prime_set)

def _is_valid_for_primes(arr, target_primes, spf):
    """Check if all pairwise sums have prime factors only in target_primes."""
    target_set = set(target_primes)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                if p not in target_set:
                    return False
                while n % p == 0:
                    n //= p
    return True

def _find_bad_pair(arr, target_primes, spf):
    """Find a pair (i, j) such that arr[i] + arr[j] has a prime factor not in target_primes."""
    target_set = set(target_primes)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                if p not in target_set:
                    return (i, j)
                while n % p == 0:
                    n //= p
    return None

def _local_search_for_target(arr, target_primes, spf, max_iters=1000):
    """
    Local search to modify arr such that all pairwise sums have prime factors in target_primes.
    Returns (True, arr) if valid, (False, arr) otherwise.
    """
    n = len(arr)
    target_set = set(target_primes)
    current_arr = list(arr)
    
    for iter_count in range(max_iters):
        bad_pair = _find_bad_pair(current_arr, target_set, spf)
        if bad_pair is None:
            return True, current_arr
        
        i, j = bad_pair
        idx_to_replace = random.choice([i, j])
        old_val = current_arr[idx_to_replace]
        
        found_replacement = False
        # Try a few random replacements
        for _ in range(10):
            new_val = random.randint(1, 10000)
            if new_val == old_val or new_val in current_arr:
                continue
            
            compatible = True
            for k in range(n):
                if k == idx_to_replace:
                    continue
                s = new_val + current_arr[k]
                tmp = s
                while tmp > 1:
                    p = spf[tmp]
                    if p not in target_set:
                        compatible = False
                        break
                    while tmp % p == 0:
                        tmp //= p
                if not compatible:
                    break
            
            if compatible:
                current_arr[idx_to_replace] = new_val
                found_replacement = True
                break
        
        if not found_replacement:
            # Try the other element in the bad pair
            other_idx = j if idx_to_replace == i else i
            old_val = current_arr[other_idx]
            for _ in range(10):
                new_val = random.randint(1, 10000)
                if new_val == old_val or new_val in current_arr:
                    continue
                
                compatible = True
                for k in range(n):
                    if k == other_idx:
                        continue
                    s = new_val + current_arr[k]
                    tmp = s
                    while tmp > 1:
                        p = spf[tmp]
                        if p not in target_set:
                            compatible = False
                            break
                        while tmp % p == 0:
                            tmp //= p
                    if not compatible:
                        break
                
                if compatible:
                    current_arr[other_idx] = new_val
                    found_replacement = True
                    break
            
            if not found_replacement:
                return False, current_arr

    return False, current_arr

def construct():
    n = 12
    max_val = 10000
    
    # Precompute smallest prime factor up to 20000
    limit = 2 * max_val
    spf = _sieve_smallest_prime(limit)
    
    # Strategy: Try to find a solution with a small set of prime factors.
    small_primes = [2, 3, 5, 7, 11, 13]
    
    # Generate candidate sets of primes, sorted by size
    candidates = []
    for size in range(2, 5): # Try sizes 2, 3, 4
        for combo in combinations(small_primes, size):
            candidates.append(list(combo))
            
    candidates.sort(key=lambda x: (len(x), sum(x)))
    
    # Initial fallback
    best_arr = list(range(1, n + 1))
    best_score = _count_distinct_primes(best_arr, spf)
    
    start_time = time.time()
    time_limit = 55.0
    
    random.seed(42)
    
    # Try specific prime sets
    for target_primes in candidates:
        if time.time() - start_time > time_limit:
            break
            
        success = False
        valid_arr = None
        
        max_restarts = 5
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            arr_init = [random.randint(1, max_val) for _ in range(n)]
            while len(set(arr_init)) < n:
                for i in range(n):
                    if arr_init.count(arr_init[i]) > 1:
                        arr_init[i] = random.randint(1, max_val)
                        
            found, res_arr = _local_search_for_target(arr_init, target_primes, spf, max_iters=500)
            
            if found:
                success = True
                valid_arr = res_arr
                break
        
        if success:
            if len(target_primes) < best_score:
                best_score = len(target_primes)
                best_arr = valid_arr
            break # Found minimal size for this iteration order
            
    # Fallback generic local search if not improved or if we want to try harder
    if best_score > 2:
        improved = True
        max_iterations = 10000
        iter_count = 0
        while improved and iter_count < max_iterations:
            if time.time() - start_time > time_limit:
                break
            improved = False
            iter_count += 1
            for i in range(n):
                old_val = best_arr[i]
                for _ in range(5):
                    new_val = random.randint(1, max_val)
                    if new_val == old_val or new_val in best_arr:
                        continue
                    trial = list(best_arr)
                    trial[i] = new_val
                    new_score = _count_distinct_primes(trial, spf)
                    if new_score < best_score:
                        best_score = new_score
                        best_arr = trial
                        improved = True
                        break

    return best_arr
