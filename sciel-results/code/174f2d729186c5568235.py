import time
import random
from math import gcd
from itertools import combinations

# Precompute LCM table for all pairs in [1, 128]
_LCM_TABLE = [[0] * 129 for _ in range(129)]
for i in range(1, 129):
    for j in range(1, 129):
        _LCM_TABLE[i][j] = i * j // gcd(i, j)

def _is_bad_triple(a, b, c):
    return _LCM_TABLE[a][b] == _LCM_TABLE[a][c] == _LCM_TABLE[b][c]

def _can_add(x, S):
    """Check if x can be added to S without creating a bad triple."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        la_x = _LCM_TABLE[x][a]
        for j in range(i + 1, n):
            b = S_list[j]
            # Check if (x, a, b) is a bad triple
            if _LCM_TABLE[x][b] == la_x and _LCM_TABLE[a][b] == la_x:
                return False
    return True

def _greedy(order):
    S = set()
    for x in order:
        if x in S:
            continue
        if _can_add(x, S):
            S.add(x)
    return S

def _local_search_add_only(S):
    """Try to add any remaining elements to S."""
    improved = True
    while improved:
        improved = False
        for x in range(1, 129):
            if x not in S and _can_add(x, S):
                S.add(x)
                improved = True
    return S

def _num_divisors(n):
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def _max_prime_factor(n):
    if n < 2:
        return 1
    largest = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            largest = d
            n //= d
        d += 1
    if n > 1:
        largest = n
    return largest

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    all_elements = list(range(1, 129))
    best_set = set()
    best_size = 0
    
    # Precomputed sorting keys
    div_counts = {x: _num_divisors(x) for x in all_elements}
    max_pf = {x: _max_prime_factor(x) for x in all_elements}
    primes = [x for x in all_elements if _is_prime(x)]
    non_primes = [x for x in all_elements if not _is_prime(x)]
    
    # 1. Deterministic Heuristics
    orderings = []
    
    # Primes first, then rest by divisors
    orderings.append(primes + sorted(non_primes, key=lambda x: div_counts[x]))
    
    # All by divisors
    orderings.append(sorted(all_elements, key=lambda x: div_counts[x]))
    
    # All by max prime factor (descending)
    orderings.append(sorted(all_elements, key=lambda x: max_pf[x], reverse=True))
    
    # Reverse natural order
    orderings.append(list(reversed(all_elements)))
    
    # Natural order
    orderings.append(all_elements)
    
    # Primes first, rest by max prime factor
    orderings.append(primes + sorted(non_primes, key=lambda x: max_pf[x], reverse=True))
    
    # Evaluate deterministic orderings
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy(order)
        S = _local_search_add_only(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # 2. Random Restarts
    random.seed(42)
    
    # Run many random greedy passes
    # Estimate time per pass: ~1ms? 128^3/3 ~ 700k ops. Python is slower, maybe 5-10ms.
    # 80s / 0.01s = 8000 passes. Let's aim for a few thousand.
    
    attempts = 0
    while time.time() < deadline:
        # Mix of fully random and biased random
        if attempts % 5 == 0:
            # Biased: Primes first, random rest
            order = primes + random.sample(non_primes, len(non_primes))
        else:
            order = random.sample(all_elements, 128)
            
        S = _greedy(order)
        S = _local_search_add_only(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        attempts += 1
        
        # Check time more frequently if we are doing many small steps
        if attempts % 10 == 0 and time.time() > deadline:
            break

    # Final verification (optional but good for safety)
    # The _can_add function ensures validity during construction.
    # However, if we want to be 100% sure, we can verify.
    # Verification is O(N^3) ~ 2M ops, very fast.
    S_list = sorted(best_set)
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            # This shouldn't happen if _can_add is correct, but if it does, remove c
            best_set.discard(c)
            break # Just one fix, or loop properly if needed. 
            # Given _can_add is strict, this block is likely dead code.
            
    if not best_set:
        best_set = {1}
        
    return list(best_set)
