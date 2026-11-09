"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

# Precompute LCM table for all pairs in [1, 128]
_LCM_TABLE = {}
def _lcm(a, b):
    key = (min(a,b), max(a,b))
    if key not in _LCM_TABLE:
        _LCM_TABLE[key] = a * b // gcd(a, b)
    return _LCM_TABLE[key]

def _is_bad_triple(a, b, c):
    """Check if {a, b, c} forms a bad triple (equal pairwise LCMs)."""
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _conflicts_with_set(x, S):
    """Check if adding x to S creates any bad triple. Uses precomputed LCM."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        la = _lcm(x, a)
        for j in range(i + 1, n):
            b = S_list[j]
            if la == _lcm(x, b) == _lcm(a, b):
                return True
    return False

def _greedy_with_order(order):
    """Add elements from order to a new set if they don't conflict."""
    S = set()
    for x in order:
        if not _conflicts_with_set(x, S):
            S.add(x)
    return S

def _local_search(S, deadline):
    """
    Try to improve S by:
    1. Adding any element that doesn't conflict
    2. 1-swaps: remove one, add one
    3. 2-swaps: remove two, add two
    Returns the improved set.
    """
    S = set(S)
    full_set = set(range(1, 129))
    S_list = sorted(S)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element not in S
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflicts_with_set(x, S):
                S.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swaps (remove one, add one)
        found_swap = False
        for a in list(S):
            S.remove(a)
            for x in range(1, 129):
                if x in S:
                    continue
                if not _conflicts_with_set(x, S):
                    S.add(x)
                    found_swap = True
                    break
            if not found_swap:
                S.add(a)
        
        if found_swap:
            continue
        
        # Phase 3: Try 2-swaps (remove two, add two)
        S_list = sorted(S)
        n = len(S_list)
        found_2swap = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S_list[i], S_list[j]
                S.remove(a)
                S.remove(b)
                for x in range(1, 129):
                    if x in S:
                        continue
                    if _conflicts_with_set(x, S):
                        continue
                    S.add(x)
                    for y in range(1, 129):
                        if y in S or y == x:
                            continue
                        if not _conflicts_with_set(y, S):
                            S.add(y)
                            found_2swap = True
                            break
                    if not found_2swap:
                        S.remove(x)
                        S.add(a)
                        S.add(b)
                        break
                if found_2swap:
                    break
            if found_2swap:
                break
        
        if not found_2swap:
            break
    
    return S

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _num_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count

def _max_prime_factor(n):
    """Return the largest prime factor of n."""
    if n < 2:
        return 1
    max_pf = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            max_pf = d
            n //= d
        d += 1
    if n > 1:
        max_pf = n
    return max_pf

def _radical(n):
    """Return the radical of n (product of distinct prime factors)."""
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            result *= d
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        result *= n
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    all_elements = list(range(1, 129))
    
    # Precompute properties for all elements
    primes = [x for x in all_elements if _is_prime(x)]
    non_primes = [x for x in all_elements if not _is_prime(x)]
    div_counts = {x: _num_divisors(x) for x in all_elements}
    max_pfs = {x: _max_prime_factor(x) for x in all_elements}
    radicals = {x: _radical(x) for x in all_elements}
    
    # Build several deterministic orderings
    orderings = []
    
    # 1. Primes first, then non-primes
    orderings.append(primes + non_primes)
    
    # 2. By number of divisors (fewer first)
    orderings.append(sorted(all_elements, key=lambda x: div_counts[x]))
    
    # 3. Reverse order (largest first)
    orderings.append(list(reversed(all_elements)))
    
    # 4. Forward order (smallest first) - baseline greedy
    orderings.append(all_elements)
    
    # 5. By max prime factor (larger first)
    orderings.append(sorted(all_elements, key=lambda x: max_pfs[x], reverse=True))
    
    # 6. By radical (larger first)
    orderings.append(sorted(all_elements, key=lambda x: radicals[x], reverse=True))
    
    # 7. By radical (smaller first)
    orderings.append(sorted(all_elements, key=lambda x: radicals[x]))
    
    # 8. Interleave primes and composites by value
    interleaved = []
    p_idx, c_idx = 0, 0
    while p_idx < len(primes) or c_idx < len(non_primes):
        if p_idx < len(primes):
            interleaved.append(primes[p_idx])
            p_idx += 1
        if c_idx < len(non_primes):
            interleaved.append(non_primes[c_idx])
            c_idx += 1
    orderings.append(interleaved)
    
    # Random orderings with different seeds
    random.seed(42)
    for _ in range(15):
        orderings.append(random.sample(all_elements, 128))
    
    best_set = set()
    best_size = 0
    
    # Run all deterministic and random greedy passes
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_with_order(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Continue with more random restarts if time permits
    while time.time() < deadline:
        candidates = random.sample(all_elements, 128)
        S = _greedy_with_order(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search on the best set found
    if time.time() < deadline:
        improved = _local_search(best_set, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Final validation: ensure no bad triples exist
    S_list = sorted(best_set)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            valid = False
            break
    
    if not valid:
        # Fallback: use the simplest valid set
        best_set = {1}
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
