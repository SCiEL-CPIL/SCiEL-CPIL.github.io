import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def _conflict_pairs_with_x(x, S):
    """Check if adding x to S creates any bad triple."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if _is_bad_triple(x, a, b):
                return True
    return False

def _greedy_with_order(order, S):
    """Add elements from order to S if they don't conflict."""
    for x in order:
        if x in S:
            continue
        if not _conflict_pairs_with_x(x, S):
            S.add(x)
    return S

def _local_search(S, max_iters=2000, deadline=None):
    """Try to improve S by swapping or adding elements."""
    S = set(S)
    best_size = len(S)
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try adding any element not in S
        for x in range(1, 129):
            if x in S:
                continue
            if not _conflict_pairs_with_x(x, S):
                S.add(x)
                best_size = max(best_size, len(S))
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add another
        for a in list(S):
            S.remove(a)
            for x in range(1, 129):
                if x in S:
                    continue
                if not _conflict_pairs_with_x(x, S):
                    S.add(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        improved = True
                    break
            if not improved:
                S.add(a)
        if not improved:
            break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    all_elements = list(range(1, 129))
    
    # Multiple orderings
    orderings = []
    
    # 1. Primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in all_elements if is_prime(x)]
    non_primes = [x for x in all_elements if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # 2. By number of divisors (fewer first)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    orderings.append(sorted(all_elements, key=lambda x: num_divisors(x)))
    
    # 3. Reverse order
    orderings.append(list(reversed(all_elements)))
    
    # 4. Random orderings
    random.seed(42)
    for _ in range(5):
        orderings.append(random.sample(all_elements, 128))
    
    best_set = set()
    best_size = 0
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_with_order(order, set())
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search on best
    if time.time() < deadline:
        improved = _local_search(best_set, max_iters=5000, deadline=deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Final check: ensure no bad triples
    S_list = sorted(best_set)
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            # Remove one to fix
            best_set.discard(c)
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)
