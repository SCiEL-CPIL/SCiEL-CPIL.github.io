import time
import random
from math import gcd
from itertools import combinations

# Precompute LCM table for all pairs in [1, 128]
_LCM_TABLE = [[0] * 129 for _ in range(129)]
for _a in range(1, 129):
    for _b in range(_a, 129):
        _val = _a * _b // gcd(_a, _b)
        _LCM_TABLE[_a][_b] = _val
        _LCM_TABLE[_b][_a] = _val

def _lcm(a, b):
    return _LCM_TABLE[a][b]

def _is_bad_triple(a, b, c):
    return _LCM_TABLE[a][b] == _LCM_TABLE[a][c] == _LCM_TABLE[b][c]

# Precompute all bad triples
_BAD_TRIPLES = set()
for _a in range(1, 129):
    for _b in range(_a + 1, 129):
        for _c in range(_b + 1, 129):
            if _is_bad_triple(_a, _b, _c):
                _BAD_TRIPLES.add((_a, _b, _c))

# For each element, precompute which pairs conflict with it
# conflicts_with[x] = set of (a, b) such that (x, a, b) is a bad triple
_CONFLICTS_WITH = [set() for _ in range(129)]
for (_a, _b, _c) in _BAD_TRIPLES:
    _CONFLICTS_WITH[_a].add((_b, _c))
    _CONFLICTS_WITH[_b].add((_a, _c))
    _CONFLICTS_WITH[_c].add((_a, _b))

def _conflict_pairs_with_x(x, S):
    """Check if adding x to S creates any bad triple."""
    # Check if any pair in S is in conflicts_with[x]
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if (a, b) in _CONFLICTS_WITH[x] or (b, a) in _CONFLICTS_WITH[x]:
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

def _local_search(S, max_iters=5000, deadline=None):
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

def _multi_start_local_search(S, deadline):
    """Run local search multiple times with different perturbations."""
    best = set(S)
    best_size = len(best)
    
    # Try multiple restarts with random perturbations
    for _ in range(20):
        if deadline and time.time() > deadline:
            break
        
        # Perturb: remove 1-3 random elements
        perturbed = set(best)
        to_remove = min(random.randint(1, 3), len(perturbed))
        for _ in range(to_remove):
            if perturbed:
                perturbed.discard(random.choice(list(perturbed)))
        
        # Run local search on perturbed set
        improved = _local_search(perturbed, max_iters=2000, deadline=deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit
    
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
    
    # 4. By maximum prime factor (larger first)
    def max_prime_factor(n):
        if n < 2:
            return 1
        mpf = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                mpf = d
                n //= d
            d += 1
        if n > 1:
            mpf = max(mpf, n)
        return mpf
    
    orderings.append(sorted(all_elements, key=lambda x: -max_prime_factor(x)))
    
    # 5. Random orderings
    random.seed(42)
    for _ in range(8):
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
    
    # Multi-start local search
    if time.time() < deadline:
        improved = _multi_start_local_search(best_set, deadline)
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
