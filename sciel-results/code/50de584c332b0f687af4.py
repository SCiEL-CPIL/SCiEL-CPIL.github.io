"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd, isqrt

DEADLINE = 82.0  # safety margin before 90s kill

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

def _fast_check_add(S, x):
    """Check if adding x to S creates any violation. S is a set."""
    for a in S:
        # a | x * c  for some c in S?
        g = gcd(a, x)
        rem = a // g
        if rem == 1:
            if len(S) >= 2:
                return False
        else:
            for c in S:
                if c != a and c % rem == 0:
                    return False
    for a in S:
        for b in S:
            if a == b:
                continue
            if (a * b) % x == 0:
                return False
    return True

def _is_valid_set(S):
    """Check all triples of distinct elements."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _violation_count(S):
    """Count number of violating triples."""
    S_list = list(S)
    n = len(S_list)
    count = 0
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    count += 1
    return count

def construct():
    start_time = time.time()
    
    primes = [p for p in range(2, 257) if _is_prime(p)]
    composites = [x for x in range(2, 257) if not _is_prime(x)]
    all_nums = list(range(2, 257))
    
    best_set = set(primes)
    best_size = len(primes)
    
    def try_greedy(candidate_order):
        S = set()
        for x in candidate_order:
            if _fast_check_add(S, x):
                S.add(x)
        return S
    
    # Multiple greedy strategies
    orderings = [
        primes + composites,
        composites + primes,
        sorted(all_nums, key=lambda x: -x),
        sorted(all_nums, key=lambda x: x),
        # Sort by number of prime factors (fewer = safer)
        sorted(all_nums, key=lambda x: sum(1 for d in range(2, isqrt(x)+1) if x % d == 0)),
        # Sort by largest prime factor (larger = safer)
        sorted(all_nums, key=lambda x: -max((d for d in range(2, x+1) if x % d == 0), default=x)),
    ]
    
    for ordering in orderings:
        S = try_greedy(ordering)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        if time.time() - start_time > DEADLINE - 15:
            break
    
    # Local search: add composites
    S = set(best_set)
    for x in composites:
        if x not in S and _fast_check_add(S, x):
            S.add(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # 1-opt: remove one, try adding others
    if time.time() - start_time < DEADLINE - 10:
        S = set(best_set)
        improved = True
        while improved:
            improved = False
            for a in list(S):
                for x in composites:
                    if x in S:
                        continue
                    S2 = S - {a}
                    if _fast_check_add(S2, x):
                        S = S2 | {x}
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = S
                        break
                if time.time() - start_time > DEADLINE - 10:
                    break
            if time.time() - start_time > DEADLINE - 10:
                break
    
    # 2-opt: remove two, add two
    if time.time() - start_time < DEADLINE - 8:
        S = set(best_set)
        for a in list(S):
            for b in list(S):
                if a == b:
                    continue
                S_ab = S - {a, b}
                for x in composites:
                    if x in S_ab:
                        continue
                    if _fast_check_add(S_ab, x):
                        S_x = S_ab | {x}
                        for y in composites:
                            if y in S_x or y == x:
                                continue
                            if _fast_check_add(S_x, y):
                                S_xy = S_x | {y}
                                if len(S_xy) > len(S):
                                    S = S_xy
                                    if len(S) > best_size:
                                        best_size = len(S)
                                        best_set = S
                                break
                        if time.time() - start_time > DEADLINE - 8:
                            break
                if time.time() - start_time > DEADLINE - 8:
                    break
            if time.time() - start_time > DEADLINE - 8:
                break
    
    # Random restarts
    if time.time() - start_time < DEADLINE - 12:
        rng = random.Random(42)
        for trial in range(500):
            if time.time() - start_time > DEADLINE - 12:
                break
            start_primes = [p for p in primes if rng.random() < 0.85]
            start_comps = [c for c in composites if rng.random() < 0.15]
            remaining = [x for x in all_nums if x not in start_primes and x not in start_comps]
            ordering = start_primes + start_comps + remaining
            rng.shuffle(ordering)
            S = try_greedy(ordering)
            # Quick local improvement
            for x in composites:
                if x not in S and _fast_check_add(S, x):
                    S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Final verification
    if not _is_valid_set(best_set):
        best_set = set(primes)
    
    result = list(best_set)
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
