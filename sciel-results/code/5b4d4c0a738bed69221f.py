"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    def valid_triple(a, b, c):
        """Check that a does not divide b*c (a,b,c distinct)."""
        return (b * c) % a != 0
    
    def can_add(x, S):
        """Check if x can be added to set S without violating constraints."""
        S_list = list(S)
        n = len(S_list)
        # For every pair (b,c) in S, b!=c: x must not divide b*c
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (S_list[i] * S_list[j]) % x == 0:
                    return False
        # For every pair (b,c) in S, b!=c: b must not divide x*c
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (x * S_list[j]) % S_list[i] == 0:
                    return False
        return True
    
    def greedy_order(candidates, order_key=None):
        """Greedy: add elements in given order if valid."""
        S = set()
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S
    
    def greedy_with_order(candidates, key_func):
        """Greedy with a specific ordering."""
        S = set()
        for x in sorted(candidates, key=key_func):
            if can_add(x, S):
                S.add(x)
        return S
    
    base_candidates = list(range(2, 257))
    
    # Strategy 1: All primes (baseline)
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for d in range(3, isqrt(n) + 1, 2):
            if n % d == 0:
                return False
        return True
    
    primes = [p for p in base_candidates if is_prime(p)]
    best_set = set(primes)
    best_size = len(best_set)
    
    # Strategy 2: Greedy with various orderings
    orderings = [
        lambda x: x,                          # ascending
        lambda x: -x,                         # descending
        lambda x: x.bit_count(),              # popcount
        lambda x: -x.bit_count(),             # reverse popcount
        lambda x: (x % 7, x),                 # mod 7 grouping
        lambda x: (-x % 7, x),                # reverse mod 7
    ]
    
    for key_func in orderings:
        if time.time() > deadline:
            break
        S = greedy_with_order(base_candidates, key_func)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 3: Randomized greedy with multiple restarts
    rng = random.Random(42)
    best_random_size = 0
    best_random_set = set()
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        # Random shuffle with some bias
        candidates = base_candidates[:]
        rng.shuffle(candidates)
        S = set()
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        if len(S) > best_random_size:
            best_random_size = len(S)
            best_random_set = S
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Strategy 4: Start from primes, try to add composites
    if time.time() < deadline:
        S = set(primes)
        composites = [x for x in base_candidates if x not in S]
        # Try adding composites in various orders
        for key_func in [lambda x: x, lambda x: -x, lambda x: x.bit_count()]:
            if time.time() > deadline:
                break
            S_copy = set(S)
            for x in sorted(composites, key=key_func):
                if can_add(x, S_copy):
                    S_copy.add(x)
            if len(S_copy) > best_size:
                best_size = len(S_copy)
                best_set = S_copy
    
    # Strategy 5: Randomized local search - start from best, try swaps
    if time.time() < deadline and best_size < 100:
        S = set(best_set)
        excluded = [x for x in base_candidates if x not in S]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(excluded)
            for x in excluded:
                if time.time() > deadline:
                    break
                # Try adding x and removing one element to make room
                for y in list(S):
                    S_trial = S - {y} | {x}
                    if len(S_trial) != len(S):
                        continue
                    # Check validity
                    valid = True
                    S_list = list(S_trial)
                    for i in range(len(S_list)):
                        for j in range(len(S_list)):
                            if i == j:
                                continue
                            for k in range(len(S_list)):
                                if i == k or j == k:
                                    continue
                                a, b, c = S_list[i], S_list[j], S_list[k]
                                if (b * c) % a == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid and len(S_trial) > len(S):
                        S = S_trial
                        improved = True
                        break
                if improved:
                    break
    
    # Ensure we return a valid nonempty set
    if not best_set:
        best_set = {2}
    
    result = list(best_set)
    # Ensure all are Python int
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
