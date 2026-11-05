import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    numbers = list(range(1, n + 1))
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]
    
    # Check if adding x to set S creates a bad triple
    def can_add_fast(x, S):
        groups = {}
        for a in S:
            L = get_lcm(x, a)
            if L not in groups:
                groups[L] = []
            groups[L].append(a)
        
        for L, elems in groups.items():
            if len(elems) < 2:
                continue
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    if get_lcm(elems[i], elems[j]) == L:
                        return False
        return True
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p
    
    core_set = prime_powers | {1}
    
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    remaining = [x for x in range(1, n + 1) if x not in core_set]
    
    # Multiple randomized greedy passes
    best_set = set(core_set)
    
    orderings = []
    orderings.append(sorted(remaining, key=lambda x: (num_distinct_primes(x), x)))
    orderings.append(sorted(remaining, key=lambda x: (num_distinct_primes(x), -x)))
    orderings.append(sorted(remaining, key=lambda x: x))
    orderings.append(sorted(remaining, key=lambda x: -x))
    
    for _ in range(5):
        rand_order = remaining[:]
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    for ordering in orderings:
        if time.time() > deadline - 15.0:
            break
        
        current = set(core_set)
        for x in ordering:
            if time.time() > deadline - 15.0:
                break
            if x in current:
                continue
            if can_add_fast(x, current):
                current.add(x)
        
        if len(current) > len(best_set):
            best_set = set(current)
    
    # Local search: try swaps
    all_numbers = set(range(1, n + 1))
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        rejected = [x for x in all_numbers if x not in best_set]
        rejected.sort(key=lambda x: (num_distinct_primes(x), x))
        
        for e in list(best_set):
            if time.time() > deadline - 5.0:
                break
            
            best_set.remove(e)
            
            added = []
            for r in rejected:
                if r in best_set:
                    continue
                if can_add_fast(r, best_set):
                    best_set.add(r)
                    added.append(r)
            
            if len(added) > 1:
                improved = True
                rejected = [x for x in rejected if x not in best_set]
            else:
                best_set.add(e)
                for r in added:
                    best_set.remove(r)
    
    # Final validation
    S = sorted(best_set)
    valid = True
    for i in range(len(S)):
        if time.time() > deadline - 1.0:
            break
        for j in range(i + 1, len(S)):
            lab = get_lcm(S[i], S[j])
            for k in range(j + 1, len(S)):
                if get_lcm(S[i], S[k]) == lab and get_lcm(S[j], S[k]) == lab:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return list(core_set)
    
    return list(best_set)
