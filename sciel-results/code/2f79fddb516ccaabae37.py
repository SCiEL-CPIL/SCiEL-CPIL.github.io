import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _count_distinct_prime_factors(n):
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _is_prime_power(n):
    if n <= 1:
        return True
    return _count_distinct_prime_factors(n) == 1

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # 1. Identify prime powers and 1
    prime_powers = [i for i in range(1, 129) if _is_prime_power(i)]
    others = [i for i in range(1, 129) if not _is_prime_power(i)]
    
    best_set = set(prime_powers)
    best_size = len(best_set)
    
    # Helper to check if x conflicts with set S
    def is_conflict(x, S):
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                L = _lcm(a, b)
                if _lcm(a, x) == L and _lcm(b, x) == L:
                    return True
        return False

    # Try different orderings for 'others'
    orderings = []
    
    # Order 1: Ascending value
    orderings.append(sorted(others))
    
    # Order 2: Descending value
    orderings.append(sorted(others, reverse=True))
    
    # Order 3: Ascending number of prime factors
    orderings.append(sorted(others, key=lambda x: _count_distinct_prime_factors(x)))
    
    # Order 4: Descending number of prime factors
    orderings.append(sorted(others, key=lambda x: _count_distinct_prime_factors(x), reverse=True))
    
    # Order 5: Random (fixed seed for reproducibility)
    random.seed(42)
    shuffled = others[:]
    random.shuffle(shuffled)
    orderings.append(shuffled)
    
    # Order 6: Random (different seed)
    random.seed(123)
    shuffled = others[:]
    random.shuffle(shuffled)
    orderings.append(shuffled)

    for order in orderings:
        if time.time() > deadline:
            break
            
        current_set = set(prime_powers)
        
        for x in order:
            if time.time() > deadline:
                break
            if not is_conflict(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    return list(best_set)
