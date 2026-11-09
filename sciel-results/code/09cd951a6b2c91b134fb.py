"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    
    N = 256
    # Build list of all numbers 1..256 with their prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Map each number to its set of prime factors
    pf = {1: set()}
    for i in range(2, N + 1):
        pf[i] = prime_factors(i)
    
    # Initial solution: all primes + 1
    primes = [i for i in range(2, N + 1) if len(pf[i]) == 1 and list(pf[i])[0] == i]
    # Actually primes are numbers where the only prime factor is itself
    primes = [i for i in range(2, N + 1) if pf[i] == {i}]
    selected = set([1] + primes)
    
    # Build conflict check: two numbers conflict if they share a prime factor
    def conflicts(a, b):
        if a == 1 or b == 1:
            return False
        return len(pf[a] & pf[b]) > 0
    
    def is_valid(selected_set):
        lst = list(selected_set)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if conflicts(lst[i], lst[j]):
                    return False
        return True
    
    def total_sum(selected_set):
        return sum(selected_set)
    
    best = set(selected)
    best_sum = total_sum(best)
    
    # Local search: try adding a number, or swapping one number for another
    all_numbers = list(range(1, N + 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unselected number
        for num in all_numbers:
            if num in best:
                continue
            new_set = best | {num}
            if is_valid(new_set):
                new_sum = total_sum(new_set)
                if new_sum > best_sum:
                    best = new_set
                    best_sum = new_sum
                    improved = True
        # Try swapping: remove one, add another
        for remove_num in list(best):
            if remove_num == 1:
                continue  # Keep 1 always
            base = best - {remove_num}
            for add_num in all_numbers:
                if add_num in base:
                    continue
                new_set = base | {add_num}
                if is_valid(new_set):
                    new_sum = total_sum(new_set)
                    if new_sum > best_sum:
                        best = new_set
                        best_sum = new_sum
                        improved = True
    
    # Convert to list of Python ints
    result = list(best)
    result.sort()
    # Ensure all are Python int
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
