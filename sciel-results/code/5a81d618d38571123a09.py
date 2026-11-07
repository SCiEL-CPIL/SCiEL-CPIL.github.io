"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    n = 256
    
    # Sieve of Eratosthenes for primes up to n
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    
    primes = [i for i in range(2, n + 1) if sieve[i]]
    num_primes = len(primes)
    
    # Precompute pairwise gcd matrix (1 = coprime, 0 = not coprime)
    # For all numbers 2..256
    numbers = list(range(2, n + 1))
    num_numbers = len(numbers)
    coprime = [[False] * num_numbers for _ in range(num_numbers)]
    for i in range(num_numbers):
        for j in range(i + 1, num_numbers):
            coprime[i][j] = coprime[j][i] = (gcd(numbers[i], numbers[j]) == 1)
    
    # Precompute prime factors for each number
    prime_factors = [[] for _ in range(n + 1)]
    for p in primes:
        for j in range(p, n + 1, p):
            prime_factors[j].append(p)
    
    # Greedy initial solution: sort by value descending, include if coprime with all selected
    selected = set()
    for num in sorted(numbers, reverse=True):
        if all(gcd(num, s) == 1 for s in selected):
            selected.add(num)
    
    best_sum = sum(selected) + 1  # +1 for the number 1
    best_solution = list(selected)
    
    # Local search: 1-for-1 and 1-for-2 swaps
    def evaluate_swap_remove_add(current_set, remove, add):
        """Check if removing 'remove' and adding 'add' is valid and returns gain."""
        if add in current_set:
            return 0
        # Check add is coprime with all in current_set except remove
        for x in current_set:
            if x != remove and gcd(add, x) != 1:
                return None
        return add - remove
    
    def evaluate_swap_remove_two_add_one(current_set, remove1, remove2, add):
        """Check if removing remove1, remove2 and adding 'add' is valid and returns gain."""
        if add in current_set:
            return 0
        for x in current_set:
            if x != remove1 and x != remove2 and gcd(add, x) != 1:
                return None
        return add - remove1 - remove2
    
    current_set = set(best_solution)
    current_sum = best_sum - 1  # exclude 1 from sum for calculations
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-for-1 swaps
        for rem in list(current_set):
            for add in numbers:
                if add in current_set:
                    continue
                gain = evaluate_swap_remove_add(current_set, rem, add)
                if gain is not None and gain > 0:
                    current_set.discard(rem)
                    current_set.add(add)
                    current_sum += gain
                    if current_sum + 1 > best_sum:
                        best_sum = current_sum + 1
                        best_solution = list(current_set)
                    improved = True
                    break
            if improved:
                break
        
        if time.time() >= deadline:
            break
        
        # Try 1-for-2 swaps (remove two, add one)
        for rem1 in list(current_set):
            for rem2 in list(current_set):
                if rem1 == rem2:
                    continue
                for add in numbers:
                    if add in current_set:
                        continue
                    gain = evaluate_swap_remove_two_add_one(current_set, rem1, rem2, add)
                    if gain is not None and gain > 0:
                        current_set.discard(rem1)
                        current_set.discard(rem2)
                        current_set.add(add)
                        current_sum += gain
                        if current_sum + 1 > best_sum:
                            best_sum = current_sum + 1
                            best_solution = list(current_set)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if time.time() >= deadline:
            break
    
    # Verify all are Python ints
    result = [1] + sorted(best_solution)
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
