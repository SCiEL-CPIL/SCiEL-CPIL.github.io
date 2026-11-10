"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill

    MAX_N = 256

    # Generate all primes up to MAX_N using Sieve of Eratosthenes
    def get_primes(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n + 1) if sieve[i]]

    primes = get_primes(MAX_N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)

    # For each number 2..256, find its distinct prime factors
    # and build a constraint matrix A (n_primes x 255) where A[i,j] = 1 if primes[i] divides (j+2)
    numbers = list(range(2, MAX_N + 1))  # 255 numbers
    n_numbers = len(numbers)
    
    # Build constraint matrix: for each prime, which numbers it divides
    A = np.zeros((n_primes, n_numbers), dtype=np.int8)
    for j, num in enumerate(numbers):
        n = num
        for p in primes:
            if p > n:
                break
            if n % p == 0:
                A[prime_to_idx[p], j] = 1
                while n % p == 0:
                    n //= p
    
    # Objective: maximize sum of selected numbers
    c = np.array([-num for num in numbers], dtype=np.float64)  # minimize -sum
    
    # Try to use scipy's milp for exact ILP solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Constraints: A @ x <= 1 (each prime used at most once), x >= 0
        cons = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n_numbers)  # all variables integer
        
        result = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            selected = [numbers[i] for i in range(n_numbers) if result.x[i] > 0.5]
            # Always include 1
            selected = [1] + selected
            
            # Verify pairwise coprimality
            valid = True
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if gcd(selected[i], selected[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and time.time() < deadline:
                return [int(x) for x in selected]
    except Exception:
        pass
    
    # Fallback: greedy approach - for each prime, take the largest power <= 256
    result = [1]
    for p in primes:
        power = p
        while power * p <= MAX_N:
            power *= p
        result.append(power)
    
    # Ensure all are Python ints
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
