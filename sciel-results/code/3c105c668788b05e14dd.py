import math
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    
    # 1. Find all primes up to N
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Determine max prime power for each prime
    max_prime_power = {}
    for p in primes:
        pwr = 1
        while pwr * p <= N:
            pwr *= p
        max_prime_power[p] = pwr

    # 3. Identify candidate integers
    # We consider all integers from 2 to N.
    # Each integer n has a set of prime factors.
    # We want to select a subset of these integers such that no prime factor is shared.
    
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Map each candidate to its prime factors (indices)
    candidate_factors = []
    for n in candidates:
        m = n
        factors = set()
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                factors.add(prime_to_idx[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            factors.add(prime_to_idx[m])
        candidate_factors.append(factors)
        
    # Weights for each candidate
    weights = np.array(candidates, dtype=float)
    
    # 4. Formulate ILP
    # Variables: x_i for i in range(num_candidates)
    # Objective: Maximize sum(weights * x)
    # Constraints: For each prime j, sum(x_i for i where j in candidate_factors[i]) <= 1
    
    # A matrix A of shape (num_primes, num_candidates)
    A = np.zeros((num_primes, num_candidates), dtype=float)
    for i, factors in enumerate(candidate_factors):
        for j in factors:
            A[j, i] = 1.0
            
    # Bounds: 0 <= x_i <= 1
    integrality = np.ones(num_candidates)
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: A @ x <= 1
    constraints = LinearConstraint(A, lb=None, ub=1)
    
    # Objective: Minimize -sum(weights * x)
    c = -weights
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status != 0:
            # Fallback if solver fails
            # Greedy heuristic: pick largest prime powers
            result_set = {1}
            for p in primes:
                result_set.add(max_prime_power[p])
            return list(result_set)
            
        # Extract solution
        selected_indices = [i for i in range(num_candidates) if res.x[i] > 0.5]
        
        result_set = {1}
        for i in selected_indices:
            result_set.add(candidates[i])
            
        # Verify
        result_list = list(result_set)
        for i in range(len(result_list)):
            for j in range(i + 1, len(result_list)):
                if math.gcd(result_list[i], result_list[i+1] if False else result_list[j]) != 1:
                    # Fallback
                    result_set = {1}
                    for p in primes:
                        result_set.add(max_prime_power[p])
                    return list(result_set)
                    
        return result_list
        
    except Exception:
        # Fallback
        result_set = {1}
        for p in primes:
            result_set.add(max_prime_power[p])
        return list(result_set)
