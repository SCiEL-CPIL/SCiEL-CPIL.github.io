"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute prime factors for each number
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
                
    # Get all primes <= N
    primes = [i for i in range(2, N + 1) if prime_factors[i] == {i}]
    prime_index = {p: idx for idx, p in enumerate(primes)}
    
    # Objective coefficients
    c = np.array(numbers, dtype=float)
    
    # Constraints: For each prime p, sum of x_i for i divisible by p <= 1
    # We build the constraint matrix A_ub such that A_ub @ x <= b_ub
    # Number of constraints = number of primes
    # Number of variables = N
    
    A_ub = np.zeros((len(primes), N))
    for i, num in enumerate(numbers):
        for p in prime_factors[num]:
            A_ub[prime_index[p], i] = 1
            
    b_ub = np.ones(len(primes))
    
    # Bounds for x_i: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer
    integrality = np.ones(N)
    
    # Define the MILP problem
    # Minimize -c^T x  (since milp minimizes)
    res = milp(
        c=-c,
        constraints=LinearConstraint(A_ub, ub=b_ub),
        integrality=integrality,
        bounds=bounds
    )
    
    # Fallback logic using the "swap" heuristic from inspiration if MILP fails or times out
    def get_fallback_solution():
        is_prime = [True] * (N + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, math.isqrt(N) + 1):
            if is_prime[i]:
                for j in range(i * i, N + 1, i):
                    is_prime[j] = False
        
        primes_fallback = [i for i in range(2, N + 1) if is_prime[i]]
        
        prime_power_map = {}
        base_sum = 1
        selected = {1: True}
        
        for p in primes_fallback:
            power = p
            while power * p <= N:
                power *= p
            prime_power_map[p] = power
            base_sum += power
            selected[power] = True
            
        best_sum = base_sum
        best_set = list(selected.keys())
        
        candidates = []
        for c_val in range(4, N + 1):
            if not is_prime[c_val] and c_val > 1:
                div_primes = [p for p in primes_fallback if c_val % p == 0]
                base_val = sum(prime_power_map[p] for p in div_primes)
                gain = c_val - base_val
                if gain > 0:
                    candidates.append((c_val, div_primes, gain))
                    
        # Sort by gain descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        num_candidates = len(candidates)
        
        # Simple greedy selection of non-conflicting candidates
        # Since candidates are sorted by gain, we just pick if no prime conflict
        selected_primes = set()
        chosen_candidates = []
        
        for c_val, div_primes, gain in candidates:
            if not selected_primes.intersection(div_primes):
                chosen_candidates.append((c_val, div_primes, gain))
                selected_primes.update(div_primes)
                
        # Construct final set from fallback
        final_set_dict = {k: True for k in selected}
        for c_val, div_primes, gain in chosen_candidates:
            for p in div_primes:
                lp = prime_power_map[p]
                if lp in final_set_dict:
                    del final_set_dict[lp]
            final_set_dict[c_val] = True
            
        return [int(x) for x in final_set_dict.keys()]

    if not res.success:
        return get_fallback_solution()
    
    # Extract solution
    x = res.x
    selected = [num for num, xi in zip(numbers, x) if xi > 0.5]
    
    # Ensure we return a list of Python ints
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
