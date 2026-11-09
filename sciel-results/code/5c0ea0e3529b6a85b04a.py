# EVOLVE-BLOCK-START
import time
from math import gcd
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    candidates = list(range(1, N + 1))
    n_cand = N
    
    # Identify primes <= 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(N)
    n_primes = len(primes)
    
    # Build constraint matrix A
    # A[p, i] = 1 if prime p divides candidate[i]
    A = np.zeros((n_primes, n_cand), dtype=int)
    for i, val in enumerate(candidates):
        for p_idx, p in enumerate(primes):
            if val % p == 0:
                A[p_idx, i] = 1
    
    # Objective: maximize sum(candidates[i] * x[i])
    # milp minimizes, so use negative costs
    c = -np.array(candidates, dtype=float)
    
    # Constraints: A x <= 1
    # x >= 0, x <= 1
    # Integer: x in {0, 1}
    
    constraints = LinearConstraint(A, ub=1)
    integrality = np.ones(n_cand)
    bounds = Bounds(0, 1)
    
    best_sum = 0
    best_result = [candidates[0]] # Fallback: at least 1
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x = res.x
            selected = [candidates[i] for i in range(n_cand) if x[i] > 0.5]
            if selected:
                best_result = selected
                best_sum = sum(selected)
        else:
            # Fallback to greedy if solver fails
            pass
            
    except Exception:
        pass
    
    if time.time() > deadline - 5 or best_sum == 0:
        # Fallback: Simple Greedy
        # This ensures we always return a valid list
        selected = set()
        blocked = 0
        # Sort by value descending
        order = sorted(range(n_cand), key=lambda x: candidates[x], reverse=True)
        # Precompute conflicts for fallback speed
        conflicts = [0] * n_cand
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                if gcd(candidates[i], candidates[j]) > 1:
                    conflicts[i] |= (1 << j)
                    conflicts[j] |= (1 << i)
        
        for idx in order:
            if not (blocked >> idx) & 1:
                selected.add(idx)
                blocked |= (1 << idx)
                blocked |= conflicts[idx]
        
        best_result = [candidates[i] for i in selected]
        best_sum = sum(best_result)

    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
