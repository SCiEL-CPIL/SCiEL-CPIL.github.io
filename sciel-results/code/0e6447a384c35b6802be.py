"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt, gcd
import time

def construct():
    N = 256
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, find the largest prime power p^k <= N
    result = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + primes
    
    best_sum = sum(result)
    best_result = [int(x) for x in result]
    
    # Try to improve by local search: replace elements with larger coprime candidates
    # Build conflict graph: two numbers conflict if gcd > 1
    # Use branch-and-bound style local search within time budget
    
    # Precompute gcd relationships for all pairs 1..256
    # For each number, compute its set of prime factors
    prime_factors = [set() for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            prime_factors[multiple].add(p)
    
    # Check if two numbers are coprime using their prime factor sets
    def are_coprime(a, b):
        return len(prime_factors[a] & prime_factors[b]) == 0
    
    # Local search: try to add or swap elements to increase sum
    current_set = set(best_result)
    current_sum = best_sum
    
    # Try adding numbers not in the set
    for candidate in range(N, 0, -1):
        if time.time() > deadline:
            break
        if candidate in current_set:
            continue
        # Check if candidate is coprime to all in current_set
        if all(are_coprime(candidate, x) for x in current_set):
            current_set.add(candidate)
            current_sum += candidate
    
    if current_sum > best_sum:
        best_sum = current_sum
        best_result = [int(x) for x in current_set]
    
    # Try swapping: for each element in the set, try to replace with a larger coprime number
    for _ in range(10):  # Multiple passes
        if time.time() > deadline:
            break
        improved = False
        for elem in list(current_set):
            # Try to find a larger number coprime to all others
            others = current_set - {elem}
            for candidate in range(N, elem + 1, -1):
                if candidate in current_set:
                    continue
                if all(are_coprime(candidate, x) for x in others):
                    current_set.remove(elem)
                    current_set.add(candidate)
                    current_sum += (candidate - elem)
                    improved = True
                    break
        if not improved:
            break
    
    if current_sum > best_sum:
        best_sum = current_sum
        best_result = [int(x) for x in current_set]
    
    # Ensure all are Python ints
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
