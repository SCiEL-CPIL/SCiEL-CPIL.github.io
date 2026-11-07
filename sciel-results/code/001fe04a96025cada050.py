import math
import time
import sys
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    N = 256
    
    # Strategy: Use the structure of the problem
    # Two numbers are coprime iff they don't share any prime factor.
    # To maximize the sum, we should pick the largest possible number for each prime factor.
    # The largest number <= N that is a power of prime p is p^k where p^k <= N.
    # Any other number with prime factor p is smaller than p^k and conflicts with it.
    # So the optimal strategy is:
    # 1. Include 1 (it's coprime to everything)
    # 2. For each prime p <= N, include the largest power of p that is <= N.
    # 3. These powers are pairwise coprime because each has a distinct prime factor.
    
    # Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    # For each prime, find the largest power <= N
    result = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # This shouldn't happen, but if it does, fall back to a safer approach
                return _fallback_construct(N, deadline - time.time())
    
    # Ensure types are Python int
    result = [int(r) for r in result]
    return result

def _fallback_construct(N, time_budget):
    """
    Fallback: return the largest prime power for each prime <= N, plus 1.
    This is the optimal solution by the structural argument.
    """
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    result = [1]
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        result.append(power)
    
    return [int(r) for r in result]

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    res = construct()
    print(f"Sum: {sum(res)}, Count: {len(res)}")
