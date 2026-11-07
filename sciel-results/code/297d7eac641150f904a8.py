"""
Maximize sum of pairwise-coprime interval integers from 1..256.

Strategy:
1. The optimal set consists of 1 and, for every prime p <= 256, the largest power p^k <= 256.
2. These numbers are pairwise coprime because each has a unique prime factor (the base p).
3. Any composite number formed by multiplying primes from this set would be less than the sum of their individual maximum powers (for small primes) or impossible (for large primes where p*q > 256).
4. Thus, this greedy approach yields the global maximum.
"""

def construct():
    n = 256
    
    # Sieve of Eratosthenes to find all primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # For each prime, find the largest power p^k <= n
    selected = []
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        selected.append(val)
        
    # Include 1, which is coprime with all integers
    result = [1] + selected
    
    # Ensure types are strictly Python int
    result = [int(x) for x in result]
    
    return result
