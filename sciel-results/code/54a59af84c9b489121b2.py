import time
from math import isqrt

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    # Generate primes up to a reasonable limit using sieve
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    # Generate primes up to 10000 to cover all possibilities
    primes_list = sieve(10000)
    
    def generate_smooth_numbers(limit, primes):
        """
        Generates all numbers <= limit that are composed of the given primes.
        Returns a set of such numbers.
        """
        smooths = set([1])
        frontier = [1]
        while frontier:
            x = frontier.pop()
            for p in primes:
                nx = x * p
                if nx > limit:
                    break
                if nx not in smooths:
                    smooths.add(nx)
                    frontier.append(nx)
        return smooths

    def is_smooth(n, primes):
        """Check if n is smooth with respect to given primes."""
        if n == 1:
            return True
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                n //= p
        return n == 1

    best_pair = None
    best_value = float('inf')
    
    current_primes = []
    for B in primes_list:
        if time.time() > deadline - 5:
            break
            
        current_primes.append(B)
        
        # Generate smooth numbers for these primes
        smooths = generate_smooth_numbers(TARGET, current_primes)
        
        # Check if any a in smooths has b = TARGET - a in smooths
        found = False
        for a in smooths:
            b = TARGET - a
            if b <= 0:
                continue
            if b in smooths:
                if a > 0 and b > 0:
                    best_pair = (a, b)
                    best_value = B
                    found = True
                    break
        if found:
            break
            
    if best_pair:
        return [best_pair[0], best_pair[1]]
    else:
        # Fallback: try a more direct approach
        # For each possible max prime factor B, check if there exists a decomposition
        # where both a and b have all prime factors <= B
        # Use a different strategy: iterate over possible B and check
        # If still no solution, return the best we can do
        
        # Try to find a pair by checking larger B values more efficiently
        # For B > 97, we can check if there exists a in [1, TARGET-1]
        # such that both a and TARGET-a have all prime factors <= B
        
        # This is expensive, so we'll just return a reasonable fallback
        # 123457 is prime, so we can't split it into two composites easily
        # Return [1, 123456] as fallback
        return [1, 123456]
