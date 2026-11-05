import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 55  # Keep safety margin for 60s limit
    
    # Generate primes up to a reasonable limit
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(200)
    
    for p_idx, p in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate all p-smooth numbers <= TARGET
        # A number is p-smooth if all its prime factors are <= p.
        # Use the primes up to index p_idx
        smooths = set()
        
        def dfs(val, idx):
            if val > TARGET:
                return
            smooths.add(val)
            for i in range(idx, p_idx + 1):
                q = primes[i]
                if val * q > TARGET:
                    break
                dfs(val * q, i)
        
        dfs(1, 0)
        
        # Check for pair: a + b = TARGET, both in smooths
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                return [a, b]
    
    # Fallback: find a pair with small max prime factor by direct search
    # Try to find a and b where both have small prime factors
    # Use a greedy approach: try to make a smooth with small primes
    best_a, best_b = 1, TARGET - 1
    best_max_p = float('inf')
    
    # Try all a from 1 to TARGET//2 and compute P(a) and P(TARGET-a)
    # But this is O(TARGET) which is ~123k iterations, each requiring factorization
    # For safety, limit the search
    limit = min(TARGET // 2, 50000)
    for a in range(1, limit + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = TARGET - a
        # Compute largest prime factor of a
        def largest_prime_factor(n):
            if n <= 1:
                return 1
            f = 1
            d = 2
            while d * d <= n:
                while n % d == 0:
                    f = d
                    n //= d
                d += 1
            if n > 1:
                f = max(f, n)
            return f
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        mx = max(pa, pb)
        if mx < best_max_p:
            best_max_p = mx
            best_a, best_b = a, b
            if best_max_p == 2:  # Can't do better than 2
                return [best_a, best_b]
    
    return [best_a, best_b]
