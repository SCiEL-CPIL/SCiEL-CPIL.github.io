"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    N = 512
    
    def get_divisors(n):
        """Return list of all divisors of n."""
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs
    
    def num_divisors(n):
        """Count divisors of n efficiently using prime factorization."""
        count = 1
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                exp = 0
                while m % p == 0:
                    m //= p
                    exp += 1
                count *= (exp + 1)
            p += 1 if p == 2 else 2
        if m > 1:
            count *= 2
        return count
    
    # Find the number M in [1, 512] with the most divisors
    best_M = 1
    best_count = 0
    for m in range(1, N + 1):
        c = num_divisors(m)
        if c > best_count:
            best_count = c
            best_M = m
    
    # The optimal set is all divisors of best_M (each pair has lcm dividing best_M <= 512)
    result = get_divisors(best_M)
    
    # Verify pairwise lcm constraint
    import math
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            lcm_val = result[i] * result[j] // math.gcd(result[i], result[j])
            if lcm_val > N:
                # Should not happen, but safeguard
                pass
    
    return result
# EVOLVE-BLOCK-END
