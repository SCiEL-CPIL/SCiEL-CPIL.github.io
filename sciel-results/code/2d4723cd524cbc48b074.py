"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def _sigma_sieve(n):
    """Compute sigma(k) for k=1..n using a sieve."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try increasing sizes to find M with most k solutions
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Try sieve sizes
    for n in [100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000]:
        if time.time() > deadline - 30:
            break
        
        sigma = _sigma_sieve(n)
        
        # Group k by M = k * sigma(k)
        groups = {}
        for k in range(1, n + 1):
            M = k * sigma[k]
            if M <= 10**18:
                if M not in groups:
                    groups[M] = []
                groups[M].append(k)
        
        # Find M with most solutions
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_ks = ks[:1000]  # At most 1000
        
        if time.time() > deadline - 20:
            break
    
    # Verify the result
    def _sigma_verify(k):
        """Compute sigma(k) by trial division."""
        if k == 1:
            return 1
        result = 1
        temp = k
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                result *= (p**(exp+1) - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp**2 - 1) // (temp - 1)
        return result
    
    # Verify all k values
    verified_ks = []
    for k in best_ks:
        if k * _sigma_verify(k) == best_M and k <= math.isqrt(best_M):
            verified_ks.append(k)
    
    if len(verified_ks) >= 1:
        return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END
