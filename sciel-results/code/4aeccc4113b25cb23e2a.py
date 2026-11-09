"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def _sigma_linear_sieve_correct(n, deadline=None):
    """Compute sigma(k) for k=1..n using a correct linear sieve."""
    if n < 1:
        return np.zeros(1, dtype=np.int64)
    
    sigma = np.zeros(n + 1, dtype=np.int64)
    sigma[1] = 1
    
    is_composite = np.zeros(n + 1, dtype=bool)
    primes = []
    spf = np.zeros(n + 1, dtype=np.int64)
    pp_sigma = np.zeros(n + 1, dtype=np.int64)
    pp_val = np.zeros(n + 1, dtype=np.int64)
    
    for i in range(2, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        if not is_composite[i]:
            primes.append(i)
            spf[i] = i
            pp_sigma[i] = 1 + i
            pp_val[i] = i
            sigma[i] = 1 + i
        for p in primes:
            if i * p > n:
                break
            is_composite[i * p] = True
            spf[i * p] = p
            if p == spf[i]:
                pp_val[i * p] = pp_val[i] * p
                pp_sigma[i * p] = pp_sigma[i] + pp_val[i * p]
                sigma[i * p] = (sigma[i] // pp_sigma[i]) * pp_sigma[i * p]
                break
            else:
                pp_val[i * p] = p
                pp_sigma[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
    
    return sigma

def _enumerate_square_free(n, deadline=None):
    """
    Enumerate all square-free integers up to n and compute sigma for each.
    Returns (list_of_square_free_k, dict mapping k -> sigma(k)).
    Uses inclusion-exclusion via Möbius function approach.
    """
    if n < 1:
        return [], {}
    
    # First, compute the Möbius function and identify square-free numbers
    # using a sieve
    mu = np.ones(n + 1, dtype=np.int8)  # Möbius function
    is_square_free = np.ones(n + 1, dtype=bool)
    is_square_free[0] = False
    
    primes = []
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    
    # Sieve for primes and compute mu
    for i in range(2, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        if is_prime[i]:
            primes.append(i)
            for j in range(i, n + 1, i):
                is_prime[j] = False
                mu[j] = -mu[j]
            i2 = i * i
            if i2 <= n:
                for j in range(i2, n + 1, i2):
                    is_square_free[j] = False
                    mu[j] = 0
    
    # For square-free numbers, sigma(k) = product of (1+p) for prime factors p
    # We can compute this efficiently using the fact that for square-free k,
    # sigma is multiplicative and sigma(p) = 1+p
    
    sigma_sq = np.zeros(n + 1, dtype=np.int64)
    sigma_sq[1] = 1
    
    # For each square-free number, compute sigma using its prime factors
    # Since k is square-free, k = p1 * p2 * ... * pm, and sigma(k) = (1+p1)(1+p2)...(1+pm)
    
    # Efficient approach: use the fact that for square-free k,
    # we can build up sigma incrementally
    # Actually, let's just use the linear sieve result filtered to square-free
    
    return is_square_free, mu

def _find_square_free_solutions(n, deadline=None):
    """
    Find all square-free k up to n, compute sigma(k), and group by M = k*sigma(k).
    Returns dict: M -> list of square-free k values.
    """
    if n < 1:
        return {}
    
    start_time = time.time()
    
    # Compute sigma for all numbers up to n using linear sieve
    sigma = _sigma_linear_sieve_correct(n, deadline=deadline)
    
    if time.time() - start_time > (deadline - start_time) * 0.7:
        # Not enough time left, work with what we have
        pass
    
    # Identify square-free numbers using a simple sieve
    is_square_free = np.ones(n + 1, dtype=bool)
    is_square_free[0] = False
    if n >= 1:
        is_square_free[1] = True
    
    # Mark non-square-free numbers
    limit = int(math.isqrt(n))
    for p in range(2, limit + 1):
        if deadline is not None and time.time() > deadline:
            break
        p2 = p * p
        for j in range(p2, n + 1, p2):
            is_square_free[j] = False
    
    # Group square-free k by M = k * sigma(k)
    groups = defaultdict(list)
    
    for k in range(1, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        if is_square_free[k]:
            m = k * sigma[k]
            if m <= 10**18:
                groups[m].append(k)
    
    return groups

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with known good result
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
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
    
    # Phase 1: General sieve approach to find collisions
    sieve_sizes = [5000000, 10000000, 20000000]
    
    for n in sieve_sizes:
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        
        # Estimate time needed: linear sieve is ~O(n) with small constant
        # For n=10M, expect ~2-5 seconds
        est_time = n / 5000000.0 * 3.0  # ~3 sec per 5M
        if est_time > remaining * 0.6:
            break
        
        sigma = _sigma_linear_sieve_correct(n, deadline=deadline - 10)
        
        if time.time() - start_time > deadline - 15:
            break
        
        # Group all k by M = k * sigma(k)
        groups = defaultdict(list)
        
        for k in range(1, n + 1):
            if time.time() - start_time > deadline - 10:
                break
            m = k * sigma[k]
            if m <= 10**18:
                groups[m].append(k)
        
        # Find M with most solutions
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_ks = ks[:1000]
        
        del groups
        del sigma
    
    # Phase 2: Targeted square-free enumeration
    # Square-free numbers have sigma(k) = prod(1+p), which tends to create
    # more collisions because the values are more "structured"
    
    if time.time() - start_time < deadline - 20:
        # Try square-free enumeration at various sizes
        sq_sizes = [5000000, 10000000, 20000000]
        
        for n in sq_sizes:
            elapsed = time.time() - start_time
            remaining = deadline - start_time - elapsed
            if remaining < 15:
                break
            
            # Estimate: square-free sieve + grouping is ~1.5x slower than plain sieve
            est_time = n / 5000000.0 * 4.5
            if est_time > remaining * 0.7:
                break
            
            groups_sq = _find_square_free_solutions(n, deadline=deadline - 10)
            
            if time.time() - start_time > deadline - 10:
                break
            
            # Find M with most square-free solutions
            for M, ks in groups_sq.items():
                if len(ks) > best_count:
                    best_count = len(ks)
                    best_M = M
                    best_ks = ks[:1000]
            
            del groups_sq
    
    # Phase 3: Merge - check if we can find more solutions for best_M
    # by also checking non-square-free k that might satisfy the equation
    if time.time() - start_time < deadline - 15 and best_count >= 3:
        # Try to find additional k values for the current best_M
        # by doing a targeted search
        max_k = math.isqrt(best_M)
        
        if max_k <= 5000000:
            # Do a targeted search for k values
            found_ks = set()
            for k in best_ks:
                found_ks.add(k)
            
            # Search in chunks
            chunk_size = 100000
            for start_k in range(1, max_k + 1, chunk_size):
                if time.time() - start_time > deadline - 10:
                    break
                end_k = min(start_k + chunk_size, max_k + 1)
                
                for k in range(start_k, end_k):
                    if k in found_ks:
                        continue
                    if k * _sigma_verify(k) == best_M:
                        found_ks.add(k)
            
            if len(found_ks) > best_count:
                best_count = len(found_ks)
                best_ks = list(found_ks)[:1000]
    
    # Final verification
    verified_ks = []
    for k in best_ks:
        if isinstance(k, int) and 1 <= k <= math.isqrt(best_M) and k * _sigma_verify(k) == best_M:
            verified_ks.append(int(k))
    
    # Remove duplicates
    verified_ks = list(set(verified_ks))
    
    if len(verified_ks) >= 1:
        return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END

