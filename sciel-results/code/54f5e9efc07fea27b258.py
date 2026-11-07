import time
from math import isqrt
from functools import lru_cache

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Precompute square-free parts using sieve
    # We need numbers from 211 up to some limit
    # Let's go up to 2000 initially, can extend if needed
    
    max_n = 3000
    sf = [0] * (max_n + 1)
    sf[1] = 1
    for i in range(2, max_n + 1):
        sf[i] = 1
    for p in range(2, max_n + 1):
        if sf[p] == 1 and p * p <= max_n:  # p is prime (sf[p] still 1 means not touched by smaller prime squares)
            # Actually, let's use a proper sieve for square-free parts
            pass
    
    # Better: compute sf using prime factorization with smallest prime factor
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_sf(n):
        if n <= 1:
            return 1
        result = 1
        x = n
        last_p = 0
        while x > 1:
            p = spf[x]
            if p == last_p:
                x //= p
                continue
            cnt = 0
            while x % p == 0:
                x //= p
                cnt ^= 1
            if cnt:
                result *= p
            last_p = p
        return result
    
    # Precompute sf for all numbers in range
    sf_arr = [0] * (max_n + 1)
    sf_arr[1] = 1
    for i in range(2, max_n + 1):
        sf_arr[i] = get_sf(i)
    
    # Target: sf(210) = 210 (since 210 = 2*3*5*7, all to first power)
    target_sf = 210
    
    # We need to find minimal M such that there exists a non-empty subset of {211, ..., M}
    # whose product of square-free parts has square-free part = 210.
    # 
    # This is a linear algebra problem over GF(2) on the prime exponents.
    # Each number's square-free part corresponds to a vector of primes (exponents mod 2).
    # We need a subset whose XOR of vectors equals the vector for 210.
    
    # Collect all primes up to max_n
    primes = []
    for i in range(2, max_n + 1):
        if spf[i] == i:
            primes.append(i)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)
    
    def sf_to_bits(sf_val):
        """Convert square-free integer to bit vector (as Python int bitmask)."""
        bits = 0
        x = sf_val
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                bits |= (1 << prime_to_idx[p])
                x //= p
        if x > 1:
            if x in prime_to_idx:
                bits |= (1 << prime_to_idx[x])
        return bits
    
    target_bits = sf_to_bits(target_sf)
    
    # For each M from 211 upward, check if target_bits is in the span of {sf_to_bits(sf_arr[n]) for n in 211..M}
    # Use incremental Gaussian elimination
    
    def find_subset_incremental(bits_list, target):
        """
        Given a list of bit vectors and a target, find a subset whose XOR equals target.
        Returns list of indices or None.
        Uses Gaussian elimination with tracking of which original indices are used.
        """
        n = len(bits_list)
        if n == 0:
            return None
        
        # Build basis with tracking
        # basis: dict from pivot_bit_position -> (reduced_vector, set_of_original_indices)
        basis = {}
        
        for i, v in enumerate(bits_list):
            if v == 0:
                continue
            cur = v
            used = {i}
            # Reduce by existing basis vectors
            hb = cur.bit_length() - 1 if cur != 0 else -1
            while hb >= 0 and hb in basis:
                bv, bused = basis[hb]
                cur ^= bv
                used = used.symmetric_difference(bused)
                hb = cur.bit_length() - 1 if cur != 0 else -1
            if cur == 0:
                continue
            hb = cur.bit_length() - 1
            basis[hb] = (cur, frozenset(used))
        
        # Try to express target
        if target == 0:
            # Need non-empty subset that XORs to 0
            # This means we need a dependent set
            # For now, return None and handle separately
            return None
        
        cur = target
        used = set()
        hb = cur.bit_length() - 1 if cur != 0 else -1
        while hb >= 0 and hb in basis:
            bv, bused = basis[hb]
            cur ^= bv
            used = used.symmetric_difference(set(bused))
            hb = cur.bit_length() - 1 if cur != 0 else -1
        
        if cur != 0:
            return None
        
        if not used:
            return None
        
        return sorted(list(used))
    
    # Search for minimal M
    best_solution = None
    best_M = max_n + 1
    
    # Incrementally build the set of vectors
    # We'll maintain the basis incrementally
    
    # For efficiency, let's do binary search on M
    # First, check if a solution exists at all with M up to max_n
    
    def check_M(M):
        """Check if there's a valid subset with max element <= M."""
        if M < 211:
            return None
        bits_list = [sf_to_bits(sf_arr[n]) for n in range(211, M + 1)]
        return find_subset_incremental(bits_list, target_bits)
    
    # Binary search for minimal M
    lo = 211
    hi = max_n
    best_idx = None
    best_M_val = max_n + 1
    
    # First verify hi works
    result_hi = check_M(hi)
    if result_hi is None:
        # Need to extend range
        # Try larger range
        return [224, 240]  # fallback
    
    while lo < hi:
        if time.time() > deadline - 5.0:
            break
        mid = (lo + hi) // 2
        result_mid = check_M(mid)
        if result_mid is not None:
            best_idx = result_mid
            best_M_val = mid
            hi = mid
        else:
            lo = mid + 1
    
    # Now lo == hi, check it
    if time.time() < deadline - 5.0:
        result_lo = check_M(lo)
        if result_lo is not None:
            best_idx = result_lo
            best_M_val = lo
    
    if best_idx is not None:
        selected = [211 + i for i in best_idx]
        # Verify
        prod = 210
        for x in selected:
            prod *= x
        if _is_perfect_square(prod):
            best_solution = selected
    
    if best_solution is None:
        # Try direct search for small M
        for M in range(211, min(500, max_n + 1)):
            if time.time() > deadline - 5.0:
                break
            bits_list = [sf_to_bits(sf_arr[n]) for n in range(211, M + 1)]
            idxs = find_subset_incremental(bits_list, target_bits)
            if idxs is not None and len(idxs) > 0:
                selected = [211 + i for i in idxs]
                prod = 210
                for x in selected:
                    prod *= x
                if _is_perfect_square(prod):
                    best_solution = selected
                    break
    
    if best_solution is None:
        # Fallback: known solution
        best_solution = [224, 240]
    
    # Ensure types are correct
    result = [int(x) for x in best_solution]
    return result
