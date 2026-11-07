"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def _sigma(n):
    """Compute sigma(n) = sum of divisors of n."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            term = 1
            pk = 1
            while temp % p == 0:
                temp //= p
                pk *= p
                term += pk
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    deadline = time.time() + 140.0
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Strategy: For various k, compute M = k*sigma(k), group by M, count solutions
    # We need to search k up to some bound. Since M <= 10^18 and k <= sqrt(M),
    # k can be up to 10^9. But sigma(k) >= k+1, so k*sigma(k) >= k*(k+1) > k^2.
    # Thus k < sqrt(M) <= 10^9.
    # We can't enumerate all k up to 10^9, but we can be strategic.
    
    # Key insight: if k = product of p_i^a_i, then sigma(k) = product of (p_i^(a_i+1)-1)/(p_i-1)
    # M = k * sigma(k) = product of p_i^a_i * (p_i^(a_i+1)-1)/(p_i-1)
    
    # For small k, let's enumerate and group.
    # For larger k, we need a smarter approach.
    
    # Phase 1: Enumerate k up to some limit L and group by M
    # The challenge is that k can be up to 10^9, but most M values will have few solutions.
    # Let's try to find M with many solutions by being clever about which k to check.
    
    # Approach: Generate k values that are likely to produce "popular" M values.
    # M = k*sigma(k). If k has many divisors, sigma(k) is large.
    # 
    # Alternative: For each candidate M, find all k such that k*sigma(k)=M.
    # This means sigma(k) = M/k, so k must divide M and sigma(k) = M/k.
    # 
    # Better approach: Enumerate k in ranges and build a dictionary M -> list of k.
    # But k goes up to 10^9 which is too large.
    #
    # Let's think differently. The number of solutions is bounded.
    # Known: M=333312 has 3 solutions. Can we do better?
    #
    # Let's enumerate k up to a moderate limit first.
    
    L = 500000  # Start with moderate limit
    m_to_ks = defaultdict(list)
    
    for k in range(1, L + 1):
        if time.time() > deadline - 20:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            m_to_ks[M].append(k)
    
    # Check for best
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
    
    # Phase 2: Try larger k values, focusing on "smooth" numbers or numbers with specific structures
    # that might produce collisions.
    # 
    # Key observation: If k1*sigma(k1) = k2*sigma(k2), then the ratio k1/k2 = sigma(k2)/sigma(k1).
    # For small ratios, we can search for pairs.
    #
    # Let's try extending the search for k in [L, L2] but only for k that are "interesting"
    # (e.g., highly composite, or with specific prime factorizations).
    
    # Actually, let's try a different approach: search for M values that are "popular"
    # by checking specific families of k.
    
    # Family 1: k = 2^a * 3^b * 5^c * 7^d (smooth numbers)
    # These are more likely to produce collisions because sigma has nice multiplicative properties.
    
    def gen_smooth(limit_primes, max_val):
        """Generate smooth numbers up to max_val using primes in limit_primes."""
        primes = limit_primes
        result = set()
        n_primes = len(primes)
        
        def helper(idx, current):
            if current > max_val:
                return
            if idx == n_primes:
                result.add(current)
                return
            p = primes[idx]
            val = 1
            while val <= max_val:
                helper(idx + 1, current * val)
                if val > max_val // p:
                    break
                val *= p
        
        helper(0, 1)
        return result
    
    # Generate 7-smooth numbers (primes 2,3,5,7) up to 10^9
    # This is a smaller set than all numbers up to 10^9
    
    smooth_numbers = gen_smooth([2, 3, 5, 7], 10**9)
    
    # Filter: only those not already checked
    already_checked = set(range(1, L + 1))
    new_smooth = [k for k in smooth_numbers if k > L and k not in already_checked and k <= 10**9]
    new_smooth.sort()
    
    for k in new_smooth:
        if time.time() > deadline - 15:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            m_to_ks[M].append(k)
    
    # Re-check for best
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
    
    # Phase 3: Try 11-smooth numbers
    smooth_numbers_11 = gen_smooth([2, 3, 5, 7, 11], 10**9)
    already_checked.update(new_smooth)
    new_smooth_11 = [k for k in smooth_numbers_11 if k not in already_checked and k <= 10**9]
    new_smooth_11.sort()
    
    for k in new_smooth_11:
        if time.time() > deadline - 10:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            m_to_ks[M].append(k)
    
    # Re-check for best
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
    
    # Phase 4: For the best M found, verify and also try to find more k values
    # by checking if there are other k (not yet found) with k*sigma(k) = best_M.
    # This is hard in general, but for specific M we can factor M and check divisors.
    
    if best_M > 333312:
        # Try to find additional k for best_M
        # k must divide best_M and sigma(k) = best_M // k
        # Let's check all divisors of best_M
        def divisors(n):
            divs = []
            for i in range(1, int(math.isqrt(n)) + 1):
                if n % i == 0:
                    divs.append(i)
                    if i != n // i:
                        divs.append(n // i)
            return divs
        
        all_divs = divisors(best_M)
        found_ks = set()
        for k in all_divs:
            if k * _sigma(k) == best_M:
                found_ks.add(k)
        
        if len(found_ks) > best_count:
            best_count = len(found_ks)
            best_k = list(found_ks)
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    # Verify all constraints
    valid_ks = []
    for k in best_k:
        if 1 <= k <= math.isqrt(best_M) and k * _sigma(k) == best_M:
            valid_ks.append(k)
    
    if len(valid_ks) >= 3:
        best_k = valid_ks
        best_count = len(valid_ks)
    else:
        # Fall back to known good answer
        best_M = 333312
        best_k = [336, 372, 434]
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
