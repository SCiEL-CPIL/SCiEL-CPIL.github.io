import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # We will search up to this limit for k
    # Given SPF optimization, 2 million should be feasible within time
    N_MAX = 2000000
    
    # Precompute Smallest Prime Factor (SPF)
    # This is much faster than trial division for sigma calculation
    spf = list(range(N_MAX + 1))
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Function to compute sigma(n) using SPF
    def sigma(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        while temp > 1:
            p = spf[temp]
            exp = 0
            p_power = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                p_power *= p
            # Sum of geometric series 1 + p + ... + p^exp
            # (p^(exp+1) - 1) / (p - 1)
            result *= (p_power * p - 1) // (p - 1)
        return result

    m_to_ks = defaultdict(list)
    best_count = 0
    best_M = 333312
    best_ks = [336, 372, 434]
    
    # Iterate k
    for k in range(1, N_MAX + 1):
        if time.time() > deadline:
            break
        
        s = sigma(k)
        M = k * s
        
        if M <= 10**18:
            # We need k <= isqrt(M)
            # Check if k is a valid solution for this M
            # Note: If M is small, isqrt(M) might be small.
            # But since we are iterating k, and M = k*sigma(k), 
            # sigma(k) >= k+1 for k>1, so M > k^2, so k < sqrt(M) is generally true for k>1.
            # For k=1, M=1, sqrt(1)=1, k<=1 holds.
            # However, we should strictly check to be safe.
            if k <= math.isqrt(M):
                m_to_ks[M].append(k)
                
                # Update best if this M has more solutions
                count = len(m_to_ks[M])
                if count > best_count:
                    best_count = count
                    best_M = M
                    best_ks = m_to_ks[M]
    
    # Final verification and type conversion
    # Ensure types are Python int
    final_ks = []
    for k in best_ks:
        if isinstance(k, int) and 1 <= k <= math.isqrt(best_M):
            # Re-verify sigma just in case, though it was computed correctly
            # This is a cheap check for the final list
            if k * sigma(k) == best_M:
                final_ks.append(k)
    
    if not final_ks:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in final_ks]}
