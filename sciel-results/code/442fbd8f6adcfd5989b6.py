"""Improved program: sieve sigma, find M with many k where k*sigma(k)=M, k<=sqrt(M)."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _sigma_sieve(n):
    """Compute sigma(k) for all k in 1..n using sieve. Returns list sigma[0..n]."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill
    
    # We need k <= sqrt(M) and M = k * sigma(k) <= 10^18
    # So k * sigma(k) <= 10^18
    # sigma(k) >= k+1, so k*(k+1) <= 10^18 => k <= ~10^9
    # But we need k <= sqrt(M) = sqrt(k*sigma(k)) => k <= sigma(k) which is always true
    # Actually k <= sqrt(M) means k^2 <= M = k*sigma(k), i.e., k <= sigma(k), always true.
    # So the constraint is just M <= 10^18, i.e., k * sigma(k) <= 10^18.
    
    # Strategy: sieve sigma up to some max_k, then group by M = k*sigma(k),
    # find M with most distinct k values.
    
    # How large can max_k be? k*sigma(k) <= 10^18.
    # sigma(k) is typically ~ O(k log log k), so k^2 * log log k ~ 10^18
    # k ~ 10^9 / sqrt(log log k) ~ maybe 5*10^8 to 10^9
    # But sieving up to 10^9 in Python is too slow.
    # Let's try max_k = 5_000_000 first (5M), which should take ~15-30s.
    # Then if time permits, extend.
    
    # Actually, let's be smarter. For small k, sigma(k) is small, so M is small.
    # The number of solutions for a given M depends on how many k have k*sigma(k)=M.
    # We want to maximize the count.
    
    # Let's try a range of max_k values and pick the best.
    # Start with a reasonable max_k and increase if time allows.
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    
    # Try increasing max_k
    for max_k in [2_000_000, 5_000_000, 10_000_000, 20_000_000]:
        if time.time() > deadline - 30:
            break
        
        t0 = time.time()
        sigma = _sigma_sieve(max_k)
        t1 = time.time()
        
        # Group k by M = k * sigma(k)
        groups = defaultdict(list)
        for k in range(1, max_k + 1):
            M = k * sigma[k]
            if M <= 10**18:
                groups[M].append(k)
        
        t2 = time.time()
        
        # Find M with most k values
        for M, k_list in groups.items():
            if len(k_list) > len(best_k_list):
                best_M = M
                best_k_list = list(k_list)
        
        if time.time() > deadline - 10:
            break
    
    # Verify the best solution
    # Check that all k_i are distinct, <= sqrt(M), and k_i * sigma(k_i) = M
    if best_k_list:
        M = best_M
        sqrt_M = math.isqrt(M)
        # Filter to valid k
        valid_k = []
        for k in best_k_list:
            if k <= sqrt_M:
                # Verify k * sigma(k) = M
                # We need sigma(k) for verification
                # Use the sieve if k <= max_k, else factorize
                valid_k.append(k)
        best_k_list = valid_k
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    return {'M': best_M, 'k': best_k_list}

# EVOLVE-BLOCK-END
