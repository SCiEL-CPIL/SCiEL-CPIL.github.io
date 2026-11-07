import time
import math
from collections import defaultdict

def _sigma_sieve(n):
    """Compute sigma(k) for all k in [1, n] using a sieve. Returns list of ints."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def _sigma_single(k):
    """Compute sigma(k) for a single integer k via factorization."""
    if k == 1:
        return 1
    result = 1
    n = k
    d = 2
    while d * d <= n:
        if n % d == 0:
            p_power = 0
            p_sum = 1
            while n % d == 0:
                n //= d
                p_power += 1
                p_sum += d ** p_power
            result *= p_sum
        d += 1 if d == 2 else 2
    if n > 1:
        result *= (1 + n)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Precomputed known good result
    # M = 333312, k = [336, 372, 434] gives score 3 (seed gives 3/2 = 1.5, current gives 5/3 = 1.667)
    # Actually let me verify: sigma(336) = sigma(2^4 * 3 * 7) = (1+2+4+8+16)*(1+3)*(1+7) = 31*4*8 = 992
    # 336 * 992 = 333312. Yes.
    # sigma(372) = sigma(2^2 * 3 * 31) = (1+2+4)*(1+3)*(1+31) = 7*4*32 = 896
    # 372 * 896 = 333312. Yes.
    # sigma(434) = sigma(2 * 7 * 31) = (1+2)*(1+7)*(1+31) = 3*8*32 = 768
    # 434 * 768 = 333312. Yes.
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3

    def run_search(N):
        """Search for M with most k values where k*sigma(k) = M, for k in [1, N]."""
        sigma = _sigma_sieve(N)
        
        # Build dictionary: M -> list of k's
        m_to_ks = defaultdict(list)
        for k in range(1, N + 1):
            m_val = k * sigma[k]
            if m_val <= 10**18:
                m_to_ks[m_val].append(k)
        
        # Find M with most k's
        max_count = 0
        best_m = 0
        best_k_list = []
        for m, ks in m_to_ks.items():
            if len(ks) > max_count:
                max_count = len(ks)
                best_m = m
                best_k_list = ks
        
        return best_m, best_k_list, max_count

    # Try increasing N values
    target_Ns = [50000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000]
    
    for N in target_Ns:
        remaining = deadline - time.time()
        
        # Conservative time estimates:
        # N=50K: ~0.1s, N=200K: ~0.5s, N=500K: ~1.5s, N=1M: ~4s, N=2M: ~10s
        # N=5M: ~30s, N=10M: ~60s, N=20M: ~120s, N=50M: too long
        if N <= 50000 and remaining < 2:
            continue
        if N <= 200000 and remaining < 3:
            continue
        if N <= 500000 and remaining < 5:
            continue
        if N <= 1000000 and remaining < 10:
            continue
        if N <= 2000000 and remaining < 20:
            continue
        if N <= 5000000 and remaining < 40:
            continue
        if N <= 10000000 and remaining < 70:
            continue
        if N <= 20000000 and remaining < 120:
            continue
        if N > 20000000 and remaining < 130:
            continue

        try:
            m, ks, count = run_search(N)
            if count > best_count:
                best_M = m
                best_ks = ks
                best_count = count
        except Exception:
            break

        if time.time() > deadline - 10:
            break

    # Verify and return
    # Ensure all k_i satisfy k_i * sigma(k_i) = M and k_i <= sqrt(M)
    valid_ks = []
    for k in best_ks:
        if k <= math.isqrt(best_M):
            s = _sigma_single(k)
            if k * s == best_M:
                valid_ks.append(k)
    
    if len(valid_ks) >= best_count or best_count <= 3:
        # Use verified ks
        return {'M': int(best_M), 'k': [int(k) for k in valid_ks]}
    else:
        # Fallback to known good
        return {'M': int(333312), 'k': [336, 372, 434]}
