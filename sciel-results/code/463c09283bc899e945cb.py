"""Search for M with many distinct k where k*sigma(k)=M, k<=isqrt(M)."""
import math
import time
import sys
from collections import defaultdict

# Precompute sigma via sieve up to a limit
def compute_sigma_sieve(limit):
    """Compute sigma(n) for all n up to limit using a divisor-sum sieve."""
    sigma = [0] * (limit + 1)
    for d in range(1, limit + 1):
        for m in range(d, limit + 1, d):
            sigma[m] += d
    return sigma

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Strategy: sieve sigma for k up to K, compute M=k*sigma(k), group by M
    # Increase K until time runs out.
    # Start with a moderate K and grow.
    K = 200000  # start
    while time.time() < deadline:
        # Ensure K won't make M exceed 1e18: k*sigma(k) <= k*k*? Actually sigma(k) can be ~ O(k log log k)
        # sigma(k) < k * H_k roughly, so k*sigma(k) < k^2 * log log k. For k=2e5, that's ~4e10*small, safe.
        # For k up to ~5e8, k^2=2.5e17, times log log ~3 => 7.5e17, still < 1e18. But sieve of 5e8 is too heavy.
        sigma = compute_sigma_sieve(K)
        groups = defaultdict(list)
        for k in range(1, K + 1):
            M = k * sigma[k]
            if M > 10**18:
                continue
            # k must be <= isqrt(M) => k^2 <= M => k^2 <= k*sigma(k) => k <= sigma(k), which is always true for k>=1? 
            # Actually we need k <= isqrt(M) => k^2 <= M = k*sigma(k) => k <= sigma(k).
            # sigma(k) >= k+1 for k>=2 (divisors include 1 and k), and sigma(1)=1 so k=1: 1<=1 ok.
            # So the constraint k<=isqrt(M) is equivalent to k<=sigma(k), which holds for all k>=1.
            # Wait: is that right? k<=isqrt(M) means k^2<=M. M=k*sigma(k). So k^2<=k*sigma(k) => k<=sigma(k) (k>0).
            # Yes. And sigma(k)>=k+1 for k>=2, sigma(1)=1. So all k>=1 satisfy it. Good.
            groups[M].append(k)
        
        # Find best M in this batch
        for M, ks in groups.items():
            cnt = len(ks)
            if cnt > best_count:
                best_count = cnt
                best_M = M
                best_k = list(ks)
        
        # Grow K; doubling tends to be efficient. But sigma sieve is O(K log K).
        # Next K:
        K = int(K * 1.6)
        if K > 30000000:  # cap to avoid memory/time blow; sieve of 30M is ~ 30M*log(30M) ~ 30M*17 ~ 5e8 ops, maybe a few seconds
            K = 30000000
            # After one more pass at max, break
            break
        # Check time
        if time.time() >= deadline - 5:
            break
    
    # Final safety: ensure types are plain Python int and list
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)
    return {'M': best_M, 'k': best_k}
