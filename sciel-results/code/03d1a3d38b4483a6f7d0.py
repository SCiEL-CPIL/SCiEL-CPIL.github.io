"""
Maximum sum of pairwise-coprime integers in [1, 256].

Key insight:
- For each prime p, the largest power p^k <= 256 is the best "representative" for that prime.
- Two numbers are coprime iff they share no prime factors.
- So the problem reduces to: partition the set of primes into groups, where each group's
  product of their largest powers must be <= 256, maximizing the sum of group products.
- Since for most primes, the largest power is just the prime itself (p^2 > 256 for p > 16),
  and even for small primes, merging their powers rarely helps (product > 256 or product < sum),
  the optimal solution is almost always to take each prime's largest power separately.
- We still do a DP to be safe and correct.

Algorithm:
1. Find all primes up to 256.
2. For each prime, compute V_p = largest power of p <= 256.
3. DP over subsets of primes: dp[mask] = max sum of products for partition of primes in mask.
4. For each valid group (product of V_p for p in group <= 256), we can form a number.
5. Include 1 as well (coprime to everything).
"""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    
    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    n = len(primes)
    
    # For each prime, compute the largest power <= N
    V = []
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        V.append(power)
    
    # DP over subsets of primes
    # dp[mask] = max sum of products for a partition of primes in mask
    # A group is a subset S of primes where prod(V[p] for p in S) <= N
    # The value of that group is prod(V[p] for p in S)
    
    full_mask = (1 << n) - 1
    
    # Precompute product of V for each mask, and validity
    mask_prod = [0] * (1 << n)
    mask_valid = [False] * (1 << n)
    mask_prod[0] = 1  # empty product
    mask_valid[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        if mask_valid[prev]:
            prod = mask_prod[prev] * V[idx]
            if prod <= N:
                mask_prod[mask] = prod
                mask_valid[mask] = True
    
    # dp[mask] = max sum of group products for partition of primes in mask
    dp = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        best = 0
        sub = mask
        while sub:
            if mask_valid[sub]:
                rest = mask ^ sub
                if rest == 0:
                    val = mask_prod[sub]
                else:
                    if dp[rest] > 0:
                        val = mask_prod[sub] + dp[rest]
                    else:
                        val = 0
                if val > best:
                    best = val
            sub = (sub - 1) & mask
        dp[mask] = best
    
    max_sum = dp[full_mask]
    
    # Reconstruct the partition
    def reconstruct(mask):
        if mask == 0:
            return []
        best_val = 0
        best_sub = 0
        sub = mask
        while sub:
            if mask_valid[sub]:
                rest = mask ^ sub
                if rest == 0:
                    val = mask_prod[sub]
                else:
                    if dp[rest] > 0:
                        val = mask_prod[sub] + dp[rest]
                    else:
                        val = 0
                if val > best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        
        if best_sub == 0:
            # Should not happen if dp is correct, but fallback
            return [mask_prod[mask]] if mask_valid[mask] else []
        
        result = [mask_prod[best_sub]]
        result.extend(reconstruct(mask ^ best_sub))
        return result
    
    selection = reconstruct(full_mask)
    
    # Add 1
    selection.append(1)
    
    # Ensure all are Python int, distinct, in [1, 256]
    final = []
    seen = set()
    for x in selection:
        x = int(x)
        if 1 <= x <= N and x not in seen:
            seen.add(x)
            final.append(x)
    
    # Verify pairwise coprimality
    for i in range(len(final)):
        for j in range(i + 1, len(final)):
            if math.gcd(final[i], final[j]) != 1:
                # Fallback: just use all V_p and 1
                final = [int(V[i]) for i in range(n)] + [1]
                break
    
    return final

# EVOLVE-BLOCK-END
