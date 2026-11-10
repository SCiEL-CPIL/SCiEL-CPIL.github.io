"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 150.0  # wall-clock limit; keep safety margin

    # We need to find M <= 10^18 with many distinct k_i <= sqrt(M) such that k_i * sigma(k_i) = M.
    # Strategy: sieve sigma(k) for k up to K, compute M_k = k * sigma(k), group by M,
    # then pick the M with the most preimages k_i where k_i <= isqrt(M) and M <= 10^18.

    # Choose K so that K * sigma(K) is manageable. sigma(K) ~ O(K log log K), so M_k ~ K^2 log log K.
    # For M <= 10^18, we need K^2 log log K <= 10^18, so K ~ 10^9 / sqrt(log log K).
    # But we can't sieve up to 10^9. Let's use a reasonable K and check constraints.
    # With K = 5_000_000, sigma(K) ~ 20M, so M_k ~ 10^11, well under 10^18.
    # Actually we want to maximize the number of preimages. Let's use a large sieve.

    # Use K = 10_000_000 for a good balance.
    K = 10_000_000

    # Sieve sigma(n) for n = 1..K
    sigma = [0] * (K + 1)
    for i in range(1, K + 1):
        for j in range(i, K + 1, i):
            sigma[j] += i

    # Check time
    if time.time() > deadline - 20:
        # Fallback to known 3-solution
        return {'M': 333312, 'k': [336, 372, 434]}

    # Compute M_k = k * sigma(k) for k = 1..K
    # Group by M_k
    from collections import defaultdict
    groups = defaultdict(list)
    for k in range(1, K + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            groups[M_val].append(k)

    # For each M, filter k_i <= isqrt(M)
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    for M_val, ks in groups.items():
        # Filter: k_i <= isqrt(M_val)
        isq = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= isq]
        # Also need k_i * sigma(k_i) = M_val (already guaranteed by construction)
        # Distinctness: they are distinct by construction (each k appears once)
        cnt = len(valid_ks)
        if cnt > best_count:
            best_count = cnt
            best_M = M_val
            best_k = valid_ks

        if time.time() > deadline - 10:
            break

    # Verify the best result with exact factorization
    def sigma_exact(n):
        if n <= 0:
            return 0
        result = 1
        temp = n
        f = 2
        while f * f <= temp:
            if temp % f == 0:
                p_power = 1
                p_sum = 1
                while temp % f == 0:
                    temp //= f
                    p_power *= f
                    p_sum += p_power
                result *= p_sum
            f += 1
        if temp > 1:
            result *= (1 + temp)
        return result

    # Verify
    valid_final = []
    for k in best_k:
        if k * sigma_exact(k) == best_M and k <= math.isqrt(best_M):
            valid_final.append(k)

    if len(valid_final) >= best_count:
        best_k = valid_final
    else:
        # Fallback
        best_M = 333312
        best_k = [336, 372, 434]

    # Ensure types are correct
    M_out = int(best_M)
    k_out = [int(k) for k in best_k]

    return {'M': M_out, 'k': k_out}
# EVOLVE-BLOCK-END
