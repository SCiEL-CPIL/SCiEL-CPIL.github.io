"""Program to find M with many k such that k*sigma(k)=M, k<=sqrt(M)."""
import math
import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Strategy: sieve for k*sigma(k) up to a large bound, then find M with many k
    # where k <= sqrt(M).
    #
    # Key insight: For k*sigma(k) = M, we need k <= sqrt(M) = sqrt(k*sigma(k)),
    # which means k^2 <= k*sigma(k) => k <= sigma(k).
    # sigma(k) >= k+1 for k >= 1 (since 1 is a divisor), so k < sigma(k) for k >= 2.
    # For k=1: sigma(1)=1, so k*sigma(k)=1, sqrt(1)=1, k<=sqrt(M) holds.
    #
    # Actually the constraint is k_i <= isqrt(M). Since M = k_i * sigma(k_i),
    # we need k_i <= isqrt(k_i * sigma(k_i)), i.e., k_i^2 <= k_i * sigma(k_i),
    # i.e., k_i <= sigma(k_i). This is always true for k_i >= 1 (sigma(k) >= k+1 for k>=2,
    # and for k=1, sigma(1)=1, so 1<=1 holds).
    #
    # So ALL k satisfy the constraint! We just need to find M that is achieved by
    # many distinct k values.
    #
    # Plan: Compute k*sigma(k) for k=1..N, group by value, find the value with
    # the most k's. Then verify M <= 10^18 and k_i <= isqrt(M).

    # How large can N be? k*sigma(k) can be up to 10^18.
    # sigma(k) is roughly O(k * log log k), so k*sigma(k) is roughly O(k^2 log log k).
    # For k*sigma(k) <= 10^18, we need k^2 * log log k <= 10^18, so k <= ~10^9.
    # That's too large for a sieve.
    #
    # But we don't need ALL k. We need to find collisions (same M from different k).
    # Collisions are rare for large k. Let's focus on smaller k where collisions are denser.
    #
    # From the knowledge: "fast O(N log N) sieve for k*sigma(k) up to 10^6"
    # gave score 1.667 (meaning 5 solutions vs 3 in baseline).
    #
    # Let's try a larger sieve. The key is: how many collisions can we find?
    #
    # Let's compute sigma(k) for k up to N using a sieve, then compute k*sigma(k),
    # and group by value.

    # Determine sieve size based on available time
    # sigma computation via sieve is O(N log N)
    # Let's try N = 5_000_000 first, check time, then scale up.

    # Actually, let's be smart: do the sieve in chunks or pick N based on time budget.
    # For N=5M, the sieve takes maybe 5-10 seconds in Python. Let's try.

    # Better approach: use a modified sieve that directly computes sigma.
    # sigma is multiplicative. We can compute it with a linear sieve.

    # Let's use a standard sieve for sigma:
    # sigma[1] = 1
    # For each prime p, for multiples m of p, add the p-power contribution.
    #
    # Actually, the simplest: initialize sigma = [1]*(N+1), then for each i from 1 to N,
    # add i to sigma[j] for all j that are multiples of i. This is O(N log N).
    # But that's the divisor-sum sieve. Let's use it.

    # For N up to ~10M, O(N log N) in pure Python might be slow.
    # Let's use NumPy for the sieve.

    try:
        import numpy as np
        have_numpy = True
    except ImportError:
        have_numpy = False

    if have_numpy:
        # Use NumPy for speed
        # We'll try increasing N until we run out of time
        # Start with a reasonable N and see how long it takes

        # First, do a quick timing test with a small N
        N_test = 100000
        t0 = time.time()
        sigma_test = np.ones(N_test + 1, dtype=np.int64)
        for i in range(1, N_test + 1):
            sigma_test[i::i] += i
        t_sieve_small = time.time() - t0
        # This is O(N log N). For N=100k, if it takes T, for N=10M it takes ~100*100*T = 10000*T
        # Actually it's N * sum(1/i) = N * log(N) roughly.
        # Ratio: (N2*log(N2)) / (N1*log(N1))

        # Let's estimate: for N=100k, H(100k)≈12. So operations ≈ 100k*12 = 1.2M
        # For N=5M, H(5M)≈15. So operations ≈ 5M*15 = 75M. Ratio ≈ 62x.
        # If small test takes 0.1s, large takes ~6s.

        # Let's just try N values and check time
        best_M = 333312
        best_k = [336, 372, 434]

        # Try several N values
        for N in [1_000_000, 3_000_000, 5_000_000, 10_000_000, 20_000_000]:
            if time.time() > deadline - 20:
                break

            t0 = time.time()
            sigma = np.ones(N + 1, dtype=np.int64)
            for i in range(1, N + 1):
                sigma[i::i] += i
            t_sieve = time.time() - t0

            if time.time() > deadline - 10:
                break

            # Compute k*sigma(k)
            # k goes from 1 to N
            ks = np.arange(1, N + 1, dtype=np.int64)
            products = ks * sigma[1:N+1]

            # Filter: product <= 10^18 (should be fine for N up to 20M, k*sigma(k) ~ k^2*log(k) ~ 20M^2*~16 = 6.4e15, ok)
            # Also need k <= isqrt(product), which is always true as shown above.
            # But let's verify: k^2 <= k*sigma(k) => k <= sigma(k). Always true.

            # Now find the value with the most occurrences
            # Use numpy unique with return_counts
            # But products can be up to ~10^15, and we have up to 20M elements
            # unique on 20M elements might be slow. Let's try.

            t0 = time.time()
            # Sort and find groups
            # Alternative: use a dict for small N, numpy for large N
            if N <= 5_000_000:
                # Use a dictionary approach
                from collections import defaultdict
                groups = defaultdict(list)
                for idx in range(N):
                    p = int(products[idx])
                    if p <= 10**18:
                        groups[p].append(int(ks[idx]))
                
                # Find the M with most k's
                best_count = 0
                best_M_candidate = None
                best_k_candidate = None
                for m, klist in groups.items():
                    if len(klist) > best_count:
                        best_count = len(klist)
                        best_M_candidate = m
                        best_k_candidate = klist
                
                if best_M_candidate is not None and best_count > len(best_k):
                    # Verify
                    M_val = best_M_candidate
                    k_list = best_k_candidate[:1000]  # at most 1000
                    # Verify each k
                    valid_k = []
                    for k in k_list:
                        if k <= math.isqrt(M_val) and k * int(sigma[k]) == M_val:
                            valid_k.append(k)
                    if len(valid_k) > len(best_k):
                        best_M = M_val
                        best_k = valid_k
            else:
                # For larger N, use numpy
                # This might use a lot of memory. Let's be careful.
                # Actually, let's just use the dict approach but it'll be slower
                # For N=10M-20M, the dict approach in Python will be too slow.
                # Let's skip large N for now or use a different approach.
                pass

            if time.time() > deadline - 5:
                break

    # Fallback: if numpy approach didn't improve, use the known solution
    # Also, let's try a pure Python sieve for moderate N if numpy failed

    if len(best_k) <= 3:
        # Try pure Python sieve for N up to ~2M
        N = 2_000_000
        if time.time() < deadline - 30:
            sigma = [0] * (N + 1)
            sigma[1] = 1
            for i in range(1, N + 1):
                for j in range(i, N + 1, i):
                    sigma[j] += i
            
            if time.time() < deadline - 10:
                from collections import defaultdict
                groups = defaultdict(list)
                for k in range(1, N + 1):
                    p = k * sigma[k]
                    if p <= 10**18:
                        groups[p].append(k)
                
                best_count = len(best_k)
                for m, klist in groups.items():
                    if len(klist) > best_count:
                        # Verify
                        valid_k = []
                        for k in klist[:1000]:
                            if k <= math.isqrt(m) and k * sigma[k] == m:
                                valid_k.append(k)
                        if len(valid_k) > best_count:
                            best_count = len(valid_k)
                            best_M = m
                            best_k = valid_k

    # Final verification
    M_final = best_M
    k_final = best_k[:1000]
    
    # Verify all constraints
    valid_k = []
    for k in k_final:
        if not isinstance(k, int) or isinstance(k, bool):
            continue
        if k < 1 or k > math.isqrt(M_final):
            continue
        # Recompute sigma(k) to verify
        # For safety, compute sigma(k) directly
        s = 0
        for d in range(1, math.isqrt(k) + 1):
            if k % d == 0:
                s += d
                if d != k // d:
                    s += k // d
        if k * s == M_final:
            valid_k.append(k)
    
    # Ensure distinct
    valid_k = list(dict.fromkeys(valid_k))
    
    if len(valid_k) < 2:
        # Fallback to known solution
        M_final = 333312
        valid_k = [336, 372, 434]

    return {'M': int(M_final), 'k': [int(x) for x in valid_k]}

# EVOLVE-BLOCK-END
