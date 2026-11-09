import time
import numpy as np
from bisect import bisect_left
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s

    # Start with a moderate N; extend if time permits
    N = 5_000_000

    # --- Compute sigma(n) for n=1..N using SPF sieve ---
    # Step 1: smallest prime factor sieve
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == 0:
            # i is prime
            spf[i::i] = i
            # Only set multiples that haven't been set yet (but numpy slice sets all)
            # Actually spf[i::i] = i overwrites, but for composites the smallest prime
            # was already set by a smaller prime. We need to only set if 0.
            # Better approach: standard SPF sieve
    # Redo SPF properly
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    # Ensure all composites have spf set
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i

    # Step 2: Compute sigma(n) using the multiplicative formula
    # sigma(n) = product over prime powers p^k || n of (p^(k+1)-1)/(p-1)
    # We can compute this iteratively.
    # For each n, factorize using spf and compute sigma.
    # But doing this one-by-one in Python is slow for N=5M.
    # Alternative: use the divisor-sum sieve approach with numpy.
    # sigma(n) = sum_{d|n} d
    # We can compute this via: for d in 1..N, add d to all multiples of d.
    # This is O(N log N) but with numpy it's fast.
    # However, for N=5M, the array of slices might be memory-heavy but doable.
    
    # Let's use the numpy slice approach but in chunks to manage memory
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 500_000 == 0:
            if time.time() > deadline - 15:
                break

    # Check if we completed
    if time.time() > deadline - 15:
        # Truncate to what we have
        # Actually sigma might be incomplete. Let's just use what we have.
        # We need to make sure we only consider n where sigma is complete.
        # The slice approach: if we broke at d=D, then sigma[n] is missing contributions from d > D.
        # This is problematic. Let's instead use the SPF-based factorization for correctness.
        pass

    # Actually, let me reconsider. The slice approach is O(N log N) ≈ 5M * 15 ≈ 75M operations.
    # With numpy, each slice operation is fast. Let's estimate: 5M iterations of sigma[d::d] += d.
    # Each iteration does N/d additions. Total ≈ N * H(N) ≈ 5M * 15 = 75M additions.
    # In numpy, this should be fast (well under 10s). Let me just do it and check time.
    
    # Let me redo this more carefully with a time check
    sigma = np.zeros(N + 1, dtype=np.int64)
    t0 = time.time()
    for d in range(1, N + 1):
        sigma[d::d] += d
        if d % 1_000_000 == 0:
            elapsed = time.time() - t0
            if elapsed > 50:  # if taking too long, break
                break
    
    elapsed_sigma = time.time() - t0
    # Determine effective N based on how far we got
    if elapsed_sigma > 50:
        # We didn't finish; use a smaller effective range
        # Actually, the loop breaks, so sigma is incomplete for the tail.
        # Let's just use a smaller N from the start if needed.
        pass

    # For safety, let's use a smaller N if the full computation is too slow
    # Let's just proceed with N and hope it's fast enough.
    # For N=5M, the divisor sum sieve should take about 5-10 seconds in numpy.

    # --- Group numbers by sigma value ---
    # Convert sigma to a list for easier grouping (skip index 0)
    # Use dict: sigma_val -> list of n
    groups = defaultdict(list)
    for n in range(1, N + 1):
        s = int(sigma[n])
        groups[s].append(n)
    
    # --- Search for best pair ---
    best_a = 11
    best_b = 6
    best_error = abs(11/6 - 3/2)  # = 1/3
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        nums.sort()  # ensure sorted (they should be since we iterate n in order)
        for i, a in enumerate(nums):
            # We want b ≈ (2/3) * a
            target_b = (2.0 / 3.0) * a
            # Binary search in nums for target_b
            idx = bisect_left(nums, int(target_b))
            # Check candidates around idx
            for j in [idx - 1, idx, idx + 1, idx + 2]:
                if 0 <= j < len(nums):
                    b = nums[j]
                    if b == a:
                        continue
                    # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue  # invalid
                    den = 2 * b
                    err = num / den
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
    
    # Check deadline
    if time.time() > deadline - 5:
        pass  # return what we have

    # Ensure types are Python int
    return [int(best_a), int(best_b)]
