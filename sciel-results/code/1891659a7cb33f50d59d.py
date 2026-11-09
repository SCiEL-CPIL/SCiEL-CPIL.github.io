"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    limit = 10000

    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    if n < 2:
        return primes

    # DP: dp[j] = list of length j, where dp[j][i] = length of longest
    # non-decreasing-gap sequence ending at primes[i], primes[j] (i < j).
    # parent[j][i] = index k < i such that dp[j][i] = dp[i][k] + 1
    # (i.e., the predecessor of i in the sequence ending at j).

    dp = [None] * n
    parent = [None] * n

    # Initialize: sequences of length 2
    for j in range(1, n):
        dp[j] = [2] * j
        parent[j] = [-1] * j  # no parent for length-2 sequences

    # Process: for each i from 1 to n-2, use dp[i] to update dp[j] for j > i
    # dp[j][i] = 1 + max{ dp[i][k] : k < i and primes[i]-primes[k] <= primes[j]-primes[i] }
    #          = 1 + max{ dp[i][k] : k < i and primes[k] >= primes[j] - 2*primes[i] + primes[i] }
    # Wait, let me re-derive:
    # Gap constraint: (primes[i] - primes[k]) <= (primes[j] - primes[i])
    # => primes[k] >= 2*primes[i] - primes[j]
    # So for fixed i and j, we need max of dp[i][k] over k in [0, i-1] with primes[k] >= 2*primes[i] - primes[j]

    # Since primes is sorted, the condition primes[k] >= threshold means k >= some index.
    # We can precompute for each i: a suffix-max array of dp[i][k] over k.

    best_len = 2
    best_i = 0
    best_j = 1

    for i in range(1, n - 1):
        if time.time() > deadline:
            break

        # Build suffix max of dp[i][k] for k = 0..i-1
        # suffix_max[k] = max(dp[i][k], dp[i][k+1], ..., dp[i][i-1])
        dpi = dp[i]
        suffix_max = [0] * i
        suffix_max[i - 1] = dpi[i - 1]
        for k in range(i - 2, -1, -1):
            suffix_max[k] = max(dpi[k], suffix_max[k + 1])

        # For each j > i, compute dp[j][i]
        for j in range(i + 1, n):
            if time.time() > deadline:
                break

            # threshold: primes[k] >= 2*primes[i] - primes[j]
            threshold = 2 * primes[i] - primes[j]

            # Find the smallest k in [0, i-1] such that primes[k] >= threshold
            # Since primes is sorted, use binary search
            # We need k such that primes[k] >= threshold
            # lo = 0, hi = i - 1
            lo, hi = 0, i - 1
            # Find leftmost k with primes[k] >= threshold
            while lo < hi:
                mid = (lo + hi) // 2
                if primes[mid] >= threshold:
                    hi = mid
                else:
                    lo = mid + 1

            if primes[lo] >= threshold:
                # Max dp[i][k] for k in [lo, i-1] is suffix_max[lo]
                new_len = 1 + suffix_max[lo]
                # Find which k achieves this max
                # We need the k in [lo, i-1] with dp[i][k] == suffix_max[lo]
                # For reconstruction, we need to track the argmax
                # Let's store parent as the k that gives the max
                # To avoid O(i) search each time, we can precompute argmax suffix
                # But for now, let's just find it
                max_val = suffix_max[lo]
                # Find rightmost k in [lo, i-1] with dp[i][k] == max_val
                # Actually any k works for correctness, but for reconstruction we need consistency
                # Let's find the first k from lo that has dp[i][k] == max_val
                k_parent = lo
                while k_parent < i - 1 and dpi[k_parent] != max_val:
                    k_parent += 1
                # k_parent is in [lo, i-1] and dpi[k_parent] == max_val

                if new_len > (dp[j][i] if dp[j] is not None else 0):
                    # dp[j] might not be initialized yet for this i
                    # Actually dp[j] is initialized as [2]*j, so dp[j][i] exists
                    if new_len > dp[j][i]:
                        dp[j][i] = new_len
                        parent[j][i] = k_parent
                        if new_len > best_len:
                            best_len = new_len
                            best_i = i
                            best_j = j

    # Reconstruct the sequence
    if best_len <= 2:
        # Fallback: greedy
        values = [primes[0]]
        for p in primes[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

    # Backtrack from (best_i, best_j)
    seq = [primes[best_j], primes[best_i]]
    i = best_i
    j = best_j
    while parent[j] is not None and parent[j][i] != -1:
        k = parent[j][i]
        seq.append(primes[k])
        j = i
        i = k

    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
