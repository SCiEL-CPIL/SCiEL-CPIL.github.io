import time
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin out of 120s

    # Seed solution: M = 333312, k in {336, 372, 434}
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # We want to find M <= 10^18 with many distinct k_i (k_i <= sqrt(M))
    # such that k_i * sigma(k_i) = M.

    # Strategy: Compute sigma(k) for k up to some K_max using a sieve.
    # Then compute M_vals = k * sigma(k) and find collisions (same M from different k).
    # The constraint k <= sqrt(M) is automatically satisfied when M = k*sigma(k)
    # and sigma(k) >= 1, since k <= k*sigma(k) = M implies k <= M, and
    # for k >= 1, k <= sqrt(M) iff k^2 <= M = k*sigma(k) iff k <= sigma(k).
    # Since sigma(k) >= 1 + k for k >= 2 (divisors include 1 and k), we have sigma(k) > k
    # for k >= 2, so k < sigma(k), hence k^2 < k*sigma(k) = M, so k < sqrt(M).
    # For k = 1: sigma(1) = 1, M = 1, sqrt(1) = 1, so k = 1 <= sqrt(M) holds.

    # So the constraint k_i <= sqrt(M) is automatically satisfied for all valid pairs.

    # Let's try to go as high as we can in K.
    # Memory: sigma array is K+1 int64 = 8*(K+1) bytes.
    # M_vals is K int64 = 8*K bytes.
    # Total ~ 16*K bytes. For K = 50M, that's 800MB.
    # For K = 100M, that's 1.6GB. Might be tight.

    # Let's start with K = 10M and go up.

    K_current = 10_000_000
    max_K = 100_000_000

    while K_current <= max_K:
        if time.time() > deadline - 15:
            break

        try:
            # Compute sigma(1..K) using sieve
            sigma = np.zeros(K_current + 1, dtype=np.int64)

            for j in range(1, K_current + 1):
                if time.time() > deadline - 10:
                    raise TimeoutError()
                # sigma[j], sigma[2j], sigma[3j], ... += j
                sigma[j::j] += j

            # Compute M = k * sigma(k) for k = 1..K
            k_vals = np.arange(1, K_current + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:]

            # Filter M <= 10^18
            MAX_M = 10**18
            valid_mask = M_vals <= MAX_M
            if not np.any(valid_mask):
                K_current = int(K_current * 1.5)
                continue

            M_valid = M_vals[valid_mask]
            k_valid = k_vals[valid_mask]

            n_valid = len(M_valid)
            if n_valid <= best_count:
                K_current = int(K_current * 1.5)
                continue

            # Sort by M to find collisions
            sort_indices = np.lexsort((k_valid, M_valid))
            M_sorted = M_valid[sort_indices]
            k_sorted = k_valid[sort_indices]

            if n_valid > 1:
                diff_M = np.diff(M_sorted)
                change_indices = np.where(diff_M != 0)[0]

                # Build run lengths
                starts = np.concatenate(([0], change_indices + 1))
                ends = np.concatenate((change_indices, [n_valid - 1]))
                lengths = ends - starts + 1

                max_len = int(np.max(lengths))

                if max_len > best_count:
                    max_idx = int(np.argmax(lengths))
                    start_idx = int(starts[max_idx])
                    end_idx = int(ends[max_idx])

                    best_M = int(M_sorted[start_idx])
                    best_k = [int(x) for x in k_sorted[start_idx:end_idx + 1]]
                    best_count = max_len

        except Exception:
            break

        K_current = int(K_current * 1.5)

    # Final validation: ensure all k_i satisfy k_i * sigma(k_i) = M
    # and k_i <= sqrt(M), and they are distinct.
    # We trust our computation, but let's do a quick sanity check on the best.
    if best_count > 0:
        # Verify distinctness
        assert len(set(best_k)) == len(best_k), "Duplicate k values found"

        # Verify each k satisfies k*sigma(k) = M
        for k_val in best_k:
            # Compute sigma(k_val) exactly
            s = 0
            n = k_val
            for d in range(1, int(math.isqrt(n)) + 1):
                if n % d == 0:
                    s += d
                    other = n // d
                    if other != d:
                        s += other
            if k_val * s != best_M:
                # Should not happen, but if it does, remove this k
                best_k.remove(k_val)
                best_count -= 1
            if k_val > math.isqrt(best_M):
                best_k.remove(k_val)
                best_count -= 1

    # Ensure types are correct
    return {'M': int(best_M), 'k': list(best_k)}
