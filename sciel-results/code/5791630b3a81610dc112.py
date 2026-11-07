import time
import bisect
import numpy as np

MAXV_LIMIT = 10000000
WALL_LIMIT = 90.0
SAFETY = 0.85

def construct():
    t0 = time.time()
    hard_stop = t0 + WALL_LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    try:
        N = MAXV_LIMIT

        # ---------- Linear Sieve for Sigma (O(N)) ----------
        spf = [0] * (N + 1)
        sigma = [0] * (N + 1)
        cnt = [0] * (N + 1)
        sigma_pow = [0] * (N + 1)
        p_power = [0] * (N + 1)

        sigma[1] = 1
        sigma_pow[1] = 1
        cnt[1] = 0
        p_power[1] = 1

        primes = []

        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                sigma[i] = 1 + i
                sigma_pow[i] = 1 + i
                cnt[i] = 1
                p_power[i] = i

            for p in primes:
                ip = i * p
                if ip > N or p > spf[i]:
                    break
                spf[ip] = p
                if p == spf[i]:
                    cnt[ip] = cnt[i] + 1
                    p_power[ip] = p_power[i] * p
                    sigma_pow[ip] = sigma_pow[i] + p_power[ip]
                    sigma[ip] = sigma[i] // sigma_pow[i] * sigma_pow[ip]
                else:
                    cnt[ip] = 1
                    p_power[ip] = p
                    sigma_pow[ip] = 1 + p
                    sigma[ip] = sigma[i] * (1 + p)

        # ---------- Convert to numpy array for fast sorting ----------
        sigma_arr = np.array(sigma[1:], dtype=np.int64)
        indices = np.arange(1, N + 1, dtype=np.int64)

        # Sort by (sigma, index) so groups are contiguous AND sorted by index
        order = np.lexsort((indices, sigma_arr))
        sorted_sigma = sigma_arr[order]
        sorted_indices = indices[order]

        # Find group boundaries
        changes = np.empty(len(sorted_sigma), dtype=bool)
        changes[0] = True
        changes[1:] = sorted_sigma[1:] != sorted_sigma[:-1]
        starts = np.nonzero(changes)[0]
        ends = np.append(starts[1:], len(sorted_sigma))

        best_val = float('inf')
        best_pair = None

        for gi in range(len(starts)):
            if time_left() < 5.0:
                break

            s0 = starts[gi]
            e0 = ends[gi]
            length = e0 - s0
            if length < 2:
                continue

            # sorted_indices is already sorted by value within each group
            gn = sorted_indices[s0:e0]
            L = len(gn)

            # For each b in the group, find a that minimizes |a/b - 3/2|
            # i.e., minimize |2a - 3b| / (2b)
            # a should be close to 3b/2

            for bi in range(L):
                b = int(gn[bi])
                target = 1.5 * b
                j = bisect.bisect_left(gn, target)

                if j < L:
                    a = int(gn[j])
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)

                if j > 0:
                    a = int(gn[j - 1])
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)

        if best_pair is None:
            return [11, 6]

        a, b = best_pair
        if not (1 <= a <= N and 1 <= b <= N):
            return [11, 6]
        if sigma[a] != sigma[b]:
            return [11, 6]
        if 2 * a == 3 * b:
            return [11, 6]

        return [int(a), int(b)]

    except Exception:
        return [11, 6]
