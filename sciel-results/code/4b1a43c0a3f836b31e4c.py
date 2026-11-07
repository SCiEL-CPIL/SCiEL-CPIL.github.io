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
        p_power = [0] * (N + 1)  # p^(cnt[i]) where p = spf[i]

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

        # ---------- Group by Sigma ----------
        order = np.argsort(sigma_arr, kind='stable')
        sorted_sigma = sigma_arr[order]
        sorted_indices = order + 1  # convert 0-based to 1-based

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

            gn = sorted_indices[s0:e0]
            gn_sorted = np.sort(gn)
            L = len(gn_sorted)

            # Vectorized search: for each b in the group, find a closest to 1.5*b
            gn_f = gn_sorted.astype(np.float64)
            targets = 1.5 * gn_f
            idx = np.searchsorted(gn_sorted, targets)

            # Check element at idx (above target) and idx-1 (below target)
            idx_hi = np.minimum(idx, L - 1)
            idx_lo = np.maximum(idx - 1, 0)

            a_hi = gn_sorted[idx_hi]
            a_lo = gn_sorted[idx_lo]
            b_arr = gn_sorted

            # Exclude a == b
            mask_hi = a_hi != b_arr
            mask_lo = a_lo != b_arr

            if np.any(mask_hi):
                a_h = a_hi[mask_hi].astype(np.float64)
                b_h = b_arr[mask_hi].astype(np.float64)
                num_h = np.abs(2.0 * a_h - 3.0 * b_h)
                # Exclude exact zero (2a == 3b)
                nonzero = num_h > 0
                if np.any(nonzero):
                    val_h = num_h[nonzero] / (2.0 * b_h[nonzero])
                    min_i = np.argmin(val_h)
                    if val_h[min_i] < best_val:
                        # Map back to original masked positions
                        pos_in_mask = np.where(nonzero)[0][min_i]
                        pos_in_full = np.where(mask_hi)[0][pos_in_mask]
                        best_val = val_h[min_i]
                        best_pair = (int(a_h[pos_in_mask]), int(b_h[pos_in_mask]))

            if np.any(mask_lo):
                a_l = a_lo[mask_lo].astype(np.float64)
                b_l = b_arr[mask_lo].astype(np.float64)
                num_l = np.abs(2.0 * a_l - 3.0 * b_l)
                nonzero = num_l > 0
                if np.any(nonzero):
                    val_l = num_l[nonzero] / (2.0 * b_l[nonzero])
                    min_i = np.argmin(val_l)
                    if val_l[min_i] < best_val:
                        pos_in_mask = np.where(nonzero)[0][min_i]
                        pos_in_full = np.where(mask_lo)[0][pos_in_mask]
                        best_val = val_l[min_i]
                        best_pair = (int(a_l[pos_in_mask]), int(b_l[pos_in_mask]))

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
