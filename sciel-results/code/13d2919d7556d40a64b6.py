import time
import numpy as np

def construct():
    t0 = time.time()
    LIMIT = 60.0
    SAFETY = 0.85
    hard_stop = t0 + LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    N = 10000000
    fallback = [11, 6]  # sigma(11)=12, sigma(6)=12. |11/6 - 3/2| = 0.3333

    try:
        # 1. SPF Sieve (vectorized)
        spf = np.arange(N + 1, dtype=np.int32)
        limit_spf = int(np.sqrt(N)) + 1
        for i in range(2, limit_spf):
            if spf[i] == i:
                start = i * i
                indices = np.arange(start, N + 1, i)
                mask = spf[indices] == indices
                spf[indices[mask]] = i

        if time_left() < 25.0:
            return fallback

        # 2. Compute Sigma using SPF-based DP
        # sigma[n] = sigma(core[n]) * sigma_p_pow[n]
        # where core[n] = n with all factors of spf[n] removed
        # p_pow[n] = highest power of spf[n] dividing n
        # sigma_p_pow[n] = sigma(p_pow[n]) = (p_pow[n]*p - 1) // (p - 1)
        
        sigma = np.ones(N + 1, dtype=np.int64)
        core = np.ones(N + 1, dtype=np.int32)
        p_pow = np.ones(N + 1, dtype=np.int32)
        sigma_p_pow = np.ones(N + 1, dtype=np.int64)
        
        _sigma = sigma
        _core = core
        _p_pow = p_pow
        _sigma_p_pow = sigma_p_pow
        _spf = spf
        
        for n in range(2, N + 1):
            p = _spf[n]
            m = n // p
            if m % p == 0:
                _core[n] = _core[m]
                _p_pow[n] = _p_pow[m] * p
                _sigma_p_pow[n] = _sigma_p_pow[m] + _p_pow[n]
            else:
                _core[n] = m
                _p_pow[n] = p
                _sigma_p_pow[n] = 1 + p
            _sigma[n] = _sigma[_core[n]] * _sigma_p_pow[n]
            
            if (n & 0x3FFFF) == 0 and time_left() < 10.0:
                return fallback

        if time_left() < 10.0:
            return fallback

        # 3. Group by Sigma value and find best pair
        idx = np.arange(1, N + 1, dtype=np.int32)
        vals = _sigma[1:]
        
        order = np.argsort(vals, kind='stable')
        svals = vals[order]
        
        M = order.size
        change = np.empty(M, dtype=bool)
        change[0] = True
        change[1:] = svals[1:] != svals[:-1]
        starts = np.nonzero(change)[0]
        ends = np.append(starts[1:], M)
        
        best_val = float('inf')
        best_pair = None
        
        for gi in range(starts.size):
            if time_left() < 3.0:
                break
                
            s0 = int(starts[gi])
            e0 = int(ends[gi])
            length = e0 - s0
            if length < 2:
                continue
                
            group_nums = np.sort(idx[order[s0:e0]])
            L = len(group_nums)
            
            targets = 1.5 * group_nums.astype(np.float64)
            j_left = np.searchsorted(group_nums, targets, side='left')
            
            valid_left = j_left < L
            if np.any(valid_left):
                a_cands = group_nums[j_left[valid_left]]
                b_cands = group_nums[valid_left]
                num = np.abs(2 * a_cands - 3 * b_cands)
                mask_valid = num > 0
                if np.any(mask_valid):
                    a_v = a_cands[mask_valid]
                    b_v = b_cands[mask_valid]
                    num_v = num[mask_valid]
                    den_v = 2 * b_v
                    val = num_v / den_v
                    min_idx = np.argmin(val)
                    if val[min_idx] < best_val:
                        best_val = val[min_idx]
                        best_pair = (int(a_v[min_idx]), int(b_v[min_idx]))
                        
            valid_right = j_left > 0
            if np.any(valid_right):
                j_right = j_left[valid_right] - 1
                a_cands = group_nums[j_right]
                b_cands = group_nums[valid_right]
                num = np.abs(2 * a_cands - 3 * b_cands)
                mask_valid = (num > 0) & (a_cands != b_cands)
                if np.any(mask_valid):
                    a_v = a_cands[mask_valid]
                    b_v = b_cands[mask_valid]
                    num_v = num[mask_valid]
                    den_v = 2 * b_v
                    val = num_v / den_v
                    min_idx = np.argmin(val)
                    if val[min_idx] < best_val:
                        best_val = val[min_idx]
                        best_pair = (int(a_v[min_idx]), int(b_v[min_idx]))

        if best_pair is None:
            return fallback
            
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= N and 1 <= b <= N):
            return fallback
        if 2 * a == 3 * b:
            return fallback
            
        return [a, b]

    except Exception:
        return [11, 6]
