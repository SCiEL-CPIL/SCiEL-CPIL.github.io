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
    fallback = [11, 6] # sigma(11)=12, sigma(6)=12. 11/6 = 1.833. |1.833 - 1.5| = 0.333.
    
    try:
        # 1. SPF Sieve
        spf = np.arange(N + 1, dtype=np.int32)
        # Only need to loop to sqrt(N)
        limit_spf = int(np.sqrt(N)) + 1
        for i in range(2, limit_spf):
            if spf[i] == i:
                # Vectorized update
                start = i * i
                indices = np.arange(start, N + 1, i)
                mask = spf[indices] == indices
                spf[indices[mask]] = i
                
        if time_left() < 15.0:
            return fallback

        # 2. Compute Sigma using DP
        sigma = np.ones(N + 1, dtype=np.int64)
        core = np.arange(N + 1, dtype=np.int32)
        p_pow = np.ones(N + 1, dtype=np.int32)
        sigma_p_pow = np.ones(N + 1, dtype=np.int64)
        
        sigma[1] = 1
        core[1] = 1
        p_pow[1] = 1
        sigma_p_pow[1] = 1

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

        if time_left() < 5.0:
            return fallback

        # 3. Group by Sigma
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
        
        # Iterate groups
        for gi in range(starts.size):
            if time_left() < 2.0:
                break
                
            s0 = int(starts[gi])
            e0 = int(ends[gi])
            length = e0 - s0
            if length < 2:
                continue
                
            group_nums = np.sort(idx[order[s0:e0]])
            L = len(group_nums)
            
            local_best_val = float('inf')
            local_best_a = 0
            local_best_b = 0
            
            # For each b, find a near 1.5*b
            # Use a window of candidates
            WINDOW = 5
            
            for bi in range(L):
                b_val = group_nums[bi]
                target = 1.5 * b_val
                
                # Find position in group_nums
                pos = np.searchsorted(group_nums, target, side='left')
                
                # Check candidates in a window around pos
                lo = max(0, pos - WINDOW)
                hi = min(L, pos + WINDOW + 1)
                
                for ai in range(lo, hi):
                    a_val = group_nums[ai]
                    if a_val == b_val:
                        continue
                    # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                    num = abs(2 * a_val - 3 * b_val)
                    if num == 0:
                        continue
                    val = num / (2 * b_val)
                    if val < local_best_val:
                        local_best_val = val
                        local_best_a = a_val
                        local_best_b = b_val
                
                if time_left() < 2.0:
                    break
            
            if local_best_val < best_val:
                best_val = local_best_val
                best_pair = (int(local_best_a), int(local_best_b))

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
