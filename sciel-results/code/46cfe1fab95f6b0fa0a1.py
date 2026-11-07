import time
import numpy as np
import bisect

def construct():
    t0 = time.time()
    LIMIT = 60.0
    SAFETY = 0.85
    hard_stop = t0 + LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    N = 10000000
    fallback = [11, 6]

    try:
        # 1. SPF Sieve - optimized
        spf = np.zeros(N + 1, dtype=np.int32)
        spf[1] = 1
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                if i * i <= N:
                    for j in range(i * i, N + 1, i):
                        if spf[j] == 0:
                            spf[j] = i

        if time_left() < 15.0:
            return fallback

        # 2. Compute Sigma
        sigma = np.ones(N + 1, dtype=np.int64)
        core = np.zeros(N + 1, dtype=np.int32)
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

        if time_left() < 5.0:
            return fallback

        # 3. Group by Sigma using Dictionary
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
                
            gn = np.sort(idx[order[s0:e0]])
            gn_list = gn.tolist()
            L = len(gn_list)
            
            # Two-pointer approach with early termination
            i_a = 0
            for bi in range(L):
                b = gn_list[bi]
                target = 1.5 * b
                
                # Advance i_a to first element >= target
                while i_a < L and gn_list[i_a] < target:
                    i_a += 1
                    
                # Check i_a
                if i_a < L:
                    a = gn_list[i_a]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                
                # Check i_a - 1
                if i_a > 0:
                    a = gn_list[i_a - 1]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                
                # Early termination: if b is large enough, further b values won't improve
                # since |a/b - 1.5| >= 1/(2b) for any valid a
                if 1.0 / (2 * b) >= best_val:
                    break

        if best_pair is None:
            return fallback
            
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= N and 1 <= b <= N):
            return fallback
        if 2 * a == 3 * b:
            return fallback
            
        # 4. High-precision polish: local search around best candidate
        if time_left() > 3.0:
            # Find the sigma value of the best pair
            sigma_best = int(_sigma[a])
            
            # Get all numbers with this sigma
            mask = _sigma[1:] == sigma_best
            candidates = np.nonzero(mask)[0] + 1
            candidates_list = candidates.tolist()
            L = len(candidates_list)
            
            if L >= 2:
                # Do a more thorough local search
                for bi in range(L):
                    b2 = candidates_list[bi]
                    target = 1.5 * b2
                    
                    # Binary search for insertion point
                    lo, hi = 0, L
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if candidates_list[mid] < target:
                            lo = mid + 1
                        else:
                            hi = mid
                    
                    # Check lo and lo-1
                    for a2 in [candidates_list[lo], candidates_list[lo - 1]] if lo > 0 and lo < L else [candidates_list[lo]] if lo < L else [candidates_list[lo - 1]]:
                        if a2 != b2:
                            num = abs(2 * a2 - 3 * b2)
                            if num > 0:
                                val = num / (2 * b2)
                                if val < best_val:
                                    best_val = val
                                    a, b = a2, b2
                            
                            if time_left() < 1.0:
                                break
                    if time_left() < 1.0:
                        break
        
        return [a, b]

    except Exception:
        return [11, 6]
