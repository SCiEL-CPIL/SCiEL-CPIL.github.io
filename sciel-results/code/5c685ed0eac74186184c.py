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
        # We can use the full limit because the linear sieve is fast.
        N = MAXV_LIMIT
        
        # ---------- Linear Sieve for Sigma ----------
        spf = np.zeros(N + 1, dtype=np.int32)
        sigma = np.zeros(N + 1, dtype=np.int64)
        cnt = np.zeros(N + 1, dtype=np.int32)
        sigma_pow = np.zeros(N + 1, dtype=np.int64)
        
        sigma[1] = 1
        sigma_pow[1] = 1
        cnt[1] = 0
        
        primes = []
        
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                sigma[i] = 1 + i
                sigma_pow[i] = 1 + i
                cnt[i] = 1
            
            for p in primes:
                if p > spf[i] or i * p > N:
                    break
                spf[i * p] = p
                if p == spf[i]:
                    # p divides i
                    cnt[i * p] = cnt[i] + 1
                    # sigma_pow[i*p] = sigma_pow[i] + p^(cnt[i]+1)
                    # We need p^(cnt[i]+1).
                    # sigma_pow[i] = sum_{k=0}^{cnt[i]} p^k
                    # sigma_pow[i*p] = sum_{k=0}^{cnt[i]+1} p^k = sigma_pow[i] + p^(cnt[i]+1)
                    # We can compute p^(cnt[i]+1) if we store it or calculate it.
                    # Storing p_power is easier.
                    # Let's add p_power array.
                    # Actually, sigma[i*p] = sigma[i // p^cnt[i]] * sigma_pow[i*p]
                    # Let m = i // p^cnt[i]
                    # sigma[i] = sigma[m] * sigma_pow[i]
                    # So sigma[m] = sigma[i] // sigma_pow[i]
                    # Then sigma[i*p] = (sigma[i] // sigma_pow[i]) * sigma_pow[i*p]
                    
                    # To avoid division and potential precision issues (though int is exact),
                    # we can store `sigma_base[i]` = sigma[i // p^cnt[i]]
                    # Then sigma[i*p] = sigma_base[i] * sigma_pow[i*p]
                    pass # Placeholder for logic refinement below
                else:
                    # p does not divide i (p > spf[i] is handled by break, so here p < spf[i] is not possible in standard linear sieve? 
                    # In linear sieve, we iterate p up to spf[i]. 
                    # If p < spf[i], then gcd(i, p) = 1.
                    cnt[i * p] = 1
                    sigma_pow[i * p] = 1 + p
                    # sigma[i*p] = sigma[i] * (1 + p)
                    pass

        # The logic above is getting complicated to implement correctly in NumPy vectorized style inside the loop.
        # The Python loop for linear sieve is O(N) but with a small constant.
        # For N=10^7, it might take ~5-10s in Python.
        # Let's implement the Python loop version of linear sigma sieve.
        
        del spf, sigma, cnt, sigma_pow
        spf = [0] * (N + 1)
        sigma = [0] * (N + 1)
        cnt = [0] * (N + 1)
        sigma_pow = [0] * (N + 1)
        sigma_base = [0] * (N + 1) # sigma[i // p^cnt[i]]
        
        sigma[1] = 1
        sigma_pow[1] = 1
        cnt[1] = 0
        sigma_base[1] = 1
        
        primes = []
        
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                sigma[i] = 1 + i
                sigma_pow[i] = 1 + i
                cnt[i] = 1
                sigma_base[i] = 1
                # sigma[i] = sigma_base[i] * sigma_pow[i] = 1 * (1+i)
            
            for p in primes:
                if i * p > N or p > spf[i]:
                    break
                spf[i * p] = p
                if p == spf[i]:
                    cnt[i * p] = cnt[i] + 1
                    # sigma_pow[i*p] = sigma_pow[i] + p^(cnt[i]+1)
                    # We need p^(cnt[i]+1).
                    # Note that sigma_pow[i] = 1 + p + ... + p^(cnt[i])
                    # So p^(cnt[i]+1) is not directly stored.
                    # However, sigma_pow[i*p] = sigma_pow[i] + p * (p^cnt[i])
                    # This is getting messy without storing p_power.
                    
                    # Let's use the formula:
                    # sigma[i*p] = sigma_base[i] * (sigma_pow[i] + p^(cnt[i]+1))
                    # We can store p_power[i] = p^cnt[i]
                    pass
                else:
                    cnt[i * p] = 1
                    sigma_pow[i * p] = 1 + p
                    sigma_base[i * p] = sigma[i]
                    # sigma[i*p] = sigma_base[i*p] * sigma_pow[i*p]
                    pass
            
            # This loop is too slow in Python for 10^7 if not optimized.
            # Let's stick to the harmonic sieve but with a smaller N, or use a C-optimized library if available?
            # No, standard library only.
            
            # Let's go back to the harmonic sieve but with N = 2,000,000.
            # It's safe and fast.
            break # Break out of the linear sieve attempt to use harmonic sieve below
        
        # ---------- Harmonic Sieve with Smaller N ----------
        N = 2000000
        sigma = np.zeros(N + 1, dtype=np.int32)
        
        for d in range(1, N + 1):
            sigma[d::d] += d
            
        # ---------- Group by Sigma ----------
        sigma_vals = sigma[1:]
        order = np.argsort(sigma_vals, kind='stable')
        sorted_sigma = sigma_vals[order]
        sorted_indices = order + 1
        
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
            gn = np.sort(gn)
            L = len(gn)
            
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
                    a = int(gn[j-1])
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
