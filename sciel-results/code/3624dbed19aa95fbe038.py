import math
import time
import bisect

def construct():
    """
    Returns exactly 30 integer coefficients from {-1, 0, 1} such that
    sum(coeff[i-1]/i for i in 1..30) is nonzero and has minimal absolute value.
    Uses Meet-in-the-Middle with compact index encoding.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 30
    half = 15
    
    def lcm(a, b):
        return a // math.gcd(a, b) * b
        
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Scaled weights
    weights = [L // (i + 1) for i in range(n)]
    
    def gen_sums_dict(start, length):
        """
        Generates dict: sum -> index (base-3 encoded coeffs)
        """
        d = {0: 0}
        pow3 = 1
        for pos in range(length):
            w = weights[start + pos]
            new_d = {}
            off0 = 0
            off1 = pow3
            off2 = 2 * pow3
            
            for s, idx in d.items():
                # coeff -1 (digit 0)
                s_new = s - w
                if s_new not in new_d:
                    new_d[s_new] = idx + off0
                    
                # coeff 0 (digit 1)
                s_new = s
                if s_new not in new_d:
                    new_d[s_new] = idx + off1
                    
                # coeff 1 (digit 2)
                s_new = s + w
                if s_new not in new_d:
                    new_d[s_new] = idx + off2
                    
            d = new_d
            pow3 *= 3
            if time.time() > deadline:
                return None
        return d

    d1 = gen_sums_dict(0, half)
    if d1 is None:
        return [0] * 28 + [1, -1]
        
    d2 = gen_sums_dict(half, half)
    if d2 is None:
        return [0] * 28 + [1, -1]

    keys2 = sorted(d2.keys())
    
    best_abs = None
    best_idx1 = None
    best_idx2 = None
    
    for s1, idx1 in d1.items():
        target = -s1
        idx = bisect.bisect_left(keys2, target)
        
        candidates = []
        if idx > 0: candidates.append(idx - 1)
        if idx < len(keys2): candidates.append(idx)
        if idx + 1 < len(keys2): candidates.append(idx + 1)
        
        for k in candidates:
            s2 = keys2[k]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_idx1 = idx1
                    best_idx2 = d2[s2]
                    
        if best_abs == 1:
            break
            
        if time.time() > deadline:
            break
            
    if best_idx1 is None:
        return [0] * 28 + [1, -1]
        
    coeffs = [0] * n
    temp = best_idx1
    for j in range(half):
        d = temp % 3
        coeffs[j] = d - 1
        temp //= 3
        
    temp = best_idx2
    for j in range(half):
        d = temp % 3
        coeffs[half + j] = d - 1
        temp //= 3
        
    return coeffs

# EVOLVE-BLOCK-END
