import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, range(1, 31))
    
    weights = np.array([LCM // (i + 1) for i in range(N)], dtype=np.int64)
    
    half = 15
    w1 = weights[:half]
    w2 = weights[half:]
    
    def enumerate_sums(w):
        n = len(w)
        total = 3 ** n
        sums = np.zeros(total, dtype=np.int64)
        for j in range(n):
            period = 3 ** (j + 1)
            block = 3 ** j
            pattern = np.concatenate([
                np.full(block, -1, dtype=np.int64),
                np.zeros(block, dtype=np.int64),
                np.full(block, 1, dtype=np.int64)
            ])
            repeats = total // period
            sums += w[j] * np.tile(pattern, repeats)
        return sums
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums1 = enumerate_sums(w1)
    sums2 = enumerate_sums(w2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    order2 = np.argsort(sums2)
    sums2_sorted = sums2[order2]
    
    best_abs = np.iinfo(np.int64).max
    best_i1 = -1
    best_i2 = -1
    
    n1 = len(sums1)
    batch_size = 200000
    offsets = np.array([-3, -2, -1, 0, 1, 2, 3])
    
    for start in range(0, n1, batch_size):
        if time.time() > deadline - 5:
            break
        end = min(start + batch_size, n1)
        s1b = sums1[start:end]
        neg = -s1b
        
        pos = np.searchsorted(sums2_sorted, neg)
        cands = pos[:, None] + offsets[None, :]
        cands = np.clip(cands, 0, len(sums2_sorted) - 1)
        
        c2 = sums2_sorted[cands]
        totals = s1b[:, None] + c2
        abs_t = np.abs(totals)
        abs_t[totals == 0] = np.iinfo(np.int64).max
        
        mi = np.argmin(abs_t, axis=1)
        mv = abs_t[np.arange(len(s1b)), mi]
        mo = cands[np.arange(len(s1b)), mi]
        
        imp = mv < best_abs
        if np.any(imp):
            ii = np.where(imp)[0]
            for k in ii:
                best_abs = mv[k]
                best_i1 = start + k
                best_i2 = mo[k]
    
    if best_i1 == -1:
        return [0] * 28 + [1, -1]
    
    def idx_to_coeffs(idx, n):
        c = []
        for j in range(n):
            d = idx % 3
            idx //= 3
            c.append(d - 1)
        return c
    
    c1 = idx_to_coeffs(best_i1, half)
    c2 = idx_to_coeffs(best_i2, N - half)
    result = c1 + c2
    
    total = sum(result[i] / (i + 1) for i in range(N))
    if total == 0:
        return [0] * 28 + [1, -1]
    
    return result
