import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # 90s limit, keep 5s margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // (i + 1) for i in range(30)]
    
    n1 = 15
    n2 = 15
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    def gen_and_compute(w, n):
        total = 3 ** n
        codes = np.arange(total, dtype=np.int64)
        sums = np.zeros(total, dtype=np.int64)
        for i in range(n):
            power = 3 ** i
            digits = (codes // power) % 3
            vals = digits - 1  # 0->-1, 1->0, 2->1
            sums += vals * w[i]
        return codes, sums
    
    if time.time() > deadline - 5:
        return [0] * 28 + [1, -1]
    
    codes1, sums1 = gen_and_compute(w1, n1)
    
    if time.time() > deadline - 30:
        return [0] * 28 + [1, -1]
    
    codes2, sums2 = gen_and_compute(w2, n2)
    
    if time.time() > deadline - 20:
        return [0] * 28 + [1, -1]
    
    # Sort sums2
    order2 = np.argsort(sums2)
    sums2_sorted = sums2[order2]
    codes2_sorted = codes2[order2]
    
    best_diff = None
    best_c1 = None
    best_c2 = None
    
    # For each sum in first half, find best match in second half
    for i in range(len(sums1)):
        if time.time() > deadline - 5:
            break
        s1 = sums1[i]
        target = -s1
        
        idx = np.searchsorted(sums2_sorted, target)
        
        for offset in range(-5, 6):
            j = idx + offset
            if 0 <= j < len(sums2_sorted):
                s2 = sums2_sorted[j]
                total = s1 + s2
                if total == 0:
                    continue
                if best_diff is None or abs(total) < abs(best_diff):
                    best_diff = total
                    best_c1 = codes1[i]
                    best_c2 = codes2_sorted[j]
                    if best_diff == 1 or best_diff == -1:
                        break
        if best_diff == 1 or best_diff == -1:
            break
    
    if best_diff is None:
        return [0] * 28 + [1, -1]
    
    # Decode coefficients
    coeffs = [0] * 30
    v = best_c1
    for i in range(n1):
        digit = v % 3
        v //= 3
        coeffs[i] = digit - 1
    
    v = best_c2
    for i in range(n2):
        digit = v % 3
        v //= 3
        coeffs[n1 + i] = digit - 1
    
    return coeffs
