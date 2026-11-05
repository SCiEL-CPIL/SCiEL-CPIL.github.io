import time
import random
from math import gcd
from functools import reduce
import numpy as np

def construct():
    deadline = time.time() + 85
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // (i + 1) for i in range(30)]
    
    best_abs = None
    best_coeffs = None
    
    def update_best(coeffs, int_sum):
        nonlocal best_abs, best_coeffs
        if int_sum == 0:
            return
        a = abs(int_sum)
        if best_abs is None or a < best_abs:
            best_abs = a
            best_coeffs = list(coeffs)
    
    half = 15
    idx1 = list(range(half))
    idx2 = list(range(half, 30))
    
    # Precompute powers of 3
    pow3 = [1]
    for i in range(1, 16):
        pow3.append(pow3[-1] * 3)
    
    # Generate all (sum, code) for half1
    def gen_half(indices):
        results = [(0, 0)]
        for pos, idx in enumerate(indices):
            w = weights[idx]
            p = pow3[pos]
            new_results = []
            for s, code in results:
                new_results.append((s, code))
                new_results.append((s + w, code + p))
                new_results.append((s - w, code + 2*p))
            results = new_results
        return results
    
    t0 = time.time()
    half1_data = gen_half(idx1)
    half1_data.sort(key=lambda x: x[0])
    sums1 = np.array([x[0] for x in half1_data], dtype=np.int64)
    codes1 = np.array([x[1] for x in half1_data], dtype=np.int64)
    print(f"Half1: {time.time()-t0:.1f}s")
    
    # Generate half2 sums and codes
    t0 = time.time()
    half2_data = gen_half(idx2)
    print(f"Half2: {time.time()-t0:.1f}s")
    
    # For each entry in half2, find best match in half1
    # Vectorize using numpy
    s2_arr = np.array([x[0] for x in half2_data], dtype=np.int64)
    c2_arr = np.array([x[1] for x in half2_data], dtype=np.int64)
    
    # For each s2, target = -s2
    # Find position in sums1
    targets = -s2_arr
    positions = np.searchsorted(sums1, targets)
    
    # Check positions-1 and positions
    best_total = None
    best_p1 = None
    best_p2 = None
    
    for i in range(len(s2_arr)):
        if time.time() > deadline:
            break
        pos = positions[i]
        for p in [pos-1, pos]:
            if 0 <= p < len(sums1):
                total = sums1[p] + s2_arr[i]
                if total == 0:
                    continue
                a = abs(total)
                if best_total is None or a < best_total:
                    best_total = a
                    best_p1 = p
                    best_p2 = i
    
    if best_total is not None:
        def decode(code, n):
            coeffs = []
            for pos in range(n):
                d = code % 3
                coeffs.append(d - 1)
                code //= 3
            return coeffs
        # Wait, my encoding: digit 0->0, 1->1, 2->-1
        # So d=0 -> 0, d=1 -> 1, d=2 -> -1
        # d-1 gives: 0->-1, 1->0, 2->1. That's wrong.
        # Let me fix: 
        def decode(code, n):
            coeffs = []
            for pos in range(n):
                d = code % 3
                if d == 0:
                    coeffs.append(0)
                elif d == 1:
                    coeffs.append(1)
                else:
                    coeffs.append(-1)
                code //= 3
            return coeffs
        
        c1 = decode(codes1[best_p1], half)
        c2 = decode(c2_arr[best_p2], half)
        return c1 + c2
    
    return [0]*28 + [1, -1]
