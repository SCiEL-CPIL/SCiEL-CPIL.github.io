"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    weights = [L // (i + 1) for i in range(n)]
    
    mid = n // 2
    half1 = weights[:mid]
    half2 = weights[mid:]
    
    def enumerate_sums(half_weights):
        sums = np.array([0], dtype=np.int64)
        for w in half_weights:
            new_sums = np.concatenate([sums - w, sums, sums + w])
            sums = new_sums
        return sums
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums1 = enumerate_sums(half1)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums2 = enumerate_sums(half2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    sums1 = np.sort(sums1)
    sums2 = np.sort(sums2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    best_abs = None
    best_s1 = None
    best_s2 = None
    
    i_ptr = 0
    j_ptr = len(sums2) - 1
    n1 = len(sums1)
    n2 = len(sums2)
    
    while i_ptr < n1 and j_ptr >= 0:
        if time.time() > deadline:
            break
        
        total = int(sums1[i_ptr]) + int(sums2[j_ptr])
        abs_total = abs(total)
        
        if total != 0:
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                best_s1 = int(sums1[i_ptr])
                best_s2 = int(sums2[j_ptr])
                if best_abs == 1:
                    break
        
        if total < 0:
            i_ptr += 1
        elif total > 0:
            j_ptr -= 1
        else:
            i_ptr += 1
            j_ptr -= 1
    
    if time.time() > deadline and best_abs is None:
        return [0] * 28 + [1, -1]
    
    if best_abs is None:
        return [0] * 28 + [1, -1]
    
    def decode_sum(half_weights, target_sum):
        n_h = len(half_weights)
        
        @lru_cache(maxsize=None)
        def can_make(pos, remaining):
            if pos == n_h:
                return remaining == 0
            w = half_weights[pos]
            for c in (1, 0, -1):
                if can_make(pos + 1, remaining - c * w):
                    return True
            return False
        
        if not can_make(0, target_sum):
            return None
        
        coeffs = [0] * n_h
        remaining = target_sum
        for pos in range(n_h):
            w = half_weights[pos]
            for c in (1, 0, -1):
                if can_make(pos + 1, remaining - c * w):
                    coeffs[pos] = c
                    remaining -= c * w
                    break
        
        can_make.cache_clear()
        return coeffs
    
    coeffs1 = decode_sum(half1, best_s1)
    coeffs2 = decode_sum(half2, best_s2)
    
    if coeffs1 is None or coeffs2 is None:
        return [0] * 28 + [1, -1]
    
    result = [int(c) for c in (coeffs1 + coeffs2)]
    
    def total_value(coeffs):
        t = 0
        for idx in range(n):
            t += coeffs[idx] * weights[idx]
        return t
    
    current_total = total_value(result)
    current_abs = abs(current_total)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(n):
            if time.time() > deadline - 5:
                break
            old_val = result[idx]
            for new_val in (-1, 0, 1):
                if new_val == old_val:
                    continue
                result[idx] = new_val
                new_total = total_value(result)
                new_abs = abs(new_total)
                if new_total != 0 and new_abs < current_abs:
                    current_abs = new_abs
                    current_total = new_total
                    improved = True
                    break
                else:
                    result[idx] = old_val
    
    final_total = total_value(result)
    if final_total == 0:
        return [0] * 28 + [1, -1]
    
    return [int(c) for c in result]

# EVOLVE-BLOCK-END
