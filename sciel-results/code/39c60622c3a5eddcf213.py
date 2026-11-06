import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    start_time = time.time()
    timeout = 55.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    values = [L // d for d in denominators]
    
    mid = n // 2
    left_vals = values[:mid]
    right_vals = values[mid:]
    left_n = len(left_vals)
    right_n = len(right_vals)
    
    # Generate left subset sums
    left_sums = [0] * (1 << left_n)
    for mask in range(1, 1 << left_n):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        left_sums[mask] = left_sums[mask ^ lsb] + left_vals[idx]
    
    # Generate right subset sums
    right_sums = [0] * (1 << right_n)
    for mask in range(1, 1 << right_n):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        right_sums[mask] = right_sums[mask ^ lsb] + right_vals[idx]
    
    # Build and sort right data
    right_data = [(right_sums[m], m) for m in range(1 << right_n) if right_sums[m] < L]
    right_data.sort(key=lambda x: x[0])
    right_sums_only = [x[0] for x in right_data]
    
    best_total = 0
    best_lm = 0
    best_rm = 0
    
    for mask in range(1 << left_n):
        ls = left_sums[mask]
        if ls >= L:
            continue
        max_rs = L - ls - 1
        if max_rs < 0:
            continue
        idx = bisect_left(right_sums_only, max_rs + 1)
        if idx > 0:
            rs = right_sums_only[idx - 1]
            rm = right_data[idx - 1][1]
            total = ls + rs
            if total > best_total:
                best_total = total
                best_lm = mask
                best_rm = rm
    
    result = []
    for i in range(left_n):
        if best_lm & (1 << i):
            result.append(i)
    for i in range(right_n):
        if best_rm & (1 << i):
            result.append(mid + i)
    
    if not result:
        result = [0]
    
    return result
