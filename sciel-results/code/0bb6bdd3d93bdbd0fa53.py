import time
import math
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 90.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = reduce(lcm, denominators, 1)
    
    W = [L // d for d in denominators]
    
    n_a = 23
    n_b = n - n_a
    
    size_a = 1 << n_a
    size_b = 1 << n_b
    
    # Generate sums for A
    s_a = [0] * size_a
    for mask in range(1, size_a):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        s_a[mask] = s_a[prev] + W[idx]
    
    # Generate sums for B
    s_b = [0] * size_b
    for mask in range(1, size_b):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        s_b[mask] = s_b[prev] + W[n_a + idx]
    
    # Sort both with masks
    pairs_a = list(zip(s_a, range(size_a)))
    pairs_a.sort()
    
    pairs_b = list(zip(s_b, range(size_b)))
    pairs_b.sort()
    
    # Two-pointer: for each sum_a, find max sum_b such that sum_a + sum_b < L
    best_deficit_num = L
    best_mask_a = 0
    best_mask_b = 0
    found = False
    
    # Pointer for B, starts at end
    ptr_b = size_b - 1
    
    for sum_a, mask_a in pairs_a:
        if time.time() > deadline - 5.0:
            break
        
        # Move ptr_b down while sum_a + s_b[ptr_b] >= L
        while ptr_b >= 0 and sum_a + pairs_b[ptr_b][0] >= L:
            ptr_b -= 1
        
        if ptr_b < 0:
            break
        
        # pairs_b[ptr_b] is the largest sum_b such that sum_a + sum_b < L
        sum_b = pairs_b[ptr_b][0]
        mask_b = pairs_b[ptr_b][1]
        
        # Ensure nonempty result
        if mask_a == 0 and mask_b == 0:
            continue
        
        deficit = L - (sum_a + sum_b)
        if deficit < best_deficit_num:
            best_deficit_num = deficit
            best_mask_a = mask_a
            best_mask_b = mask_b
            found = True
    
    if not found:
        return [0]
    
    result = []
    for i in range(n_a):
        if best_mask_a & (1 << i):
            result.append(i)
    
    for i in range(n_b):
        if best_mask_b & (1 << i):
            result.append(n_a + i)
    
    return result
