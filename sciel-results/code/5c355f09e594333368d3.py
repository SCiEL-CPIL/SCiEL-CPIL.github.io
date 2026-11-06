"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import bisect
from itertools import product
from fractions import Fraction

def _lcm_range(a, b):
    result = 1
    for i in range(a, b + 1):
        result = result * i // math.gcd(result, i)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    N = 30
    mid = 15
    
    D1 = _lcm_range(1, mid)
    D2 = _lcm_range(mid + 1, N)
    
    weights1 = [D1 // i for i in range(1, mid + 1)]
    weights2 = [D2 // i for i in range(mid + 1, N + 1)]
    
    # Enumerate first half
    sums1 = []
    coeffs1_map = {}
    for coeffs in product([-1, 0, 1], repeat=mid):
        s = sum(c * w for c, w in zip(coeffs, weights1))
        sums1.append(s)
        if s not in coeffs1_map:
            coeffs1_map[s] = coeffs
    
    # Enumerate second half
    sums2 = []
    coeffs2_list = []
    for coeffs in product([-1, 0, 1], repeat=N - mid):
        s = sum(c * w for c, w in zip(coeffs, weights2))
        sums2.append(s)
        coeffs2_list.append(coeffs)
    
    # Sort unique sums1
    unique_sums1 = sorted(set(sums1))
    
    best_val = float('inf')
    best_coeffs = None
    
    for idx, (n2, c2) in enumerate(zip(sums2, coeffs2_list)):
        if time.time() > deadline:
            break
        
        target = -n2 * D1 / D2
        
        # Find insertion point
        pos = bisect.bisect_left(unique_sums1, target)
        
        # Check candidates around pos
        for offset in range(-5, 6):
            j = pos + offset
            if 0 <= j < len(unique_sums1):
                n1 = unique_sums1[j]
                total_num = n1 * D2 + n2 * D1
                if total_num == 0:
                    continue
                abs_val = abs(total_num)
                if abs_val < best_val:
                    best_val = abs_val
                    c1 = coeffs1_map[n1]
                    best_coeffs = list(c1) + list(c2)
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    # Verify
    total = Fraction(0, 1)
    for i, c in enumerate(best_coeffs, 1):
        total += Fraction(c, i)
    if total == 0:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
