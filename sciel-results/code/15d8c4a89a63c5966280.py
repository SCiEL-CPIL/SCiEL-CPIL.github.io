import time
import math
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # Weights w_i = L / (i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    n1 = n // 2  # 15
    n2 = n - n1  # 15
    
    def generate_half(n_vars, start_idx):
        sums = np.array([0], dtype=np.int64)
        idxs = np.array([0], dtype=np.int64)
        
        for i in range(n_vars):
            w = weights[start_idx + i]
            len_old = len(sums)
            
            s_new = np.empty(len_old * 3, dtype=np.int64)
            i_new = np.empty(len_old * 3, dtype=np.int64)
            
            s_new[0::3] = sums
            s_new[1::3] = sums + w
            s_new[2::3] = sums - w
            
            i_new[0::3] = idxs
            i_new[1::3] = idxs + 3**i
            i_new[2::3] = idxs + 2 * 3**i
            
            sums = s_new
            idxs = i_new
            
        return sums, idxs
    
    sums1, idx1 = generate_half(n1, 0)
    sums2, idx2 = generate_half(n2, n1)
    
    if time.time() > deadline - 15:
        return [0] * 28 + [1, -1]
    
    sort_idx = np.argsort(sums2, kind='stable')
    sums2_sorted = sums2[sort_idx]
    idx2_sorted = idx2[sort_idx]
    
    targets = -sums1
    positions = np.searchsorted(sums2_sorted, targets)
    
    pos_left = np.clip(positions - 1, 0, len(sums2_sorted) - 1)
    pos_right = np.clip(positions, 0, len(sums2_sorted) - 1)
    
    s2_left = sums2_sorted[pos_left]
    s2_right = sums2_sorted[pos_right]
    
    total_left = sums1 + s2_left
    total_right = sums1 + s2_right
    
    abs_left = np.abs(total_left)
    abs_right = np.abs(total_right)
    
    max_int = np.iinfo(np.int64).max
    abs_left[total_left == 0] = max_int
    abs_right[total_right == 0] = max_int
    
    min_abs_left = np.min(abs_left)
    min_abs_right = np.min(abs_right)
    
    if min_abs_left < min_abs_right:
        best_idx1 = np.argmin(abs_left)
        best_idx2 = pos_left[best_idx1]
    else:
        best_idx1 = np.argmin(abs_right)
        best_idx2 = pos_right[best_idx1]
    
    coeffs = [0] * n
    
    c = int(idx1[best_idx1])
    for i in range(n1):
        digit = c % 3
        c //= 3
        if digit == 1:
            coeffs[i] = 1
        elif digit == 2:
            coeffs[i] = -1
    
    c = int(idx2_sorted[best_idx2])
    for i in range(n2):
        digit = c % 3
        c //= 3
        if digit == 1:
            coeffs[n1 + i] = 1
        elif digit == 2:
            coeffs[n1 + i] = -1
    
    exact_sum = Fraction(0, 1)
    for i in range(n):
        if coeffs[i] != 0:
            exact_sum += Fraction(coeffs[i], i + 1)
    
    if exact_sum == 0:
        return [0] * 28 + [1, -1]
    
    # Local search: try flipping single coefficients
    best_abs = abs(exact_sum)
    best_coeffs = coeffs[:]
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            old_val = coeffs[i]
            for new_val in (-1, 0, 1):
                if new_val == old_val:
                    continue
                new_sum = exact_sum + Fraction(new_val - old_val, i + 1)
                if new_sum != 0 and abs(new_sum) < best_abs:
                    best_abs = abs(new_sum)
                    best_coeffs = coeffs[:]
                    best_coeffs[i] = new_val
                    coeffs[i] = new_val
                    exact_sum = new_sum
                    improved = True
    
    # Try flipping pairs
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(n):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 3:
                        break
                    old_i, old_j = coeffs[i], coeffs[j]
                    for new_i in (-1, 0, 1):
                        for new_j in (-1, 0, 1):
                            if new_i == old_i and new_j == old_j:
                                continue
                            new_sum = exact_sum + Fraction(new_i - old_i, i + 1) + Fraction(new_j - old_j, j + 1)
                            if new_sum != 0 and abs(new_sum) < best_abs:
                                best_abs = abs(new_sum)
                                best_coeffs = coeffs[:]
                                best_coeffs[i] = new_i
                                best_coeffs[j] = new_j
                                coeffs[i] = new_i
                                coeffs[j] = new_j
                                exact_sum = new_sum
                                improved = True
    
    return best_coeffs
