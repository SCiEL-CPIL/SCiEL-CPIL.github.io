import numpy as np
import time
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def gen_sums_vec(w):
        k = len(w)
        sums = np.array([0], dtype=np.int64)
        idxs = np.array([0], dtype=np.int64)
        for i, weight in enumerate(w):
            pow3 = 3 ** i
            new_sums = np.concatenate([
                sums - weight,
                sums,
                sums + weight
            ])
            new_idxs = np.concatenate([
                idxs,
                idxs + pow3,
                idxs + 2 * pow3
            ])
            sums = new_sums
            idxs = new_idxs
        return sums, idxs
    
    # Generate first half
    sums1, idxs1 = gen_sums_vec(w1)
    
    # Sort first half by sum value
    sorted_order = np.argsort(sums1)
    sorted_sums1 = sums1[sorted_order]
    sorted_idxs1 = idxs1[sorted_order]
    
    # Generate second half
    sums2, idxs2 = gen_sums_vec(w2)
    
    best_abs = np.iinfo(np.int64).max
    best_idx1 = -1
    best_idx2 = -1
    
    # Vectorized search
    # For each s2, we want to find s1 such that s1 + s2 is close to 0 but not 0.
    # targets = -sums2
    targets = -sums2
    
    # Find insertion points
    positions = np.searchsorted(sorted_sums1, targets)
    
    # We need to check positions, positions-1, positions+1
    # Create arrays for the 3 candidates
    
    # Candidate 1: positions
    pos1 = np.clip(positions, 0, len(sorted_sums1) - 1)
    s1_cand1 = sorted_sums1[pos1]
    idx1_cand1 = sorted_idxs1[pos1]
    total1 = s1_cand1 + sums2
    abs1 = np.abs(total1)
    mask1 = total1 != 0
    abs1 = np.where(mask1, abs1, np.iinfo(np.int64).max)
    
    # Candidate 2: positions - 1
    pos2 = np.clip(positions - 1, 0, len(sorted_sums1) - 1)
    s1_cand2 = sorted_sums1[pos2]
    idx1_cand2 = sorted_idxs1[pos2]
    total2 = s1_cand2 + sums2
    abs2 = np.abs(total2)
    mask2 = total2 != 0
    abs2 = np.where(mask2, abs2, np.iinfo(np.int64).max)
    
    # Candidate 3: positions + 1
    pos3 = np.clip(positions + 1, 0, len(sorted_sums1) - 1)
    s1_cand3 = sorted_sums1[pos3]
    idx1_cand3 = sorted_idxs1[pos3]
    total3 = s1_cand3 + sums2
    abs3 = np.abs(total3)
    mask3 = total3 != 0
    abs3 = np.where(mask3, abs3, np.iinfo(np.int64).max)
    
    # Combine
    all_abs = np.minimum(np.minimum(abs1, abs2), abs3)
    
    # Determine which candidate gave the min
    # We need to reconstruct idx1 and idx2
    
    # Create a structure to pick the right idx1
    # If abs1 is min, pick idx1_cand1
    # If abs2 is min, pick idx1_cand2
    # If abs3 is min, pick idx1_cand3
    
    # To handle ties, we can just pick the first one that is min.
    # Or simpler: create a 3D array? No, memory.
    # We can compare step by step.
    
    min_abs_12 = np.minimum(abs1, abs2)
    use_1_vs_2 = abs1 <= abs2
    
    idx1_temp = np.where(use_1_vs_2, idx1_cand1, idx1_cand2)
    
    min_abs_all = np.minimum(min_abs_12, abs3)
    
    # If abs3 is strictly less than min_abs_12, use idx1_cand3
    # Else use idx1_temp
    use_3 = abs3 < min_abs_12
    all_idx1 = np.where(use_3, idx1_cand3, idx1_temp)
    
    all_idx2 = np.arange(len(sums2))
    
    min_pos = np.argmin(all_abs)
    best_abs_val = all_abs[min_pos]
    
    if best_abs_val < best_abs:
        best_abs = best_abs_val
        best_idx1 = all_idx1[min_pos]
        best_idx2 = all_idx2[min_pos]
    
    if best_idx1 == -1 or best_abs == np.iinfo(np.int64).max:
        # Fallback
        return [0]*28 + [1, -1]
    
    def decode(idx, k):
        coeffs = []
        rem = int(idx)
        for j in range(k):
            digit = rem % 3
            rem //= 3
            if digit == 0:
                coeffs.append(-1)
            elif digit == 1:
                coeffs.append(0)
            else:
                coeffs.append(1)
        return coeffs
    
    c1 = decode(best_idx1, half)
    c2 = decode(best_idx2, n - half)
    best_coeffs = c1 + c2
    
    # Local search refinement
    if time.time() < deadline - 10:
        # Convert to list for easier manipulation
        best_coeffs = [int(c) for c in best_coeffs]
        
        def compute_sum(coeffs):
            s = 0
            for i, c in enumerate(coeffs):
                if c != 0:
                    s += Fraction(c, i + 1)
            return s
        
        current_sum = compute_sum(best_coeffs)
        current_abs = abs(current_sum)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() >= deadline - 5:
                    break
                for new_val in [-1, 0, 1]:
                    if new_val == best_coeffs[i]:
                        continue
                    old_val = best_coeffs[i]
                    best_coeffs[i] = new_val
                    new_sum = current_sum + Fraction(new_val - old_val, i + 1)
                    if new_sum != 0:
                        new_abs = abs(new_sum)
                        if new_abs < current_abs:
                            current_sum = new_sum
                            current_abs = new_abs
                            improved = True
                    else:
                        # Sum is zero, revert
                        best_coeffs[i] = old_val
                        continue
                    # Keep the change
            # If improved, restart the loop to try more changes
            if not improved:
                break
    
    return [int(c) for c in best_coeffs]
