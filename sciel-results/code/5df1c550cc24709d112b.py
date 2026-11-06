import time
import random
from math import gcd
import bisect
import numpy as np

def construct():
    start_time = time.time()
    # Deadline with safety margin
    deadline = start_time + 55.0
    
    n = 30
    
    # Precompute LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    
    # W[i] = L // (i+1) for i in 0..29
    # Sum c[i]/(i+1) = (sum c[i] * W[i]) / L
    W = [L // (i + 1) for i in range(n)]
    W_np = np.array(W, dtype=np.int64)
    
    best_abs_K = float('inf')
    best_coeffs = None
    
    mid = n // 2  # 15
    
    def generate_half(start, end):
        """Generate all (sum, mask) pairs for indices start to end-1."""
        length = end - start
        sums = [0]
        masks = [0]
        
        for i in range(length):
            w = W[start + i]
            new_sums = []
            new_masks = []
            multiplier = 3 ** i
            
            for k in range(len(sums)):
                s = sums[k]
                m = masks[k]
                
                # c = -1
                new_sums.append(s - w)
                new_masks.append(m)
                
                # c = 0
                new_sums.append(s)
                new_masks.append(m + multiplier)
                
                # c = 1
                new_sums.append(s + w)
                new_masks.append(m + 2 * multiplier)
                
            sums = new_sums
            masks = new_masks
        
        return sums, masks

    # Generate Half A (indices 0 to 14)
    sums_a, masks_a = generate_half(0, mid)
    
    # Generate Half B (indices 15 to 29)
    sums_b, masks_b = generate_half(mid, n)
    
    # Sort Half B by sum
    pairs_b = list(zip(sums_b, masks_b))
    pairs_b.sort(key=lambda x: x[0])
    
    arr_b_sums = [p[0] for p in pairs_b]
    arr_b_masks = [p[1] for p in pairs_b]
    
    # Precompute powers of 3 for fast decoding
    pow3 = [3 ** j for j in range(n)]
    
    # Helper to decode mask to coeffs (optimized)
    def decode_mask(mask, offset, length):
        coeffs = [0] * length
        for j in range(length):
            digit = (mask // pow3[j]) % 3
            coeffs[j] = digit - 1
        return coeffs

    # Search for minimum non-zero |sum_a + sum_b|
    for i in range(len(sums_a)):
        if time.time() > deadline:
            break
            
        a_sum = sums_a[i]
        target = -a_sum
        
        # Find insertion point
        idx = bisect.bisect_left(arr_b_sums, target)
        
        # Check candidates around idx
        candidates = set()
        
        # Check immediate neighbors
        if idx < len(arr_b_sums):
            candidates.add(idx)
        if idx > 0:
            candidates.add(idx - 1)
            
        # If the closest is exactly target, we need to go further
        if idx < len(arr_b_sums) and arr_b_sums[idx] == target:
            # Find next > target
            j = idx
            while j < len(arr_b_sums) and arr_b_sums[j] == target:
                j += 1
            if j < len(arr_b_sums):
                candidates.add(j)
                
            # Find prev < target
            j = idx - 1
            while j >= 0 and arr_b_sums[j] == target:
                j -= 1
            if j >= 0:
                candidates.add(j)
        
        for c_idx in candidates:
            if 0 <= c_idx < len(arr_b_sums):
                b_sum = arr_b_sums[c_idx]
                total = a_sum + b_sum
                if total != 0:
                    abs_total = abs(total)
                    if abs_total < best_abs_K:
                        best_abs_K = abs_total
                        # Decode
                        coeffs_a = decode_mask(masks_a[i], 0, mid)
                        coeffs_b = decode_mask(arr_b_masks[c_idx], 0, mid)
                        best_coeffs = coeffs_a + coeffs_b

    if best_coeffs is None:
        # Fallback
        best_coeffs = [1] + [-1] + [0]*28
        best_abs_K = abs(W[0] - W[1])

    # Local Search / Polish - more aggressive
    random.seed(42)
    current = list(best_coeffs)
    current_K = sum(current[i] * W[i] for i in range(n))
    
    # Multiple passes with different orderings
    for pass_num in range(20):
        if time.time() > deadline:
            break
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                    
                old_c = current[i]
                
                # Try all three values for position i
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    
                    new_K = current_K + (new_c - old_c) * W[i]
                    
                    if new_K == 0:
                        continue
                    
                    new_abs_K = abs(new_K)
                    if new_abs_K < best_abs_K:
                        current[i] = new_c
                        current_K = new_K
                        best_abs_K = new_abs_K
                        best_coeffs = list(current)
                        improved = True
                        old_c = new_c
                        break  # Move to next index

    # Additional random restarts with more aggressive perturbation
    if time.time() < deadline - 3:
        for restart in range(300):
            if time.time() > deadline:
                break
            
            current = list(best_coeffs)
            current_K = sum(current[i] * W[i] for i in range(n))
            
            # Perturb more aggressively
            num_perturb = random.randint(2, 8)
            for _ in range(num_perturb):
                idx = random.randint(0, n - 1)
                current[idx] = random.choice([-1, 0, 1])
            
            current_K = sum(current[i] * W[i] for i in range(n))
            if current_K == 0:
                continue
            
            # Local search with multiple passes
            for pass_num in range(3):
                if time.time() > deadline:
                    break
                improved_local = True
                while improved_local and time.time() < deadline:
                    improved_local = False
                    indices = list(range(n))
                    random.shuffle(indices)
                    
                    for i in indices:
                        if time.time() > deadline:
                            break
                            
                        old_c = current[i]
                        
                        for new_c in [-1, 0, 1]:
                            if new_c == old_c:
                                continue
                            
                            new_K = current_K + (new_c - old_c) * W[i]
                            
                            if new_K == 0:
                                continue
                            
                            new_abs_K = abs(new_K)
                            if new_abs_K < best_abs_K:
                                current[i] = new_c
                                current_K = new_K
                                best_abs_K = new_abs_K
                                best_coeffs = list(current)
                                improved_local = True
                                old_c = new_c
                                break

    return best_coeffs
