"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import bisect
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Weights: w[i] = L // (i+1) for i in 0..29
    # We want to minimize |sum(c_i * w_i)| != 0
    # c_i in {-1, 0, 1}
    
    # Strategy: 2-Way MITM with 15/15 split
    # 3^15 = 14,348,907 sums per half
    # Use numpy for efficient generation and search
    
    def generate_sums_numpy(indices):
        """
        Generates all sums for the given indices using numpy.
        Returns a numpy array of sums.
        """
        sums = np.array([0], dtype=np.int64)
        for idx in indices:
            val = L // idx
            # Create three copies and shift
            new_sums = np.concatenate([sums - val, sums, sums + val])
            sums = new_sums
        return sums

    def generate_sums_with_coeffs(indices):
        """
        Generates all sums for the given indices.
        Returns a list of (sum, coeffs_tuple).
        This is only used for small sets (<= 10 terms).
        """
        sums = [(0, ())]
        for idx in indices:
            val = L // idx
            new_sums = []
            for s, c in sums:
                # -1
                new_sums.append((s - val, c + (-1,)))
                # 0
                new_sums.append((s, c + (0,)))
                # 1
                new_sums.append((s + val, c + (1,)))
            sums = new_sums
        return sums

    # Split indices 1..30 into two halves of 15
    half1 = list(range(1, 16))
    half2 = list(range(16, 31))
    
    # Generate sums for half 1
    sums1 = generate_sums_numpy(half1)
    
    if time.time() > deadline - 30:
        # Fallback if generation is too slow (unlikely)
        return [0] * 28 + [1, -1]
        
    # Sort sums1
    sums1_sorted = np.sort(sums1)
    
    # Generate sums for half 2
    sums2 = generate_sums_numpy(half2)
    
    if time.time() > deadline - 15:
        return [0] * 28 + [1, -1]
    
    # Find best pair (s1, s2) such that |s1 + s2| is minimized and non-zero
    # Use numpy searchsorted for efficiency
    best_found_val = np.iinfo(np.int64).max
    best_s1 = None
    best_s2 = None
    
    # Process in chunks to manage memory and allow time checks
    chunk_size = 1000000
    total_len = len(sums2)
    
    for start in range(0, total_len, chunk_size):
        if time.time() > deadline - 10:
            break
            
        end = min(start + chunk_size, total_len)
        chunk = sums2[start:end]
        
        # For each s2 in chunk, find best s1
        targets = -chunk
        pos = np.searchsorted(sums1_sorted, targets)
        
        # Check pos and pos-1
        # Clip pos to valid range
        pos_clipped = np.clip(pos, 0, len(sums1_sorted) - 1)
        
        # Get candidates at pos and pos-1
        candidates_pos = sums1_sorted[pos_clipped]
        candidates_pos_minus = sums1_sorted[np.clip(pos_clipped - 1, 0, len(sums1_sorted) - 1)]
        
        # Compute totals for both candidates
        totals_pos = candidates_pos + chunk
        totals_pos_minus = candidates_pos_minus + chunk
        
        # Find non-zero totals with minimum absolute value
        # Create a mask for non-zero
        mask_pos = totals_pos != 0
        mask_pos_minus = totals_pos_minus != 0
        
        # Combine both candidate sets
        # We need to find the minimum absolute value among non-zero totals
        abs_pos = np.abs(totals_pos[mask_pos]) if np.any(mask_pos) else np.array([], dtype=np.int64)
        abs_pos_minus = np.abs(totals_pos_minus[mask_pos_minus]) if np.any(mask_pos_minus) else np.array([], dtype=np.int64)
        
        if len(abs_pos) > 0 or len(abs_pos_minus) > 0:
            # Find the minimum
            min_val = np.iinfo(np.int64).max
            min_idx = -1
            min_type = -1  # 0 for pos, 1 for pos_minus
            
            if len(abs_pos) > 0:
                idx_in_pos = np.argmin(abs_pos)
                if abs_pos[idx_in_pos] < min_val:
                    min_val = abs_pos[idx_in_pos]
                    min_idx = idx_in_pos
                    min_type = 0
            
            if len(abs_pos_minus) > 0:
                idx_in_pos_minus = np.argmin(abs_pos_minus)
                if abs_pos_minus[idx_in_pos_minus] < min_val:
                    min_val = abs_pos_minus[idx_in_pos_minus]
                    min_idx = idx_in_pos_minus
                    min_type = 1
            
            if min_val < best_found_val:
                best_found_val = min_val
                
                if min_type == 0:
                    # Find the actual index in the original chunk
                    # We need to map back from the masked array
                    # This is tricky; let's do it differently
                    # Actually, let's just iterate through the chunk for the best candidate
                    pass
                
                # Simplified: just track the best and reconstruct later
                # For now, let's find the specific s1 and s2
                # We'll do a second pass to find the exact values
                pass
    
    # Alternative: simpler but still fast approach
    # Iterate over sums2 and use bisect
    best_found_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    sums1_list = sums1_sorted.tolist()
    sums2_list = sums2.tolist()
    
    for i, s2 in enumerate(sums2_list):
        if time.time() > deadline - 8:
            break
        if i % 1000000 == 0 and i > 0:
            # Time check every million iterations
            pass
            
        target = -s2
        pos = bisect.bisect_left(sums1_list, target)
        
        # Check pos and pos-1
        if pos < len(sums1_list):
            s1_candidate = sums1_list[pos]
            total = s1_candidate + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_found_val:
                    best_found_val = abs_total
                    best_s1 = s1_candidate
                    best_s2 = s2
                    if best_found_val == 1:
                        break
        
        if pos > 0:
            s1_candidate = sums1_list[pos - 1]
            total = s1_candidate + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_found_val:
                    best_found_val = abs_total
                    best_s1 = s1_candidate
                    best_s2 = s2
                    if best_found_val == 1:
                        break
    
    if best_s1 is None:
        # Should not happen
        return [0] * 28 + [1, -1]

    # Now reconstruct coefficients for best_s1 and best_s2
    # We use the "small MITM" trick for each half.
    
    def find_coeffs_for_target(target, indices):
        """
        Finds coefficients for the given indices that sum to target.
        Uses MITM on the indices themselves.
        """
        mid = len(indices) // 2
        left_idx = indices[:mid]
        right_idx = indices[mid:]
        
        # Generate sums with coeffs for left and right
        # Since len(indices) <= 15, mid <= 8.
        # 3^8 = 6561. This is small.
        left_sums = generate_sums_with_coeffs(left_idx)
        right_sums = generate_sums_with_coeffs(right_idx)
        
        # Create a dict for right sums: {sum: coeffs}
        # If multiple coeffs give the same sum, we just keep one.
        right_dict = {}
        for s, c in right_sums:
            if s not in right_dict:
                right_dict[s] = c
                
        # Search for match
        for s_l, c_l in left_sums:
            needed = target - s_l
            if needed in right_dict:
                c_r = right_dict[needed]
                return c_l + c_r
                
        return None

    coeffs1 = find_coeffs_for_target(best_s1, half1)
    if coeffs1 is None:
        # Error in logic?
        return [0] * 28 + [1, -1]
        
    coeffs2 = find_coeffs_for_target(best_s2, half2)
    if coeffs2 is None:
        return [0] * 28 + [1, -1]
        
    final_coeffs = list(coeffs1) + list(coeffs2)
    
    # Local search polish: try to improve by flipping single coefficients
    # Compute current sum
    current_sum = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
    current_abs = abs(current_sum)
    
    # Try flipping each coefficient
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            # Try changing coefficient i
            old_coeff = final_coeffs[i]
            val = L // (i + 1)
            
            for new_coeff in [-1, 0, 1]:
                if new_coeff == old_coeff:
                    continue
                
                new_sum = current_sum + (new_coeff - old_coeff) * val
                if new_sum == 0:
                    continue
                
                new_abs = abs(new_sum)
                if new_abs < current_abs:
                    final_coeffs[i] = new_coeff
                    current_sum = new_sum
                    current_abs = new_abs
                    improved = True
                    break
    
    # Verify
    total = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
    if total == 0:
        # This should not happen as we excluded 0
        # But just in case, flip last coeff
        if final_coeffs[-1] == 0:
            final_coeffs[-1] = 1
        else:
            final_coeffs[-1] = -final_coeffs[-1]
            
    return final_coeffs

# EVOLVE-BLOCK-END
