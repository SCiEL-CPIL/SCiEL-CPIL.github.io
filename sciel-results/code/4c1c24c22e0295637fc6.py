"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
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

    def generate_sums_only(indices):
        """
        Generates all sums for the given indices.
        Returns a list of sums.
        Used for larger sets (15 terms).
        """
        sums = [0]
        for idx in indices:
            val = L // idx
            # Pre-allocate list to avoid multiple concatenations
            new_sums = [0] * (len(sums) * 3)
            base = 0
            for s in sums:
                new_sums[base] = s - val
                new_sums[base + 1] = s
                new_sums[base + 2] = s + val
                base += 3
            sums = new_sums
        return sums

    # Split indices 1..30 into two halves of 15
    half1 = list(range(1, 16))
    half2 = list(range(16, 31))
    
    # Generate sums for half 1
    sums1 = generate_sums_only(half1)
    
    if time.time() > deadline - 20:
        # Fallback if generation is too slow (unlikely)
        return [0] * 28 + [1, -1]
        
    sums1.sort()
    
    # Generate sums for half 2
    sums2 = generate_sums_only(half2)
    
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]
        
    # Find best pair (s1, s2) such that |s1 + s2| is minimized and non-zero
    best_found_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    len_sums1 = len(sums1)
    # Check time less frequently to reduce overhead
    check_interval = 1000000
    iteration_count = 0
    
    for s2 in sums2:
        iteration_count += 1
        if iteration_count % check_interval == 0 and time.time() > deadline - 5:
            break
            
        target = -s2
        pos = bisect.bisect_left(sums1, target)
        
        # Check pos and pos-1
        if pos < len_sums1:
            s1 = sums1[pos]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_found_val:
                    best_found_val = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_found_val == 1:
                        break
        
        if pos > 0:
            s1 = sums1[pos-1]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_found_val:
                    best_found_val = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_found_val == 1:
                        break
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
    
    # Verify
    total = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
    if total == 0:
        # This should not happen as we excluded 0
        # But just in case, flip last coeff
        if final_coeffs[-1] == 0:
            final_coeffs[-1] = 1
        else:
            final_coeffs[-1] = -final_coeffs[-1]
    
    # Local search refinement: try to improve by flipping single coefficients
    if time.time() < deadline - 10:
        current_val = abs(sum(final_coeffs[i] * (L // (i + 1)) for i in range(n)))
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(n):
                # Try changing coefficient at position i
                old_val = final_coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    final_coeffs[i] = new_val
                    new_total = sum(final_coeffs[j] * (L // (j + 1)) for j in range(n))
                    if new_total != 0 and abs(new_total) < current_val:
                        current_val = abs(new_total)
                        improved = True
                        if current_val == 1:
                            break
                    else:
                        final_coeffs[i] = old_val
                if current_val == 1:
                    break
        # Final verification
        total = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
        if total == 0:
            if final_coeffs[-1] == 0:
                final_coeffs[-1] = 1
            else:
                final_coeffs[-1] = -final_coeffs[-1]
    
    return final_coeffs

# EVOLVE-BLOCK-END
