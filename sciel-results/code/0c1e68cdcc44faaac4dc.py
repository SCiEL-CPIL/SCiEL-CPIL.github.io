import numpy as np
from math import gcd
from functools import reduce

def construct():
    n = 30
    # LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n+1))
    
    # Values: L/i for i=1..n
    vals = np.array([L // i for i in range(1, n+1)], dtype=np.int64)
    
    # Split into two halves
    mid = n // 2
    left_vals = vals[:mid]
    right_vals = vals[mid:]
    
    # Function to generate all 3^k signed sums
    # Returns a numpy array of sums
    def generate_sums(values):
        # Start with [0]
        sums = np.array([0], dtype=np.int64)
        for v in values:
            # For each existing sum, create 3 new sums: s, s+v, s-v
            # This effectively expands the dimension
            # We need to be careful with memory.
            # 3^15 is ~14M.
            sums = np.concatenate([sums, sums + v, sums - v])
        return sums

    # Generate sums for left and right halves
    # This will take some time and memory, but is feasible.
    left_sums = generate_sums(left_vals)
    right_sums = generate_sums(right_vals)
    
    # Sort right sums and keep track of original indices
    sorted_right_sums = np.sort(right_sums)
    right_indices = np.argsort(right_sums)
    
    best_abs = np.iinfo(np.int64).max
    best_left_idx = -1
    best_right_idx = -1
    
    # Iterate through all left sums
    # For each, find the best matching right sum
    # We want to minimize |left_sum + right_sum| != 0
    
    # Vectorized search is difficult due to the "!= 0" constraint and argmin logic per element.
    # We can use searchsorted to find the position where -left_sum would be inserted.
    
    targets = -left_sums
    # Find insertion points in sorted_right_sums
    pos = np.searchsorted(sorted_right_sums, targets)
    
    # Candidates are at pos and pos-1
    # We need to handle boundaries
    # pos can be 0 or len(sorted_right_sums)
    
    # Create candidate indices arrays
    # Candidate 1: pos
    cand1_pos = np.clip(pos, 0, len(sorted_right_sums) - 1)
    # Candidate 2: pos - 1
    cand2_pos = np.clip(pos - 1, 0, len(sorted_right_sums) - 1)
    
    # Calculate sums for candidate 1
    right_vals_c1 = sorted_right_sums[cand1_pos]
    totals_c1 = left_sums + right_vals_c1
    
    # Calculate sums for candidate 2
    right_vals_c2 = sorted_right_sums[cand2_pos]
    totals_c2 = left_sums + right_vals_c2
    
    # Filter out zero totals
    # We want the minimum absolute value among non-zero totals
    
    # Combine candidates
    # We can process c1 and c2 separately to avoid complex masking logic on combined arrays
    
    # Process C1
    mask1 = totals_c1 != 0
    if np.any(mask1):
        abs_totals_c1 = np.abs(totals_c1[mask1])
        valid_indices_c1 = np.where(mask1)[0]
        min_idx_c1 = np.argmin(abs_totals_c1)
        min_val_c1 = abs_totals_c1[min_idx_c1]
        
        if min_val_c1 < best_abs:
            best_abs = min_val_c1
            best_left_idx = valid_indices_c1[min_idx_c1]
            best_right_idx = right_indices[cand1_pos[valid_indices_c1[min_idx_c1]]]
            
    # Process C2
    mask2 = totals_c2 != 0
    if np.any(mask2):
        abs_totals_c2 = np.abs(totals_c2[mask2])
        valid_indices_c2 = np.where(mask2)[0]
        min_idx_c2 = np.argmin(abs_totals_c2)
        min_val_c2 = abs_totals_c2[min_idx_c2]
        
        if min_val_c2 < best_abs:
            best_abs = min_val_c2
            best_left_idx = valid_indices_c2[min_idx_c2]
            best_right_idx = right_indices[cand2_pos[valid_indices_c2[min_idx_c2]]]

    if best_left_idx == -1:
        # Fallback, though unlikely
        return [0] * 28 + [1, -1]

    # Decode base-3 indices to coefficients
    # The index represents the choice for each variable in base 3.
    # Order of generation:
    # sums starts [0]
    # for v in values:
    #   sums = [sums, sums+v, sums-v]
    # This corresponds to:
    # index 0: all 0
    # index 1: last var is 1?
    # Let's trace:
    # k=0: sums=[0]
    # k=1: sums=[0, v0, -v0]
    #   idx 0: coeff 0
    #   idx 1: coeff 1
    #   idx 2: coeff -1
    # k=2:
    #   prev: [0, v0, -v0]
    #   new: [0, v0, -v0, v1, v0+v1, -v0+v1, -v1, -v0-v1, -v0-v1? No]
    #   Let's check the concatenation:
    #   [sums] + [sums+v1] + [sums-v1]
    #   idx 0: 0 (from sums[0]) -> coeffs [0, 0]
    #   idx 1: v0 (from sums[1]) -> coeffs [1, 0]
    #   idx 2: -v0 (from sums[2]) -> coeffs [-1, 0]
    #   idx 3: v1 (from sums[0]+v1) -> coeffs [0, 1]
    #   idx 4: v0+v1 (from sums[1]+v1) -> coeffs [1, 1]
    #   idx 5: -v0+v1 (from sums[2]+v1) -> coeffs [-1, 1]
    #   idx 6: -v1 (from sums[0]-v1) -> coeffs [0, -1]
    #   idx 7: v0-v1 (from sums[1]-v1) -> coeffs [1, -1]
    #   idx 8: -v0-v1 (from sums[2]-v1) -> coeffs [-1, -1]
    
    # So the index is in base 3, where the least significant digit corresponds to the last variable processed?
    # No, look at the indices:
    # idx 0: 00
    # idx 1: 10 (first var 1, second 0)
    # idx 2: 20 (first var -1 mapped to 2? No, -1 is separate)
    # The mapping is:
    # digit 0 -> coeff 0
    # digit 1 -> coeff 1
    # digit 2 -> coeff -1
    # And the digits are ordered such that the first variable is the most significant digit?
    # Let's check idx 3 (binary 11? No, base 3 10).
    # idx 3: base 3 is 10.
    #   Most significant digit (1) -> coeff for first var?
    #   Least significant digit (0) -> coeff for second var?
    #   If MSB is first var:
    #   idx 3 (10_3): first var digit 1 (coeff 1), second var digit 0 (coeff 0).
    #   But we established idx 3 corresponds to [0, 1] (first 0, second 1).
    #   So MSB is NOT the first var.
    
    # Let's re-evaluate the index structure.
    # The array is built by appending.
    # The index $k$ in the final array of size $3^N$ can be decomposed.
    # Actually, it's easier to just decode it.
    # The index $k$ corresponds to a sequence of choices $d_{N-1}, \dots, d_0$ in base 3?
    # Let's look at $N=2$.
    # $k=0 \rightarrow 00$
    # $k=1 \rightarrow 01$ (if we assume standard base 3 counting 0,1,2)
    # But our generation:
    # Step 1 (v0): [0, v0, -v0]
    # Step 2 (v1):
    #   0: 0
    #   1: v0
    #   2: -v0
    #   3: v1
    #   4: v0+v1
    #   5: -v0+v1
    #   6: -v1
    #   7: v0-v1
    #   8: -v0-v1
    
    # Let's map index to coefficients:
    # 0: [0, 0]
    # 1: [1, 0]
    # 2: [-1, 0]
    # 3: [0, 1]
    # 4: [1, 1]
    # 5: [-1, 1]
    # 6: [0, -1]
    # 7: [1, -1]
    # 8: [-1, -1]
    
    # It looks like:
    # index = d0 * 1 + d1 * 3 ... ?
    # Let's check base 3 representation of index $k$.
    # $k=0: 00_3 \rightarrow [0,0]$
    # $k=1: 01_3 \rightarrow [1,0]$ ? No, $1 = 0*3 + 1$. If digits are $d_1 d_0$, then $d_1=0, d_0=1$.
    # If $d_0$ corresponds to the last variable (v1) and $d_1$ to the first (v0):
    #   $k=1 (01_3) \rightarrow d_1=0, d_0=1 \rightarrow [0, 1]$?
    #   But we said $k=1$ is $[1, 0]$.
    #   So $d_0$ corresponds to the FIRST variable?
    #   $k=1 (01_3) \rightarrow d_0=1, d_1=0 \rightarrow [1, 0]$. Correct.
    #   $k=3 (10_3) \rightarrow d_0=0, d_1=1 \rightarrow [0, 1]$. Correct.
    #   $k=4 (11_3) \rightarrow d_0=1, d_1=1 \rightarrow [1, 1]$. Correct.
    #   $k=2 (02_3) \rightarrow d_0=2, d_1=0 \rightarrow [-1, 0]$. Correct.
    
    # So, the index $k$ in base 3 gives digits $d_{N-1} \dots d_0$.
    # $d_0$ is the coefficient for the FIRST variable in the list.
    # $d_1$ is the coefficient for the SECOND variable.
    # ...
    # $d_{N-1}$ is the coefficient for the LAST variable.
    
    # Mapping digit to coefficient:
    # 0 -> 0
    # 1 -> 1
    # 2 -> -1
    
    def decode_index(idx, num_vars):
        coeffs = [0] * num_vars
        for i in range(num_vars):
            d = idx % 3
            idx //= 3
            if d == 0:
                coeffs[i] = 0
            elif d == 1:
                coeffs[i] = 1
            else: # d == 2
                coeffs[i] = -1
        return coeffs

    left_coeffs = decode_index(best_left_idx, mid)
    right_coeffs = decode_index(best_right_idx, n - mid)
    
    full_coeffs = left_coeffs + right_coeffs
    
    # Verify
    total = 0
    for i, c in enumerate(full_coeffs):
        if c != 0:
            total += c * (L // (i + 1))
            
    if total == 0:
        # This should not happen with the logic above, but just in case
        return [0] * 28 + [1, -1]
        
    return full_coeffs
