"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that the signed sum
        S = sum_{i=1}^{30} c[i-1] / i
    is nonzero and has minimal absolute value.
    
    Uses meet-in-the-middle:
    - Split indices into two halves of 15.
    - Enumerate all 3^15 combinations for each half.
    - Store (scaled_sum, index) for each combination.
    - Sort one half by sum.
    - For each sum in the other half, binary search for the complement.
    - Reconstruct coefficients from the best pair of indices.
    """
    n = 30
    
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute scaled weights
    weights = [L // (i + 1) for i in range(n)]
    
    half = n // 2
    
    # Generate all sums and indices for a half
    def generate_half(start, length):
        """
        Returns lists of sums and indices.
        index is the base-3 representation of the coefficients for this half.
        Mapping: -1 -> 0, 0 -> 1, 1 -> 2
        """
        sums = [0]
        indices = [0]
        for i in range(start, start + length):
            w = weights[i]
            new_sums = []
            new_indices = []
            # Pre-allocate if possible? Lists are dynamic.
            # We iterate through current sums
            for s, idx in zip(sums, indices):
                # -1 (digit 0)
                new_sums.append(s - w)
                new_indices.append(idx * 3)
                # 0 (digit 1)
                new_sums.append(s)
                new_indices.append(idx * 3 + 1)
                # 1 (digit 2)
                new_sums.append(s + w)
                new_indices.append(idx * 3 + 2)
            sums = new_sums
            indices = new_indices
        return sums, indices

    sums1, inds1 = generate_half(0, half)
    sums2, inds2 = generate_half(half, n - half)
    
    # Sort sums2 with corresponding indices
    # To save memory, we sort an array of indices based on sums
    order = list(range(len(sums2)))
    order.sort(key=sums2.__getitem__)
    
    # Create sorted views
    sorted_sums2 = [sums2[i] for i in order]
    sorted_inds2 = [inds2[i] for i in order]
    
    best_abs = float('inf')
    best_idx1 = -1
    best_idx2 = -1
    
    # Iterate through first half
    for s1, idx1 in zip(sums1, inds1):
        target = -s1
        pos = bisect_left(sorted_sums2, target)
        
        # Check neighbors
        for p in (pos - 1, pos, pos + 1):
            if 0 <= p < len(sorted_sums2):
                s2 = sorted_sums2[p]
                total = s1 + s2
                if total == 0:
                    continue
                abs_total = abs(total)
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_idx1 = idx1
                    best_idx2 = sorted_inds2[p]
    
    # Reconstruct coefficients
    def decode_index(idx, length):
        coeffs = []
        for _ in range(length):
            digit = idx % 3
            idx //= 3
            coeffs.append(digit - 1)
        return coeffs
    
    coeffs1 = decode_index(best_idx1, half)
    coeffs2 = decode_index(best_idx2, n - half)
    
    return coeffs1 + coeffs2

# EVOLVE-BLOCK-END
