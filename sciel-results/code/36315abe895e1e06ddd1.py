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
    
    # Enumerate all signed sums for left half (3^15 possibilities)
    # Use iterative construction
    def enumerate_signed_sums(values):
        sums = [0]
        for v in values:
            new_sums = []
            for s in sums:
                new_sums.append(s)      # coefficient 0
                new_sums.append(s + v)   # coefficient 1
                new_sums.append(s - v)   # coefficient -1
            sums = new_sums
        return np.array(sums, dtype=np.int64)
    
    # This will be 3^15 = 14M elements, which is large but let's try
    # Actually, let's use a more memory-efficient approach with numpy
    def enumerate_sums_np(values):
        sums = np.array([0], dtype=np.int64)
        for v in values:
            # For each existing sum, create three new sums
            sums = np.concatenate([sums, sums + v, sums - v])
        return sums
    
    left_sums = enumerate_sums_np(left_vals)
    right_sums = enumerate_sums_np(right_vals)
    
    # Sort right_sums for binary search
    right_sums_sorted = np.sort(right_sums)
    
    # For each left sum, find the right sum that minimizes |left + right|
    # We want left + right != 0 and |left + right| minimized
    
    # Use searchsorted to find insertion point
    # For each left_sum, we want right_sum ≈ -left_sum
    # Search for -left_sum in right_sums_sorted
    
    targets = -left_sums
    # Find insertion points
    indices = np.searchsorted(right_sums_sorted, targets)
    
    # Check candidates: index and index-1
    # Handle boundaries
    indices_clipped = np.clip(indices, 0, len(right_sums_sorted) - 1)
    
    best_abs = None
    best_left_idx = None
    best_right_idx = None
    
    # Vectorized check of both candidates
    for offset in [0, 1]:
        idx = np.clip(indices + offset, 0, len(right_sums_sorted) - 1)
        total = left_sums + right_sums_sorted[idx]
        # Exclude zero
        mask = total != 0
        if not np.any(mask):
            continue
        abs_total = np.abs(total[mask])
        idx_valid = np.where(mask)[0]
        min_idx = np.argmin(abs_total)
        min_val = abs_total[min_idx]
        if best_abs is None or min_val < best_abs:
            best_abs = min_val
            best_left_idx = idx_valid[min_idx]
            best_right_idx = idx[idx_valid[min_idx]]
    
    if best_abs is None:
        # Fallback
        return [0] * 28 + [1, -1]
    
    # Reconstruct coefficients
    # We need to recover which signs were used for each term
    # This is tricky with the numpy approach. Let's use a different strategy.
    
    # Actually, let's just do a local search from a random start to find the actual coefficients
    # Use the best_abs as a target to verify
    
    # Better approach: use simulated annealing to directly find coefficients
    import random
    random.seed(42)
    
    def compute_sum(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * (L // (i + 1))
        return s
    
    # Start with random assignment
    coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
    current_sum = compute_sum(coeffs)
    if current_sum == 0:
        coeffs[0] = 1
        current_sum = compute_sum(coeffs)
    
    best_sum = current_sum
    best_coeffs = coeffs[:]
    
    # Simulated annealing
    T = 1000.0
    cooling = 0.99
    iterations = 50000
    
    for it in range(iterations):
        # Pick random position
        pos = random.randint(0, n - 1)
        old_val = coeffs[pos]
        # Try changing to another value
        new_val = random.choice([v for v in [-1, 0, 1] if v != old_val])
        coeffs[pos] = new_val
        new_sum = compute_sum(coeffs)
        
        if new_sum == 0:
            # Reject zero
            coeffs[pos] = old_val
            continue
        
        delta = abs(new_sum) - abs(best_sum)
        if delta < 0 or random.random() < np.exp(-delta / T):
            if abs(new_sum) < abs(best_sum):
                best_sum = new_sum
                best_coeffs = coeffs[:]
        else:
            coeffs[pos] = old_val
        
        T *= cooling
    
    # Verify and return
    final_sum = compute_sum(best_coeffs)
    if final_sum == 0:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
