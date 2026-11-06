"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _max_abs_ap_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    contained in [1, n] (1-indexed), i.e., for all start s, step d > 0, length L >= 1
    such that s + (L-1)*d <= n.
    
    signs: 1D numpy array of length n with values -1 or 1 (0-indexed, so index i corresponds to position i+1)
    Returns: max absolute sum (float or int)
    """
    n = len(signs)
    max_disc = 0
    
    # For efficiency, we can iterate over step d and start s
    # The number of APs is O(n^2 log n) roughly, but for n=256 it's manageable
    # Let's estimate: sum_{d=1}^{n} sum_{s=1}^{n} floor((n-s)/d) + 1
    # For n=256, this is around 256 * (256 * H_256) ~ 256 * 256 * 6 ~ 400k, which is fine in Python if optimized,
    # but vectorized NumPy is better.
    
    # However, doing it in pure Python loops might be too slow if called many times.
    # Let's try to vectorize or at least use efficient loops.
    # Given the 90s limit and need for many evaluations, we should be careful.
    
    # Alternative: Precompute all APs? No, too many.
    # Let's stick to a semi-vectorized approach or optimized Python.
    # Actually, for n=256, a pure Python loop over all APs might take ~0.1-0.5s per eval,
    # which is too slow for many iterations. We need a faster discrepancy calculation.
    
    # Let's try to compute it using cumulative sums along arithmetic progressions.
    # For a fixed step d, we can process each residue class modulo d independently.
    # For a fixed d and start residue r (0 <= r < d), the positions are r, r+d, r+2d, ...
    # The sums over contiguous segments of this sequence are what we need.
    # For a fixed d and residue class, we can compute the prefix sums and then find the max absolute sum over any contiguous subarray.
    # This is the classic "maximum subarray sum" problem, but we need max absolute sum, which is max(max_subarray_sum, -min_subarray_sum).
    
    max_disc = 0
    for d in range(1, n + 1):
        for r in range(d):
            # Extract elements at positions r, r+d, r+2d, ... (0-indexed)
            # These are indices r, r+d, r+2d, ... in the signs array
            # Length of this sequence: L = floor((n-1-r)/d) + 1 if r < n else 0
            if r >= n:
                continue
            L = (n - 1 - r) // d + 1
            if L == 0:
                continue
            
            # Get the subarray
            # Using slicing: signs[r::d]
            sub = signs[r::d]
            # Compute prefix sums
            prefix = np.cumsum(sub)
            # To find max absolute sum over any contiguous subarray:
            # We need max_{i<=j} (prefix[j] - prefix[i-1]) and min_{i<=j} (prefix[j] - prefix[i-1])
            # Which is equivalent to max_{j} prefix[j] - min_{i<j} prefix[i] and min_{j} prefix[j] - max_{i<j} prefix[i]
            # But we can also just compute all contiguous sums via a sliding window or using the fact that:
            # max_sum = max(prefix[j] - min(prefix[0..j-1]))
            # min_sum = min(prefix[j] - max(prefix[0..j-1]))
            
            # Efficient way:
            # Let min_prefix = array of min(prefix[0..j]) for each j
            # Let max_prefix = array of max(prefix[0..j]) for each j
            # Then max_subarray_sum = max(prefix[j] - min_prefix[j-1] for j>=1) and also prefix[0] itself (for j=0, i=1 which is empty, but we consider non-empty)
            # Actually, for non-empty subarrays ending at j (0-indexed in sub), the sum is prefix[j] - prefix[i] where 0 <= i < j+1.
            # If we define prefix_extended = [0, prefix[0], prefix[1], ...], then sum from i to j is prefix_extended[j+1] - prefix_extended[i].
            
            # Simpler: Use the standard Kadane-like approach for max and min subarray sum.
            # But we can vectorize the "max over j of prefix[j] - min(prefix[0..j-1])" part.
            
            # Let's create an array of min_prefix up to each index (exclusive of current for the subtraction, but including current for the value)
            # Actually, let's just compute:
            # min_prefix_so_far = np.minimum.accumulate(prefix)  # This gives min(prefix[0..j]) for each j
            # But we need min(prefix[0..j-1]) for the sum ending at j.
            # Let min_prefix_excl = np.zeros_like(prefix)
            # min_prefix_excl[0] = 0  # min of empty set is 0? No, for sum starting at 0, it's prefix[0] - 0.
            # Actually, the sum of subarray [i..j] is prefix[j] - (prefix[i-1] if i>0 else 0).
            # So we can define P = np.concatenate(([0], prefix))
            # Then sum [i..j] = P[j+1] - P[i] for 0 <= i <= j+1.
            # Max sum = max_{j+1} P[j+1] - min_{i <= j+1} P[i]
            # Min sum = min_{j+1} P[j+1] - max_{i <= j+1} P[i]
            
            P = np.concatenate(([0], prefix))
            # min_P_so_far[k] = min(P[0..k])
            min_P_so_far = np.minimum.accumulate(P)
            max_P_so_far = np.maximum.accumulate(P)
            
            # Max subarray sum = max(P[k] - min_P_so_far[k] for k in 1..len(P)-1)
            # But note: min_P_so_far[k] includes P[k], so P[k] - min_P_so_far[k] is the max sum of a subarray ending at k-1 (in original sub indexing) or something.
            # Let's be precise:
            # Sum of subarray from i to j (0-indexed in sub) is P[j+1] - P[i].
            # For a fixed j, the max sum ending at j is P[j+1] - min(P[0..j]) = P[j+1] - min_P_so_far[j]
            # The min sum ending at j is P[j+1] - max(P[0..j]) = P[j+1] - max_P_so_far[j]
            
            # So:
            max_sums_ending_at_j = P[1:] - min_P_so_far[:-1]  # P[k] - min_P_so_far[k-1] for k=1..len(P)-1
            min_sums_ending_at_j = P[1:] - max_P_so_far[:-1]
            
            current_max = np.max(max_sums_ending_at_j)
            current_min = np.min(min_sums_ending_at_j)
            
            abs_max = max(abs(current_max), abs(current_min))
            if abs_max > max_disc:
                max_disc = abs_max
    
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 256
    rng = np.random.default_rng(12345)
    
    # Initial solution: random
    signs = rng.choice([-1, 1], size=n).astype(np.int8)
    best_signs = signs.copy()
    best_disc = _max_abs_ap_discrepancy(signs)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    alpha = 0.01  # probability of accepting worse solutions initially
    
    while time.time() < deadline:
        # Generate a neighbor: flip a random position
        idx = rng.integers(0, n)
        new_signs = signs.copy()
        new_signs[idx] = -new_signs[idx]
        
        new_disc = _max_abs_ap_discrepancy(new_signs)
        
        if new_disc < best_disc:
            best_signs = new_signs.copy()
            best_disc = new_disc
        
        # Acceptance criterion
        if new_disc < signs_disc or (T > T_min and rng.random() < np.exp(-(new_disc - signs_disc) / T)):
            signs = new_signs
            signs_disc = new_disc
        else:
            signs_disc = _max_abs_ap_discrepancy(signs)  # This is inefficient, let's store it
        
        T *= cooling
    
    return [int(x) for x in best_signs]

# Wait, the above has a bug: signs_disc is not initialized properly in the loop.
# Let's rewrite more carefully.

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 256
    rng = np.random.default_rng(12345)
    
    # Initial solution: random
    signs = rng.choice([-1, 1], size=n).astype(np.int8)
    signs_disc = _max_abs_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = signs_disc
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor: flip a random position
        idx = rng.integers(0, n)
        new_signs = signs.copy()
        new_signs[idx] = -new_signs[idx]
        
        new_disc = _max_abs_ap_discrepancy(new_signs)
        
        if new_disc < best_disc:
            best_signs = new_signs.copy()
            best_disc = new_disc
        
        # Acceptance criterion
        if new_disc < signs_disc:
            signs = new_signs
            signs_disc = new_disc
        else:
            # Accept worse solution with probability exp(-delta/T)
            if rng.random() < np.exp(-(new_disc - signs_disc) / T):
                signs = new_signs
                signs_disc = new_disc
        
        T *= cooling
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
