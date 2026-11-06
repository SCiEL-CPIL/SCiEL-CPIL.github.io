"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_max_abs_ap_sum(signs):
    """Compute the maximum absolute signed sum over all arithmetic progressions in [1, n].
    signs: list or array of length n of -1/+1 (1-indexed conceptually, but stored 0-indexed).
    Returns the max absolute sum (an int).
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_abs = 0
    # All arithmetic progressions: start i (0-indexed), step d >= 1, length L >= 1
    # Position: i + k*d for k=0..L-1, must have i+(L-1)*d < n
    # We can iterate by step d, then by start i, then accumulate.
    # To be efficient, for a fixed d, we can compute sums for each residue class mod d.
    for d in range(1, n + 1):
        # For each residue r in [0, d-1], consider the sequence r, r+d, r+2d, ...
        # Any AP with step d is a contiguous subsequence of one of these residue-class sequences.
        # So we just need the max abs sum over all contiguous subarrays of each residue-class sequence.
        # But wait: an AP with step d starting at i has length L such that i + (L-1)*d < n.
        # This is exactly a contiguous subarray of the sequence [i, i+d, i+2d, ...] truncated to < n.
        # So for each residue class, we compute the max abs subarray sum (Kadane-like).
        for r in range(d):
            if r >= n:
                break
            # Build the sequence for this residue class
            indices = list(range(r, n, d))
            if len(indices) <= 1:
                # Only one element, the AP sum is just signs[r]
                s = arr[r]
                if abs(s) > max_abs:
                    max_abs = abs(s)
                continue
            vals = arr[indices]
            # Max abs subarray sum: max(max_subarray_sum, -min_subarray_sum)
            # Using Kadane's algorithm for max subarray sum
            max_ending_here = 0
            max_so_far = 0
            min_ending_here = 0
            min_so_far = 0
            for v in vals:
                max_ending_here = max(0, max_ending_here + v)
                max_so_far = max(max_so_far, max_ending_here)
                min_ending_here = min(0, min_ending_here + v)
                min_so_far = min(min_so_far, min_ending_here)
            cur_max_abs = max(max_so_far, -min_so_far)
            if cur_max_abs > max_abs:
                max_abs = cur_max_abs
    return max_abs

def construct():
    n = 256
    deadline = time.time() + 75  # safety margin
    
    # Start with a random initial assignment
    rng = random.Random(42)
    signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_signs = signs[:]
    best_val = _compute_max_abs_ap_sum(best_signs)
    
    # Local search: flip one sign at a time if it improves the max abs AP sum
    # To speed up, we can cache the current max and only recompute when needed,
    # but for simplicity, we'll recompute after each flip. If too slow, we'll stop.
    
    # Precompute which APs are "critical" (those achieving the max) to guide flips?
    # For now, simple hill-climbing with random restarts.
    
    while time.time() < deadline:
        improved = False
        # Try flipping each position
        for i in range(n):
            if time.time() > deadline:
                break
            # Flip sign at position i
            signs[i] = -signs[i]
            new_val = _compute_max_abs_ap_sum(signs)
            if new_val < best_val:
                best_val = new_val
                best_signs = signs[:]
                improved = True
            else:
                # Revert
                signs[i] = -signs[i]
        
        if not improved:
            # Random restart with perturbation
            # Perturb best_signs by flipping a few random positions
            perturb_count = rng.randint(5, 15)
            signs = best_signs[:]
            for _ in range(perturb_count):
                j = rng.randint(0, n - 1)
                signs[j] = -signs[j]
            # Optionally, re-optimize from this perturbed state
            # But to keep it simple, just set signs to perturbed and continue hill-climbing
    
    return best_signs
# EVOLVE-BLOCK-END
