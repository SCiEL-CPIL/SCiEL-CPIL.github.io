"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _build_ap_index():
    """
    Build a list of arithmetic progressions (as index arrays into 0..N-1).
    We focus on 'critical' APs that are likely to constrain the maximum sum.
    To keep it fast, we include:
      - All singletons (length 1)
      - All length-2 APs (steps 1..N-1, starts such that start+step < N)
      - All length-3 APs
      - All length-4 APs
      - Longer APs for small steps (step=1: all contiguous; step=2,3,... up to some limit)
    
    Actually, for correctness we need to check ALL APs. But that's too many.
    However, the maximum over all APs will be achieved by some AP. 
    Key insight: for a fixed step d, the APs partition [1..N] into d residue classes.
    The worst AP for step d is the longest contiguous subsequence in one residue class.
    
    But to be safe and correct, we should evaluate the true maximum. 
    Let's use a smarter evaluation: for each step d from 1 to N-1, and for each residue r in 0..d-1,
    compute the prefix sums along the AP r, r+d, r+2d, ... and find the max subarray sum.
    The maximum over all such is the answer.
    
    This is O(N^2) total: for each d, total length across all residues is N, and we do O(length) work per residue.
    Sum over d of N = N^2 = 65536, which is fast.
    """
    pass

def _evaluate_max_abs_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions.
    signs: array of +1/-1 of length N.
    Returns the maximum absolute sum.
    """
    max_abs = 0
    # For each step d from 1 to N-1
    for d in range(1, N):
        # For each residue r in 0..d-1
        for r in range(d):
            # Extract the AP: indices r, r+d, r+2d, ... < N
            # Compute prefix sums and find max subarray sum (and min for absolute)
            # We want max |sum| over all contiguous sub-APs
            # Use Kadane-like for max and min
            arr = signs[r::d]
            if len(arr) == 0:
                continue
            # Max subarray sum
            cur_max = arr[0]
            best_max = arr[0]
            cur_min = arr[0]
            best_min = arr[0]
            for i in range(1, len(arr)):
                v = arr[i]
                cur_max = max(v, cur_max + v)
                best_max = max(best_max, cur_max)
                cur_min = min(v, cur_min + v)
                best_min = min(best_min, cur_min)
            abs_max = max(abs(best_max), abs(best_min))
            if abs_max > max_abs:
                max_abs = abs_max
    # Singletons are covered by d=1, r=0..N-1? No, d=1 gives one AP of length N.
    # Wait, for d=1, r=0 gives indices 0,1,...,N-1. That's the whole array.
    # But singletons are APs of length 1 with any step. They are covered when the sub-AP has length 1.
    # In our loop, for any d and r, the AP includes single elements. So singletons are covered.
    return max_abs

def _local_search(signs, max_iter=5000, deadline=None):
    """
    Local search: flip one sign at a time if it reduces the max absolute sum.
    Use a fast evaluation.
    """
    signs = signs.copy()
    current_max = _evaluate_max_abs_sum(signs)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try flipping each position
        # To speed up, we can try random positions first
        for pos in range(N):
            old_val = signs[pos]
            signs[pos] = -old_val
            new_max = _evaluate_max_abs_sum(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
                # Don't revert
                break
            else:
                signs[pos] = old_val
        if not improved:
            # Try random flips to escape local minima
            for _ in range(5):
                pos = random.randint(0, N-1)
                signs[pos] = -signs[pos]
                new_max = _evaluate_max_abs_sum(signs)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    break
                else:
                    signs[pos] = -signs[pos]
            if not improved:
                # Stuck, try a few random restarts of local search
                for _ in range(10):
                    pos = random.randint(0, N-1)
                    signs[pos] = -signs[pos]
                    new_max = _evaluate_max_abs_sum(signs)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        break
                    else:
                        signs[pos] = -signs[pos]
                if not improved:
                    break
    return signs, current_max

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with a random sign sequence
    rng = random.Random(42)
    best_signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    best_max = _evaluate_max_abs_sum(best_signs)
    
    # Run local search
    best_signs, best_max = _local_search(best_signs, max_iter=10000, deadline=deadline)
    
    # Try multiple restarts if time permits
    while time.time() < deadline - 5.0:
        # Random restart
        candidate = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
        candidate, cand_max = _local_search(candidate, max_iter=5000, deadline=deadline)
        if cand_max < best_max:
            best_max = cand_max
            best_signs = candidate
    
    # Return as list of Python integers
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
