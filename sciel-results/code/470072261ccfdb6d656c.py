import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin before 60s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    # Weights for the first 15 terms (1..15) and next 15 (16..30)
    # Sum is c_i / i. Integer sum is c_i * (L // i)
    weights_left = [L // i for i in range(1, 16)]
    weights_right = [L // i for i in range(16, 31)]
    
    def generate_sums(weights):
        """
        Generates all possible integer sums for the given weights.
        Returns a numpy array of sums.
        """
        sums = np.array([0], dtype=np.int64)
        for w in weights:
            # New sums are: old, old + w, old - w
            # We can do this with np.concatenate
            sums = np.concatenate([sums, sums + w, sums - w])
        return sums

    # Generate sums
    left_sums = generate_sums(weights_left)
    right_sums = generate_sums(weights_right)
    
    if time.time() > deadline:
        # Fallback: try to return something valid quickly
        # e.g. c_1=1, c_2=-1, rest 0 -> 1 - 1/2 = 1/2
        return [1, -1] + [0]*28

    # Sort left sums for binary search
    left_sums_sorted = np.sort(left_sums)
    
    best_abs_val = np.iinfo(np.int64).max
    best_left_val = 0
    best_right_val = 0
    
    # For each right sum, find the best left sum
    # We want to minimize |L + R| subject to L + R != 0
    
    # Vectorized search is tricky because we need to handle the "not zero" constraint
    # and find the global min.
    # We can iterate through right sums. 3^15 is ~14M. 
    # 14M iterations in Python is too slow (~10-20s).
    # We need a faster way.
    
    # Alternative:
    # The optimal sum is likely very small.
    # We can check small integer values for the total sum.
    # Let Total = L_total + R_total.
    # We want Total != 0 and |Total| minimal.
    # Total is an integer.
    # We can check Total = 1, -1, 2, -2, ...
    # For a given Total T, does there exist L in left_sums and R in right_sums such that L + R = T?
    # This is equivalent to: does (left_sums) intersect with (T - right_sums)?
    # Or: left_sums + right_sums contains T?
    
    # We can use a set for right_sums?
    # 14M elements in a set is ~100-200MB and slow to build, but lookup is O(1).
    # Building the set takes time.
    
    # Let's try the set approach for Right sums.
    # right_set = set(right_sums)
    # Then for each T in [1, -1, 2, -2, ...]:
    #   Check if there is an L in left_sums such that (T - L) is in right_set.
    #   We can iterate left_sums? No, 14M * T is too slow.
    
    # Better:
    # We have sorted left_sums and sorted right_sums.
    # We want to find pairs (L, R) such that L + R = T.
    # This is a "two sum" problem on sorted arrays, but we have duplicates.
    # We can use two pointers?
    # But we want to find the smallest |T| != 0.
    
    # Let's go back to the vectorized searchsorted approach but optimize it.
    # We can process right_sums in chunks or use a more clever vectorization.
    # Actually, 14M iterations in a tight Python loop is the bottleneck.
    # Can we vectorize the "find best L for R" logic?
    
    # For each R, we want L closest to -R.
    # pos = searchsorted(left_sums_sorted, -R)
    # Candidates: left_sums_sorted[pos-1], left_sums_sorted[pos]
    # We need to check which one gives non-zero sum and smaller abs.
    
    # We can do this in numpy if we can avoid the loop.
    # But `searchsorted` returns an array of positions.
    # Then we gather the candidates.
    # Then we compute the sums.
    # Then we find the min.
    
    # This is what the original code tried, but it stored paths.
    # Here we only store sums.
    
    targets = -right_sums
    insert_pos = np.searchsorted(left_sums_sorted, targets)
    
    # Clip positions
    pos_left = np.clip(insert_pos - 1, 0, len(left_sums_sorted) - 1)
    pos_right = np.clip(insert_pos, 0, len(left_sums_sorted) - 1)
    
    # Get candidate left values
    cand_L1 = left_sums_sorted[pos_left]
    cand_L2 = left_sums_sorted[pos_right]
    
    # Compute sums S1 = cand_L1 + right_sums, S2 = cand_L2 + right_sums
    S1 = cand_L1 + right_sums
    S2 = cand_L2 + right_sums
    
    # We want min |S| where S != 0.
    # Create a mask for non-zero
    mask1 = S1 != 0
    mask2 = S2 != 0
    
    # For zero sums, set abs value to infinity
    inf = np.iinfo(np.int64).max
    abs_S1 = np.where(mask1, np.abs(S1), inf)
    abs_S2 = np.where(mask2, np.abs(S2), inf)
    
    # Find the minimum for each right sum
    min_abs_S = np.minimum(abs_S1, abs_S2)
    
    # Find the global minimum
    global_min_idx = np.argmin(min_abs_S)
    best_abs_val = min_abs_S[global_min_idx]
    
    # Determine which candidate was better
    # If abs_S1[global_min_idx] < abs_S2[global_min_idx], then cand_L1 was better
    if abs_S1[global_min_idx] <= abs_S2[global_min_idx]:
        best_left_val = cand_L1[global_min_idx]
        best_right_val = right_sums[global_min_idx]
    else:
        best_left_val = cand_L2[global_min_idx]
        best_right_val = right_sums[global_min_idx]
        
    if best_abs_val == inf:
        # Should not happen, but fallback
        return [1, -1] + [0]*28
        
    if time.time() > deadline:
        return [1, -1] + [0]*28

    # Now recover coefficients
    # We need to find c_left such that sum(c_i * w_i) = best_left_val
    # and c_right such that sum(c_i * w_i) = best_right_val
    
    def find_coeffs(weights, target):
        n_w = len(weights)
        # Precompute suffix sums for pruning
        suffix_sums = [0] * (n_w + 1)
        for i in range(n_w - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i+1] + weights[i]
            
        path = [0] * n_w
        
        def dfs(idx, current_sum):
            if idx == n_w:
                return current_sum == target
            if abs(target - current_sum) > suffix_sums[idx]:
                return False
                
            # Try -1, 0, 1
            # Heuristic: try the one that moves closer to target first?
            # delta = target - current_sum
            # If delta > 0, we need positive contribution. Try 1, 0, -1
            # If delta < 0, we need negative contribution. Try -1, 0, 1
            delta = target - current_sum
            if delta > 0:
                choices = [1, 0, -1]
            elif delta < 0:
                choices = [-1, 0, 1]
            else:
                choices = [0, 1, -1]
                
            for c in choices:
                path[idx] = c
                if dfs(idx + 1, current_sum + c * weights[idx]):
                    return True
            return False
            
        if dfs(0, 0):
            return path
        return None
        
    left_coeffs = find_coeffs(weights_left, best_left_val)
    right_coeffs = find_coeffs(weights_right, best_right_val)
    
    if left_coeffs is None or right_coeffs is None:
        return [1, -1] + [0]*28
        
    return left_coeffs + right_coeffs
