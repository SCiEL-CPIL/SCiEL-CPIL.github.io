import numpy as np
import time
import sys
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds margin
    
    N = 10**7
    
    # 1. Compute sigma(n) for n=1..N
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve
    # This loop is the main bottleneck.
    # We can try to optimize by using a list of indices? No, slicing is best.
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 2. Group indices by sigma value
    groups = defaultdict(list)
    
    # Iterating in Python is slow.
    # Can we vectorize this?
    # We have an array sigma[1..N].
    # We want to group indices i by sigma[i].
    # np.unique with return_inverse can help, but we need the indices.
    
    # Let's try to use np.argsort?
    # sort_indices = np.argsort(sigma[1:N+1])
    # sorted_sigmas = sigma[1:N+1][sort_indices]
    # Then we can find groups where sorted_sigmas are equal.
    # This might be faster than a Python loop of 10^7 dict inserts.
    
    # Let's estimate:
    # np.argsort on 10^7 elements takes ~1-2s.
    # Python loop of 10^7 dict inserts takes ~1-2s.
    # They are comparable.
    # But np.argsort gives us sorted sigmas, so the groups are contiguous in the sorted array.
    # We can then extract the indices for each group.
    
    # Let's try the numpy approach for grouping.
    
    sigmas = sigma[1:N+1]
    sorted_indices = np.argsort(sigmas)
    sorted_sigmas = sigmas[sorted_indices]
    
    # Find boundaries of groups
    # A new group starts where sorted_sigmas[i] != sorted_sigmas[i-1]
    
    # We can use np.diff to find changes
    # But np.diff on 10^7 elements is fast.
    
    # However, we need to iterate through groups to do the binary search.
    # The number of groups can be large (up to N).
    # Most groups will be small (size 1).
    # We only care about groups with size >= 2.
    
    # Let's identify groups with size >= 2.
    
    # We can iterate through the sorted array.
    # To avoid a Python loop of 10^7, we can process in chunks or just let it run.
    # The number of distinct sigma values is much smaller than N.
    # sigma(n) is roughly n log log n.
    # The number of distinct values is probably around 10^6 - 10^7?
    # Actually, sigma(n) can be quite large.
    # Let's just iterate.
    
    # Optimization: 
    # We can create a list of (start, end) for each group.
    
    # Let's use a simple loop to find group boundaries.
    # This loop is O(N) but in Python.
    # Is there a vectorized way to find group boundaries?
    # changes = np.diff(sorted_sigmas) != 0
    # starts = np.where(changes)[0]
    # ends = np.where(changes)[0] + 1
    # This is O(N) in C, very fast.
    
    changes = np.diff(sorted_sigmas) != 0
    starts = np.where(changes)[0]
    ends = np.where(changes)[0] + 1
    
    # Add the first and last boundaries
    # starts should start at 0
    # ends should end at N
    
    # The groups are [0, starts[0]], [starts[0], starts[1]], ..., [starts[-1], N]
    # Wait, np.where(changes)[0] gives indices where change happens.
    # If sorted_sigmas = [1, 1, 2, 2, 2, 3]
    # diff = [0, 1, 0, 1, 1]
    # changes = [False, True, False, True, True]
    # starts = [1, 3, 4]
    # ends = [2, 4, 5]
    # Group 0: [0, 1) -> indices 0
    # Group 1: [1, 3) -> indices 1, 2
    # Group 2: [3, 4) -> index 3
    # Group 3: [4, 6) -> indices 4, 5
    
    # So the groups are:
    # [0, starts[0]]
    # [starts[i], starts[i+1]]
    # [starts[-1], N]
    
    # We need to handle the boundaries carefully.
    
    # Let's create a list of (start, end) pairs.
    group_bounds = []
    prev_start = 0
    for s in starts:
        group_bounds.append((prev_start, s))
        prev_start = s
    group_bounds.append((prev_start, N))
    
    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Search for best pair
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # Iterate through groups
    # We only care about groups with size >= 2
    for start, end in group_bounds:
        if end - start < 2:
            continue
            
        # Extract indices for this group
        # The indices in sorted_indices[start:end] are the original indices (0-based)
        # We need 1-based indices for a and b.
        # sorted_indices contains indices 0..N-1 corresponding to 1..N.
        # So actual value is sorted_indices + 1.
        
        vals = sorted_indices[start:end] + 1
        
        # Sort vals to use bisect
        # vals is already sorted by sigma, but not by value.
        # We need to sort by value (the integer a or b).
        vals = np.sort(vals)
        
        # Now vals is a sorted numpy array of integers.
        # We want to find a, b in vals such that |a/b - 1.5| is minimized.
        # This is equivalent to finding a closest to 1.5 * b.
        
        # Convert to list for bisect? Or use np.searchsorted?
        # np.searchsorted is faster.
        
        # For each b in vals:
        # target = 1.5 * b
        # pos = np.searchsorted(vals, target)
        
        # This is O(K log K) where K is the group size.
        # Sum of K log K over all groups is bounded by N log N.
        
        # Let's vectorize the search?
        # targets = 1.5 * vals
        # pos = np.searchsorted(vals, targets)
        
        # Then we need to check pos and pos-1 for each b.
        
        # Let's try to do this in a loop for now, it's simpler and K is usually small.
        # If K is large, we can optimize.
        
        # To speed up, we can skip groups where min(vals) * 1.5 > max(vals) or max(vals) / 1.5 < min(vals)?
        # No, a/b can be > 1.5 or < 1.5.
        # We want |a/b - 1.5|.
        # So a can be close to 1.5 b.
        
        # Let's iterate.
        for b in vals:
            target = 1.5 * b
            pos = np.searchsorted(vals, target)
            
            # Check pos
            if pos < len(vals):
                a = vals[pos]
                if a != b:
                    err = abs(a/b - 1.5)
                    if err < best_err:
                        if err != 0:
                            best_err = err
                            best_a, best_b = int(a), int(b)
            
            # Check pos - 1
            if pos - 1 >= 0:
                a = vals[pos - 1]
                if a != b:
                    err = abs(a/b - 1.5)
                    if err < best_err:
                        if err != 0:
                            best_err = err
                            best_a, best_b = int(a), int(b)
                            
        if time.time() > deadline - 2:
            break

    return [int(best_a), int(best_b)]
