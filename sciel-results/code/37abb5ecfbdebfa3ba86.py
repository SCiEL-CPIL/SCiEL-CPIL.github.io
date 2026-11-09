import numpy as np
import time
import bisect
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin out of 60s
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n in [1, N]
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Vectorized divisor sum sieve
    # For each d, add d to all multiples of d
    for d in range(1, N + 1):
        if time.time() > deadline - 15:
            break
        sigma[d::d] += d
    
    if time.time() > deadline - 10:
        # If we ran out of time computing sigma, return best known
        return [11, 6]
    
    # 2. Group indices by sigma value using np.unique
    vals = sigma[1:]  # sigma[1], sigma[2], ..., sigma[N]
    
    # Get unique sigma values and their inverse mapping
    unique_sigmas, inverse = np.unique(vals, return_inverse=True)
    
    # Count how many numbers have each sigma value
    counts = np.bincount(inverse)
    
    # Only consider sigma values that appear at least twice
    valid_sigma_mask = counts >= 2
    valid_sigma_indices = np.where(valid_sigma_mask)[0]
    
    # Get the actual numbers grouped by sigma value
    # We need to find, for each valid sigma, the list of n values
    
    # Strategy: for each valid sigma value, collect all n with that sigma
    # Then find pairs (a,b) in that group minimizing |a/b - 3/2|
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # For efficiency, we'll process sigma values in order of potential
    # The best possible error for a group with max value M is ~1/(2M)
    # So we should process groups with larger values first
    
    # Build a list of (max_value, sigma_index) for valid groups
    # We need to know the max n in each group
    
    # Alternative: iterate through all n, and for each, check if sigma(n) has been seen
    # and if so, try to find a good pair
    
    # Better approach: for each valid sigma group, get all n values, sort them,
    # and for each b, find a closest to 1.5*b
    
    # To avoid building all groups at once (memory), process one sigma at a time
    # But np.unique gives us the inverse mapping, so we can use it
    
    # Create a list of groups: for each unique sigma index, list of n values
    # This is memory intensive for 10M elements, but let's try
    
    # Actually, let's use a different approach:
    # Sort indices by sigma value, then process contiguous segments
    
    # Sort 1..N by sigma value
    # This is O(N log N) but numpy is fast
    
    indices = np.arange(1, N + 1)
    # argsort of sigma[1:]
    sorted_order = np.argsort(vals, kind='stable')
    sorted_indices = indices[sorted_order]
    sorted_sigmas = vals[sorted_order]
    
    # Find segment boundaries where sigma value changes
    diff = np.diff(sorted_sigmas)
    change_points = np.where(diff != 0)[0] + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(sorted_sigmas)]))
    
    sizes = ends - starts
    
    # Only process segments with size > 1
    valid_segs = np.where(sizes > 1)[0]
    
    # Sort valid segments by their max value (descending) to find good pairs early
    # Max value in segment i is max(sorted_indices[starts[i]:ends[i]])
    # Computing this for all segments is expensive, so let's just process in order
    # but skip small segments
    
    # Process segments, skipping those that can't beat current best
    for i in valid_segs:
        if time.time() > deadline - 3:
            break
        
        s = starts[i]
        e = ends[i]
        seg_size = e - s
        
        if seg_size < 2:
            continue
        
        # Get the numbers in this segment
        seg = sorted_indices[s:e]
        
        # Max b in this segment
        max_b = np.max(seg)
        
        # Best possible error is 1/(2*max_b)
        if 1.0 / (2.0 * max_b) >= best_error:
            continue
        
        # Sort the segment by value
        seg_sorted = np.sort(seg)
        n_seg = len(seg_sorted)
        
        # For each b in seg_sorted, find a closest to 1.5*b
        # Use binary search for efficiency
        
        if n_seg <= 20:
            # Small segment: brute force
            for bi in range(n_seg):
                b = int(seg_sorted[bi])
                for ai in range(n_seg):
                    if ai == bi:
                        continue
                    a = int(seg_sorted[ai])
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    err = num / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
        else:
            # Larger segment: binary search
            for b in seg_sorted:
                b_int = int(b)
                target = 1.5 * b_int
                pos = bisect.bisect_left(seg_sorted, target)
                
                # Check candidates around pos
                checked = set()
                for offset in [-2, -1, 0, 1, 2]:
                    idx = pos + offset
                    if 0 <= idx < n_seg and idx not in checked:
                        checked.add(idx)
                        a = int(seg_sorted[idx])
                        if a == b_int:
                            continue
                        num = abs(2 * a - 3 * b_int)
                        if num == 0:
                            continue
                        err = num / (2.0 * b_int)
                        if err < best_error:
                            best_error = err
                            best_a = a
                            best_b = b_int
    
    return [int(best_a), int(best_b)]

if __name__ == "__main__":
    pass
