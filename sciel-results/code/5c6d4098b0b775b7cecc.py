import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    
    points_list = []
    
    # E8 roots: 240 vectors in 8 dimensions with norm^2 = 2
    # Type 1: (+/-1, +/-1, 0, 0, 0, 0, 0, 0) - 8*7*2*2 = 224... 
    # Actually E8 roots are:
    # Type 1: vectors with two non-zero entries of magnitude 1 (signs independent)
    #   C(8,2) * 4 = 28 * 4 = 112
    # Type 2: vectors with all entries +/-1/2 (even number of minus signs)
    #   2^7 = 128
    # Total: 112 + 128 = 240
    
    # But we need integer vectors! E8 roots with half-integers won't work directly.
    # We need to scale E8 by 2 to get integer vectors, giving norm^2 = 8.
    
    # E8 roots scaled by 2:
    # Type 1: vectors with two non-zero entries of magnitude 2 (signs independent)
    #   C(8,2) * 4 = 112
    # Type 2: vectors with all entries +/-1 (even number of minus signs)  
    #   2^7 = 128
    # Total: 240 vectors, each with norm^2 = 8
    
    # Embed E8 roots in first 8 dimensions (dims 0-7), zeros in dims 8-10
    # Type 1: two non-zero entries of magnitude 2
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    # Type 2: all entries +/-1, even number of minus signs
    # Generate all 2^8 sign patterns, keep those with even number of -1's
    for signs in product([-1, 1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for idx, val in enumerate(signs):
                p[idx] = val
            points_list.append(p)
    
    # Now add A3 roots in the last 3 dimensions (dims 8-10)
    # A3 roots: 12 vectors in 3 dimensions with norm^2 = 2
    # These are: (+/-1, +/-1, 0) and permutations, but only in 3D
    # A3 roots: e_i - e_j for i < j, and their negatives
    # In 3D: (1,-1,0), (1,0,-1), (0,1,-1) and their negatives = 6 vectors
    # Wait, A3 has 12 roots. Let me reconsider.
    # A3 root system: all permutations of (+/-1, +/-1, 0, 0) with even number of minus signs?
    # No, A3 is the root system of sl(4), with 12 roots.
    # A3 roots: e_i - e_j for 1 <= i < j <= 4, and their negatives.
    # That's C(4,2) * 2 = 12 roots.
    # But we only have 3 dimensions. We can embed A3 in 4 dimensions, but we only have 3.
    # 
    # Alternative: Use D3 roots. D3 = A3, and D3 has 12 roots:
    # (+/-1, +/-1, 0) and permutations = 3 * 4 = 12
    # These have norm^2 = 2.
    # 
    # Scale D3 roots by 2: norm^2 = 8
    # (+/-2, +/-2, 0) and permutations in dims 8,9,10
    
    for perm in [(8,9,10), (8,10,9), (9,8,10), (9,10,8), (10,8,9), (10,9,8)]:
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[perm[0]] = 2 * si
                p[perm[1]] = 2 * sj
                points_list.append(p)
    
    # Wait, that gives 6 * 4 = 24 points, but D3 only has 12 roots.
    # The issue is that (+2,2,0) and (+2,2,0) are the same. Let me be more careful.
    # D3 roots: for each pair of distinct coordinates, assign +/-1 to those two, 0 to the third.
    # Pairs: (0,1), (0,2), (1,2)
    # For each pair, 4 sign combinations.
    # Total: 3 * 4 = 12
    
    # Let me redo this properly
    points_list = []  # Reset
    
    # E8 roots scaled by 2, embedded in dims 0-7
    # Type 1: two non-zero entries of magnitude 2
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    # Type 2: all entries +/-1, even number of minus signs
    for signs in product([-1, 1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for idx, val in enumerate(signs):
                p[idx] = val
            points_list.append(p)
    
    # D3 roots scaled by 2, embedded in dims 8-10
    # D3 roots: (+/-1, +/-1, 0) with the zero in any of the 3 positions
    # Scaled by 2: (+/-2, +/-2, 0)
    for zero_pos in range(3):
        # The two non-zero positions
        nonzero_positions = [i for i in range(3) if i != zero_pos]
        for s1 in (1, -1):
            for s2 in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[8 + nonzero_positions[0]] = 2 * s1
                p[8 + nonzero_positions[1]] = 2 * s2
                points_list.append(p)
    
    # Total should be 112 + 128 + 12 = 252
    
    # Verify: all points have norm^2 = 8
    # Check min pairwise distance
    points = np.array(points_list, dtype=np.int64)
    n = len(points)
    
    # Quick verification
    norms = np.sum(points**2, axis=1)
    assert np.all(norms == 8), f"Not all norms are 8: {np.unique(norms)}"
    
    # Check pairwise distances (sample check for speed)
    # For the full check, we need min_dist^2 >= 8
    # This is expensive for 252 points: 252*251/2 = 31626 pairs
    # Let's do it since it's fast enough
    
    min_dist_sq = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    assert min_dist_sq >= 8, f"Min dist^2 = {min_dist_sq} < 8"
    
    return points
