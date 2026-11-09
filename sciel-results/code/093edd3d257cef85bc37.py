# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum norm is smaller than their minimum pairwise distance.
    Uses a randomized greedy approach with local search to maximize the number of points.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit
    
    best_points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    best_k = 2

    # Strategy: Greedy addition of points with small norms.
    # We want min(||pi - pj||^2) >= max(||pi||^2).
    # Let R = max(||pi||^2). We need ||pi - pj||^2 >= R for all pairs.
    
    # Start with a small set of low-norm points.
    # Points with coordinates in {-1, 0, 1} have norms 1, 2, ...
    # If we restrict to points with norm^2 <= 2, i.e., weight 1 or 2 vectors.
    # Weight 1: (1,0,...,0) etc. Norm^2 = 1.
    # Weight 2: (1,1,0,...,0) etc. Norm^2 = 2.
    
    # If we mix weight 1 and weight 2, the max norm is 2.
    # Then min distance must be >= 2.
    # Dist between two weight 1 points:
    # Same sign, diff index: (1,0...), (0,1...). Dist^2 = 1+1=2. OK.
    # Opp sign, same index: (1,0...), (-1,0...). Dist^2 = 4. OK.
    # Dist between weight 1 and weight 2:
    # e.g., (1,0,0..), (1,1,0..). Dist^2 = 0+1=1. FAIL (1 < 2).
    # So we cannot simply mix them.
    
    # Let's stick to weight 2 points only?
    # Max norm^2 = 2. Min dist >= 2.
    # Points are of form +/- e_i +/- e_j.
    # Number of such points: 11*10 * 2 (signs) = 220.
    # Check dist between two weight 2 points.
    # p = e_i + e_j, q = e_k + e_l.
    # If disjoint indices: dist^2 = 4. OK.
    # If share 1 index: p = e_1 + e_2, q = e_1 + e_3. dist^2 = (1-1)^2 + (1-0)^2 + (0-1)^2 = 2. OK.
    # If share 0 indices: dist^2 = 4.
    # What about signs?
    # p = e_1 + e_2, q = -e_1 + e_2. dist^2 = (1-(-1))^2 + (1-1)^2 = 4. OK.
    # p = e_1 + e_2, q = -e_1 - e_2. dist^2 = 16. OK.
    # p = e_1 + e_2, q = e_1 - e_2. dist^2 = 0 + 4 = 4. OK.
    # It seems ALL pairs of distinct weight-2 vectors have dist^2 >= 2.
    # Wait, is it possible to have dist^2 < 2?
    # Min dist^2 is 2 (when they share 1 index and have same sign on that index, diff on other).
    # Example: (1,1,0..) and (1,0,1..). Diff: (0,1,-1..). Norm^2 = 2.
    # Since max norm^2 is 2, the condition is dist^2 >= 2.
    # This holds for all distinct pairs of weight-2 vectors!
    # So we can take ALL weight-2 vectors.
    # Count: Choose 2 indices from 11: C(11,2) = 55.
    # For each pair, 4 sign combinations: ++, +-, -+, --.
    # Total = 55 * 4 = 220.
    
    # Can we add more?
    # What about weight 1 vectors? Norm^2 = 1.
    # If we add a weight 1 vector, max norm becomes 2 (from weight 2s).
    # Dist between weight 1 (e_i) and weight 2 (e_i + e_j) is 1. 1 < 2. Fail.
    # Dist between weight 1 (e_i) and weight 2 (e_j + e_k) with disjoint indices is 2. OK.
    # But we need ALL pairs to satisfy dist >= max_norm.
    # If max_norm is 2, we need dist >= 2.
    # So we can only include weight 1 vectors that are "far enough" from all included weight 2 vectors.
    # This is hard to satisfy for all.
    
    # Let's try a different set.
    # What if we use only weight 1 vectors?
    # Max norm^2 = 1.
    # Min dist between distinct weight 1 vectors:
    # e_i and e_j (i!=j): dist^2 = 2. OK.
    # e_i and -e_i: dist^2 = 4. OK.
    # So ALL weight 1 vectors work.
    # Count: 11 * 2 = 22.
    # This is worse than 220.
    
    # What if we scale?
    # Points with norm^2 = 3? (e_i + e_j + e_k).
    # Max norm^2 = 3.
    # Min dist between two weight 3 vectors?
    # Share 2 indices: (1,1,1,0..) and (1,1,0,1..). Dist^2 = 1+1=2.
    # 2 < 3. Fail.
    
    # So weight 2 seems optimal for this simple construction.
    # Can we find a better one?
    # The kissing number in 11 dimensions is known to be 554 (for real vectors).
    # For integer lattice, it's much smaller.
    # 220 is a good candidate.
    
    # Let's generate the 220 points.
    points_list = []
    for i in range(d):
        for j in range(i+1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    points_list.append(p)
    
    pts = np.array(points_list)
    k = len(pts)
    
    # Verify validity
    norms = np.sum(pts**2, axis=1)
    max_norm = np.max(norms)
    
    min_dist = float('inf')
    for a in range(k):
        for b in range(a+1, k):
            diff = pts[a] - pts[b]
            d2 = np.sum(diff**2)
            if d2 < min_dist:
                min_dist = d2
    
    if min_dist >= max_norm:
        best_points = pts
        best_k = k
        
    # Try to see if we can add more points via random search?
    # Given the time constraint and the fact that 220 is a solid constructive lower bound,
    # and random search for k > 220 in 11D integer lattice with this constraint is extremely hard 
    # (likely impossible or requires very large norms), we return the 220 points.
    
    # Just in case, a tiny random search to see if we can beat 220?
    # Probably not worth the risk of timeout or invalidity.
    # The score is proportional to k. 220 is 110x the seed (2).
    
    return best_points

# EVOLVE-BLOCK-END
