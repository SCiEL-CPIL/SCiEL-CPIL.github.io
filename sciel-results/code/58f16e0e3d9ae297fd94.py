# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that 
    the minimum squared distance between any pair is at least the maximum squared norm.
    
    Strategy:
    1. Start with all integer points of squared norm 2 (220 points).
    2. Use a greedy maximum independent set approach on the conflict graph where edges
       connect points with squared distance < max_norm_sq.
    3. Since all points have norm^2 = 2, we need min pairwise dist^2 >= 2.
       Actually, we need min_dist_sq >= max_norm_sq = 2.
       So we need pairwise dist^2 >= 2.
       Points with dist^2 = 0 are same point (excluded).
       Points with dist^2 = 2 conflict? No, dist^2 >= 2 is the requirement.
       So dist^2 = 2 is allowed.
       
       Wait, let's re-read: "minimum squared distance over all pairs of points is >= the maximum squared norm"
       So min_dist_sq >= max_norm_sq.
       If all points have norm^2 = 2, then max_norm_sq = 2.
       So we need min_dist_sq >= 2.
       Since points are distinct, dist^2 >= 1 (for integer points). 
       But dist^2 can be 1? No, for points with norm^2=2, the minimum distance between distinct points...
       
       Let's check: u, v with u·u = v·v = 2.
       ||u-v||^2 = u·u + v·v - 2u·v = 4 - 2u·v.
       u·v can be -2, -1, 0, 1, 2.
       If u·v = 2, then u=v (same point), dist^2 = 0.
       If u·v = 1, dist^2 = 2.
       If u·v = 0, dist^2 = 4.
       If u·v = -1, dist^2 = 6.
       If u·v = -2, dist^2 = 8.
       
       So the minimum possible distance between distinct points of norm^2=2 is 2.
       So ALL 220 points of norm^2=2 form a valid set! Because min_dist_sq = 2 >= max_norm_sq = 2.
       
       Wait, is that right? Let's verify with an example.
       u = (1, 1, 0, ..., 0), v = (1, -1, 0, ..., 0).
       u·u = 2, v·v = 2.
       u·v = 1*(-1) + (-1)*1 + ... = 1 - 1 = 0? No.
       u = (1, 1, 0, ...), v = (1, -1, 0, ...).
       u·v = 1*1 + 1*(-1) = 1 - 1 = 0.
       dist^2 = 4 - 2*0 = 4.
       
       Another example: u = (1, 1, 0, ...), v = (1, 0, 1, 0, ...).
       u·v = 1*1 + 1*0 = 1.
       dist^2 = 4 - 2*1 = 2.
       
       So yes, min_dist_sq = 2, max_norm_sq = 2. Valid!
       
       But wait, can we do better? The known kissing number in 11D is 558.
       We can try to add points of higher norm, or use a different strategy.
       
       Actually, the task says "maximum squared norm of the points". So if we include points with norm^2 > 2,
       then max_norm_sq increases, and we need min_dist_sq to be even larger.
       
       Alternative: use points with norm^2 = 4?
       For points with norm^2 = 4, max_norm_sq = 4.
       We need min_dist_sq >= 4.
       ||u-v||^2 = 8 - 2u·v >= 4 => u·v <= 2.
       
       How many integer points have norm^2 = 4?
       (2,0,...,0): 11*2 = 22 points.
       (1,1,1,0,...,0) with three 1s: C(11,3)*2^3 = 165*8 = 1320 points.
       (1,1,0,...,0) with two 1s: already counted? No, norm^2=2.
       (sqrt(4),...) no, integer only.
       (2,0,...) norm^2=4.
       (1,1,1,...) norm^2=3.
       (1,1,0,...) norm^2=2.
       (0,0,...) norm^2=0.
       
       So norm^2=4 points: 22 (type (2,0,..)) + 1320 (type (1,1,1,0,..)) = 1342 points.
       But we need u·v <= 2 for all pairs.
       
       This is more complex. Let's stick with norm^2=2 for now, which gives 220 points.
       
       Actually, can we mix norms? If we include points of norm^2=2 and norm^2=4,
       max_norm_sq = 4, so we need min_dist_sq >= 4 for ALL pairs.
       For two norm^2=2 points, min_dist_sq = 2 < 4. So they would conflict.
       So we can't mix.
       
       Let's try to find a larger set using norm^2=4 points with the constraint u·v <= 2.
       
       Or, we can use a greedy algorithm on the 220 norm^2=2 points, but since they all work,
       we get 220.
       
       Wait, the current program gets 64. The inspiration says 110 -> 219.
       So 220 should be achievable. Let me just return all 220 norm^2=2 points.
       
       Let me verify: are all 220 points valid?
       - All have norm^2 = 2, so max_norm_sq = 2.
       - Min pairwise dist^2: as computed, min is 2.
       - 2 >= 2, so valid.
       - k = 220 >= 2.
       - No point is zero.
       - Second dimension is 11.
       
       So returning all 220 should give score 220.
    """
    start_time = time.time()
    deadline = 600.0
    
    d = 11
    
    # Generate all integer points with squared norm 2 in 11 dimensions.
    # These are points with exactly two non-zero coordinates, each being +/-1.
    points_list = []
    
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify the condition
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    n_points = len(points)
    min_dist_sq = float('inf')
    for i in range(n_points):
        for j in range(i + 1, n_points):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    if min_dist_sq >= max_norm_sq and n_points >= 2:
        return np.around(points).astype(np.int64)
    else:
        # Fallback
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

# EVOLVE-BLOCK-END
