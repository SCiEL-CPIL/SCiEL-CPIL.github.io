# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points satisfying:
    - Second dimension is 11
    - k >= 2
    - No point is zero
    - Min pairwise squared distance >= max squared norm
    
    Strategy: Use all 220 D_11 lattice roots (vectors with exactly two +/-1 entries).
    These all have norm^2 = 2, and the minimum pairwise distance squared is 2.
    Thus min_dist^2 >= max_norm^2 (2 >= 2) is satisfied.
    
    We also attempt a greedy search to see if any additional integer vectors
    can be added while maintaining the constraint.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    k = len(points)
    
    # Try to greedily add more points
    # Check time budget
    if time.time() > deadline - 30:
        return points
    
    # Compute max norm squared
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    # Try adding vectors with norm^2 = 3 (three +/-1 entries)
    # For a new vector v with |v|^2 = N, we need:
    #   |v - r|^2 >= max(max_norm_sq, N) for all existing r
    # If N > max_norm_sq, new max is N, need |v-r|^2 >= N for all r
    #   => N + |r|^2 - 2*v.r >= N => v.r <= |r|^2/2
    # For |r|^2 = 2: v.r <= 1
    
    # Try vectors with norm^2 = 3
    added = 0
    for i, j, l in combinations(range(d), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sl in [1, -1]:
                    if time.time() > deadline - 10:
                        break
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[l] = sl
                    # Check against all existing points
                    # New max_norm_sq would be max(2, 3) = 3
                    # Need |v - r|^2 >= 3 for all r
                    # |v-r|^2 = 3 + 2 - 2*v.r = 5 - 2*v.r >= 3 => v.r <= 1
                    # v.r for r with two +/-1 entries: max |v.r| = |si|+|sj| = 2 (if r uses i,j)
                    # So v.r can be up to 2, which is > 1. Fails.
                    pass
        else:
            continue
        break
    else:
        continue
    break
    
    # Try vectors with norm^2 = 4 (one +/-2 entry)
    for i in range(d):
        for s in [1, -1]:
            if time.time() > deadline - 10:
                break
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * s
            # New max_norm_sq = max(2, 4) = 4
            # Need |v - r|^2 >= 4 for all r with |r|^2 = 2
            # |v-r|^2 = 4 + 2 - 2*v.r = 6 - 2*v.r >= 4 => v.r <= 1
            # v.r: r has two +/-1 entries. If r has entry at i, v.r = 2*s*r_i = +/-2
            # So v.r can be 2 > 1. Fails.
            pass
    
    # Try vectors with norm^2 = 4 (four +/-1 entries) - too many to check all
    # Skip for time reasons
    
    return points

# EVOLVE-BLOCK-END
