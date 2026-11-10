# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    def get_points(a):
        # 4x4 grid structure with square symmetry
        # b is fixed to 1.0 for scale invariance
        b = 1.0
        pts = []
        # (±a, ±a)
        for x in [-a, a]:
            for y in [-a, a]:
                pts.append([x, y])
        # (±a, ±b)
        for x in [-a, a]:
            for y in [-b, b]:
                pts.append([x, y])
        # (±b, ±a)
        for x in [-b, b]:
            for y in [-a, a]:
                pts.append([x, y])
        # (±b, ±b)
        for x in [-b, b]:
            for y in [-b, b]:
                pts.append([x, y])
        
        return np.array(pts)

    def score(a):
        if a <= 0 or a >= 1.0:
            return 0.0
        pts = get_points(a)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
            
        return (d_min / d_max) ** 2

    # Grid search for a in (0, 1)
    best_val = 0
    best_a = 0.5
    
    # Coarse search
    a_vals = np.linspace(0.1, 0.9, 100)
    for a in a_vals:
        val = score(a)
        if val > best_val:
            best_val = val
            best_a = a
            
    # Fine search around best_a
    a_fine = np.linspace(max(0.01, best_a - 0.05), min(0.99, best_a + 0.05), 100)
    for a in a_fine:
        val = score(a)
        if val > best_val:
            best_val = val
            best_a = a
            
    return get_points(best_a)

# EVOLVE-BLOCK-END
