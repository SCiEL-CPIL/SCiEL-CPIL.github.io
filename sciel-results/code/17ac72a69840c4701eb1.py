# EVOLVE-BLOCK-START
import numpy as np


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance.
    Uses the standard basis vectors and their negatives, which form a valid configuration of size 2*d.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    # Generate 2*d points: e_1, ..., e_d, -e_1, ..., -e_d
    # These are integer vectors.
    # Norms are 1. Max norm squared is 1.
    # Distances:
    # e_i to -e_i is 2. d^2 = 4.
    # e_i to e_j (i!=j) is sqrt(2). d^2 = 2.
    # Min squared distance is 2.
    # Condition: 2 > 1. Valid.
    
    points = np.zeros((2 * d, d), dtype=np.int64)
    for i in range(d):
        points[i, i] = 1
        points[i + d, i] = -1

    return points


# EVOLVE-BLOCK-END
