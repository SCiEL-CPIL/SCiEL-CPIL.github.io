# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def kissing_number11() -> np.ndarray:
    """
    Constructs 240 integer points in Z^11 using the scaled E8 lattice minimal vectors
    embedded in the first 8 coordinates (last 3 are zero).
    
    E8 minimal vectors (norm^2 = 2, total 240):
      - 112 of type (+/-1, +/-1, 0, 0, 0, 0, 0, 0)
      - 128 of type (+/-1/2, ..., +/-1/2) with even number of minus signs
    
    Scaled by 2 (to get integer coords), all have norm^2 = 8.
    Min distance between distinct scaled minimal vectors: original min dist^2 = 2,
    scaled by 2 gives min dist^2 = 8. So min_dist^2 = 8 = max_norm^2. Valid.
    
    k = 240 > 220 (previous weight-2 construction).
    """
    deadline = time.time() + 580  # safety margin before 630s limit
    
    d = 11
    points_list = []
    
    # Type 1: (+/-1, +/-1, 0, 0, 0, 0, 0, 0) in first 8 coords
    # Choose 2 positions out of 8, assign +/-1 to each
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 8
                    v[i] = si
                    v[j] = sj
                    # Scale by 2
                    v_scaled = [2 * x for x in v]
                    points_list.append(v_scaled + [0, 0, 0])
    
    # Type 2: (+/-1/2, ..., +/-1/2) with even number of minus signs
    # Scaled by 2: (+/-1, ..., +/-1) with even number of minus signs
    # All 8 coordinates are +/-1, even number of -1s
    count_type2 = 0
    for signs in product([1, -1], repeat=8):
        minus_count = sum(1 for s in signs if s == -1)
        if minus_count % 2 == 0:
            points_list.append(list(signs) + [0, 0, 0])
            count_type2 += 1
    
    # Should have 112 + 128 = 240 points
    # Verify count
    assert len(points_list) == 240, f"Expected 240, got {len(points_list)}"
    
    points = np.array(points_list, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
