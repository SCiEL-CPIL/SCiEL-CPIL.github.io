import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs 220 points in Z^11 with squared norm 2 and mutual squared distance >= 2.
    Uses the D11 root system: all vectors with exactly two non-zero entries ±1.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

    # D11 construction: all (±1, ±1, 0, ..., 0) with exactly two non-zero entries
    points_list = []
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)

    points = np.array(points_list, dtype=np.int64)
    k = points.shape[0]

    # Quick sanity checks
    if k < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],
                         [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    norms_sq = np.sum(points**2, axis=1)
    if np.any(norms_sq == 0):
        mask = norms_sq > 0
        points = points[mask]
        k = points.shape[0]
        norms_sq = np.sum(points**2, axis=1)

    max_norm_sq = int(np.max(norms_sq))

    # Verify pairwise minimum squared distance
    min_dist_sq = float('inf')
    valid = True
    for i in range(k):
        if time.time() > deadline:
            break
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break

    valid = valid and (min_dist_sq >= max_norm_sq)

    if not valid:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0],
                         [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    return np.around(points).astype(np.int64)
