# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a maximal kissing configuration in Z^11 by taking all integer
    vectors of a fixed small norm. For norm^2 = 2, every pair of distinct
    vectors has squared distance >= 2 (since they are distinct and integer),
    and the max squared norm is 2, so the validity condition holds.

    Returns:
        points: np.ndarray of shape (k, 11) with int64 entries
    """
    d = 11
    t0 = time.time()
    deadline = t0 + 550  # safety margin before 630s kill

    # Generate all integer vectors in Z^11 with squared norm exactly 2.
    # These are vectors with exactly two non-zero entries, each ±1.
    # Count: C(11,2) * 2^2 = 55 * 4 = 220
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)

    points = np.array(points, dtype=np.int64)

    # Verify validity (should always pass for this construction)
    k = points.shape[0]
    assert k >= 2
    assert points.shape[1] == d
    # No zero vector
    norms_sq = np.sum(points ** 2, axis=1)
    assert np.all(norms_sq > 0)
    # Min pairwise distance squared >= max norm squared
    max_norm_sq = int(np.max(norms_sq))
    # For 220 points, the pair check is 220*219/2 = 24090 pairs, fast enough
    min_dist_sq = float('inf')
    for a in range(k):
        for b in range(a + 1, k):
            diff = points[a] - points[b]
            d2 = int(np.dot(diff, diff))
            if d2 < min_dist_sq:
                min_dist_sq = d2
                if min_dist_sq < max_norm_sq:
                    # Should not happen for this construction
                    break
        if min_dist_sq < max_norm_sq:
            break

    assert min_dist_sq >= max_norm_sq, f"Invalid: min_dist_sq={min_dist_sq} < max_norm_sq={max_norm_sq}"

    # Round and cast as required
    points = np.around(points).astype(np.int64)

    elapsed = time.time() - t0
    if elapsed > deadline:
        raise RuntimeError("Deadline exceeded")

    return points


# EVOLVE-BLOCK-END
