# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs 252 integer points in 11D from the 2E8 ⊕ 2D3 lattice.
    All points have squared norm 8, and minimum pairwise squared distance is 8.
    """
    start_time = time.time()
    deadline = 600.0

    # --- Construct 2*E8 minimal vectors (240 vectors in 8D, all norm^2 = 8) ---
    e8_vectors = []

    # Type 1: (±2, ±2, 0, 0, 0, 0, 0, 0) all permutations: C(8,2)*4 = 112
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 8
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    e8_vectors.append(v)

    # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs: 128
    for signs in range(256):
        vec = []
        minus_count = 0
        for bit in range(8):
            if (signs >> bit) & 1:
                vec.append(-1)
                minus_count += 1
            else:
                vec.append(1)
        if minus_count % 2 == 0:
            e8_vectors.append(vec)

    # --- Construct 2*D3 minimal vectors (12 vectors in 3D, all norm^2 = 8) ---
    # Vectors with norm^2 = 8 in Z^3: (±2, ±2, 0) and permutations: C(3,2)*4 = 12
    d3_vectors = []
    for i in range(3):
        for j in range(i + 1, 3):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * 3
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    d3_vectors.append(v)

    # --- Combine into 11D points ---
    points = []

    # 2*E8 vectors (240 points): 8D E8 part + 3D zeros
    for v in e8_vectors:
        points.append(v + [0, 0, 0])

    # 2*D3 vectors (12 points): 8D zeros + 3D D3 part
    for v in d3_vectors:
        points.append([0] * 8 + v)

    # Convert to numpy array
    points_arr = np.array(points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)

    # Quick validity check
    assert points_arr.shape == (252, 11), f"Shape is {points_arr.shape}"
    assert not np.any(points_arr == 0), "Zero vector found"

    # Verify min pairwise squared distance >= max squared norm
    norms_sq = np.sum(points_arr ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)

    n = len(points_arr)
    min_pair_dist_sq = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            diff = points_arr[i] - points_arr[j]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_pair_dist_sq:
                min_pair_dist_sq = dist_sq

    assert min_pair_dist_sq >= max_norm_sq, \
        f"min_dist_sq={min_pair_dist_sq}, max_norm_sq={max_norm_sq}"

    elapsed = time.time() - start_time
    if elapsed > deadline:
        raise TimeoutError("Computation exceeded deadline")

    return points_arr


# EVOLVE-BLOCK-END
