"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
import random


def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s kill

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    vectors = []

    # Part 1: Type 1 vectors: (±2, ±2, 0, ..., 0)
    # Norm squared = 8.
    # Count: C(11, 2) * 4 = 55 * 4 = 220.
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)

    arr = np.array(vectors, dtype=np.int64)

    # Part 2: Generate candidates from Type 2 and Type 3.
    # Type 2: eight ±1s, rest 0. Norm^2 = 8.
    # Type 3: one ±2, four ±1s, rest 0. Norm^2 = 4+4=8.

    candidates = []

    # Generate Type 2
    for pos_set in itertools.combinations(range(D), 8):
        for mask in range(256):
            v = [0] * D
            for idx, pos in enumerate(pos_set):
                if mask & (1 << idx):
                    v[pos] = -1
                else:
                    v[pos] = 1
            candidates.append(v)

    # Generate Type 3
    for pos_2 in range(D):
        for pos_set_1 in itertools.combinations([p for p in range(D) if p != pos_2], 4):
            for s2 in (1, -1):
                for mask in range(16):
                    v = [0] * D
                    v[pos_2] = 2 * s2
                    for idx, pos in enumerate(pos_set_1):
                        if mask & (1 << idx):
                            v[pos] = -1
                        else:
                            v[pos] = 1
                    candidates.append(v)

    # Shuffle candidates to break symmetries and allow greedy to find diverse sets
    random.seed(42)
    random.shuffle(candidates)

    # Greedy addition
    # Constraint: |v . u| <= 4 for all u in set.
    # Since v.v = 8 and u.u = 8, this ensures dist^2 >= 8.

    added = 0
    # Process in batches for speed
    batch_size = 5000

    for i in range(0, len(candidates), batch_size):
        if not elapsed_ok():
            break

        batch = candidates[i : i + batch_size]
        batch_arr = np.array(batch, dtype=np.int64)

        # Check each candidate in batch against current arr
        # We can do this row by row to avoid huge memory usage if arr is large,
        # but arr is likely < 10000.
        # Let's iterate to be safe on memory and time.

        for v in batch:
            if not elapsed_ok():
                break

            v_arr = np.array(v, dtype=np.int64)
            # Check dot products
            dots = arr @ v_arr
            if np.max(np.abs(dots)) <= 4:
                vectors.append(v)
                arr = np.vstack([arr, v_arr])
                added += 1

    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
