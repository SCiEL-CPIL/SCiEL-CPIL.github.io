"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    vectors = []

    # Part 1: Type 1 vectors: (±2, ±2, 0, ..., 0)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)

    arr = np.array(vectors, dtype=np.int64)

    # Part 2: Type 2 vectors (eight ±1s)
    # Only one support can be used effectively due to intersection constraints.
    # Use support 0..7.
    # Generate even parity vectors to maximize compatibility.
    for mask in range(256):
        if bin(mask).count('1') % 2 == 0:
            v = [0] * D
            for k in range(8):
                if mask & (1 << k):
                    v[k] = -1
                else:
                    v[k] = 1
            # Check against existing
            v_arr = np.array(v, dtype=np.int64)
            if np.max(np.abs(arr @ v_arr)) <= 4:
                vectors.append(v)
                arr = np.vstack([arr, v_arr])
            if not elapsed_ok():
                break

    # Part 3: Type 3 vectors (one ±2, four ±1s)
    # Generate and check
    candidates = []
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
    
    random.seed(42)
    random.shuffle(candidates)
    
    for v in candidates:
        if not elapsed_ok():
            break
        v_arr = np.array(v, dtype=np.int64)
        if np.max(np.abs(arr @ v_arr)) <= 4:
            vectors.append(v)
            arr = np.vstack([arr, v_arr])

    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
