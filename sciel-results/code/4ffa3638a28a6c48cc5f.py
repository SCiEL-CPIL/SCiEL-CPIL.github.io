D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def construct():
    deadline = time.time() + 130  # safety margin before 150s kill

    # Base set: all vectors with exactly 1 or 2 non-zero entries of ±1
    vectors = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            vectors.append(tuple(v))
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(tuple(v))

    vec_list = list(set(vectors))
    vec_array = np.array(vec_list, dtype=np.int64)
    max_norm_sq = int(np.max(np.sum(vec_array * vec_array, axis=1)))

    # Generate additional candidates: vectors with 3+ non-zero ±1 entries
    candidates = []
    for nz in range(3, D + 1):
        for indices in itertools.combinations(range(D), nz):
            for signs in itertools.product([1, -1], repeat=nz):
                v = [0] * D
                for idx, s in zip(indices, signs):
                    v[idx] = s
                candidates.append(tuple(v))

    existing = set(vec_list)
    candidates = [c for c in candidates if c not in existing]
    candidates.sort(key=lambda v: sum(x * x for x in v))

    # Greedily add candidates that satisfy the constraint
    added = 0
    for c in candidates:
        if time.time() > deadline:
            break
        c_arr = np.array(c, dtype=np.int64)
        c_norm_sq = int(np.sum(c_arr * c_arr))
        new_max_norm_sq = max(max_norm_sq, c_norm_sq)

        diffs = vec_array - c_arr
        dists_sq = np.sum(diffs * diffs, axis=1)
        min_d = int(np.min(dists_sq))

        if min_d >= new_max_norm_sq:
            vec_list.append(c)
            vec_array = np.vstack([vec_array, c_arr])
            max_norm_sq = new_max_norm_sq
            added += 1

    result = np.array(vec_list, dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
