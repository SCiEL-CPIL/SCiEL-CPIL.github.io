"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    deadline = time.time() + 110

    # Generate norm^2=4 integer vectors in R^11
    candidates = []
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
                candidates.append(v)

    n = len(candidates)
    cand_arr = np.array(candidates)

    # Conflict: dot > 2
    dots = cand_arr @ cand_arr.T
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)

    best_size = 0
    best_indices = None

    rng = np.random.RandomState(42)

    # Phase 1: Greedy with random restarts
    num_restarts = 5000
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        order = rng.permutation(n)
        selected = np.zeros(n, dtype=bool)
        count = 0
        sel_indices = []

        for idx in order:
            if not np.any(conflict[idx] & selected):
                selected[idx] = True
                count += 1
                sel_indices.append(idx)

        if count > best_size:
            best_size = count
            best_indices = sel_indices[:]

        if best_size >= 600:
            break

    # Phase 2: Local search improvement
    if best_indices is not None:
        selected = np.zeros(n, dtype=bool)
        for idx in best_indices:
            selected[idx] = True

        # Try to add any vector that fits
        for j in range(n):
            if not selected[j] and not np.any(conflict[j] & selected):
                selected[j] = True
                best_indices.append(j)
                best_size += 1

        # Iterative improvement: remove one, greedily fill
        for iteration in range(200):
            if time.time() > deadline - 5:
                break

            improved = False
            for idx in list(best_indices):
                if time.time() > deadline - 5:
                    break
                selected[idx] = False
                old_count = len(best_indices)
                new_indices = [i for i in best_indices if i != idx]

                # Greedily add
                for j in range(n):
                    if not selected[j] and not np.any(conflict[j] & selected):
                        selected[j] = True
                        new_indices.append(j)

                if len(new_indices) > old_count:
                    best_indices = new_indices
                    best_size = len(new_indices)
                    improved = True
                else:
                    # Restore
                    selected[idx] = True

            if not improved:
                break

    if best_indices is not None:
        return cand_arr[best_indices].astype(np.int64)

    # Fallback
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
