"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def _min_pairwise_dist2(vectors):
    """Return minimum squared pairwise distance among rows of vectors (k x d)."""
    if len(vectors) < 2:
        return np.inf
    # Use chunked computation to avoid O(k^2) memory explosion for large k
    k = len(vectors)
    min_d2 = np.inf
    chunk = 256
    for start in range(0, k, chunk):
        end = min(start + chunk, k)
        # For each row in [start, end), compute squared distances to all later rows
        sub = vectors[start:end]  # shape (chunk, d)
        for i_local in range(end - start):
            v = sub[i_local]
            # Compare with all rows after (start + i_local)
            idx = start + i_local
            later = vectors[idx + 1:]
            if len(later) == 0:
                continue
            d2 = np.sum((later - v) ** 2, axis=1)
            local_min = np.min(d2)
            if local_min < min_d2:
                min_d2 = local_min
    return min_d2


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock limit

    # Start with the seed: all vectors with exactly two non-zero entries of +/-1
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)

    best_vectors = np.array(vectors, dtype=np.int64)

    # Try to add more vectors from the set of integer vectors with small norm
    # We'll consider vectors with entries in {-2, -1, 0, 1, 2} and norm <= 4
    # But generating all such is expensive, so we use a structured approach.

    # Generate candidate vectors by considering support sizes and value patterns
    candidates = set()
    for v in vectors:
        candidates.add(tuple(v))

    # Add vectors with three non-zero coordinates
    # For each triple of coordinates, try combinations of +/-1
    # This gives C(11,3) * 8 = 165 * 8 = 1320 additional candidates
    for triple in itertools.combinations(range(D), 3):
        i, j, k = triple
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.add(tuple(v))

    # Add vectors with one non-zero coordinate of value 2
    for i in range(D):
        for val in (2, -2):
            v = [0] * D
            v[i] = val
            candidates.add(tuple(v))

    # Add vectors with two non-zero coordinates where one is 2 and other is +/-1
    for i, j in itertools.combinations(range(D), 2):
        for si in (2, -2):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.add(tuple(v))
                v2 = [0] * D
                v2[i], v2[j] = sj, si
                candidates.add(tuple(v2))

    # Convert to list and try to greedily add candidates that maintain the constraint
    cand_list = [np.array(c, dtype=np.int64) for c in candidates]

    # Sort candidates by norm squared (ascending) to prefer lower-norm vectors
    norms2 = np.sum(np.array(cand_list) ** 2, axis=1)
    order = np.argsort(norms2)
    cand_list = [cand_list[i] for i in order]

    current = list(best_vectors)
    current_norm2 = np.sum(np.array(current) ** 2, axis=1)
    max_norm2 = np.max(current_norm2) if len(current) > 0 else 0
    min_d2 = _min_pairwise_dist2(np.array(current)) if len(current) >= 2 else np.inf

    for cv in cand_list:
        if time.time() > deadline:
            break
        cv_norm2 = int(np.sum(cv ** 2))
        # Check if adding this vector still satisfies min_d2 >= max_norm
        # New max norm squared = max(old_max_norm2, cv_norm2)
        new_max_norm2 = max(int(max_norm2), cv_norm2)
        
        # Check distance from cv to all current vectors
        # d2 >= new_max_norm2 required for all pairs
        if len(current) > 0:
            cur_arr = np.array(current)
            d2s = np.sum((cur_arr - cv) ** 2, axis=1)
            if np.min(d2s) < new_max_norm2:
                continue
            new_min_d2 = min(int(min_d2), int(np.min(d2s)))
        else:
            new_min_d2 = np.inf

        if new_min_d2 >= new_max_norm2:
            current.append(cv)
            max_norm2 = new_max_norm2
            min_d2 = new_min_d2

    if len(current) > len(best_vectors):
        best_vectors = np.array(current, dtype=np.int64)

    # Verify final result
    if len(best_vectors) >= 2:
        final_min_d2 = _min_pairwise_dist2(best_vectors)
        final_max_norm2 = np.max(np.sum(best_vectors ** 2, axis=1))
        if final_min_d2 < final_max_norm2:
            # Fall back to seed
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
            best_vectors = np.array(vectors, dtype=np.int64)

    return best_vectors
# EVOLVE-BLOCK-END
