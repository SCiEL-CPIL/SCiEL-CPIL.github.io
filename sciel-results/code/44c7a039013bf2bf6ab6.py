import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second margin before 150s kill

    # Generate candidate vectors with small integer coordinates.
    # We want vectors with norm^2 up to some bound so that the kissing
    # configuration (min pairwise distance >= max norm) can include many vectors.
    #
    # Key insight: for a kissing configuration, we need
    #   ||v_i - v_j||^2 >= max(||v_i||^2, ||v_j||^2)  for all i != j
    # If all vectors have the same norm^2 = N, this becomes
    #   ||v_i - v_j||^2 >= N  =>  2N - 2 v_i.v_j >= N  =>  v_i.v_j <= N/2
    #
    # For norm^2 = 2 (two 1s): v_i.v_j <= 1. Since each vector has two
    #   nonzero entries of magnitude 1, the dot product is at most 2 (if
    #   same positions) or 0 or -1. We need dot <= 1, so we can't have
    #   two vectors sharing both positions with same signs.
    #
    # For norm^2 = 4: v_i.v_j <= 2. More flexibility.
    #
    # Strategy: generate all vectors with small coordinates, then do a
    #   fast greedy selection using a pre-allocated array and a boolean
    #   "occupied" mask to avoid repeated vstack.

    candidates = []

    def add_candidate(v):
        if any(x != 0 for x in v):
            candidates.append(v)

    # Norm^2 = 2: two 1s
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                add_candidate(v)

    # Norm^2 = 3: three 1s
    for i, j, k in itertools.combinations(range(D), 3):
        for s in itertools.product([1, -1], repeat=3):
            v = [0] * D
            v[i], v[j], v[k] = s
            add_candidate(v)

    # Norm^2 = 4:
    # Type 1: four 1s
    for i, j, k, l in itertools.combinations(range(D), 4):
        for s in itertools.product([1, -1], repeat=4):
            v = [0] * D
            v[i], v[j], v[k], v[l] = s
            add_candidate(v)
    # Type 2: one 2, two 1s (norm^2 = 4+1+1 = 6, skip for now)
    # Type 3: two 2s (norm^2 = 4+4 = 8, skip)
    # Type 4: one 2, one 0... no.
    # Actually norm^2=4 with two 2s: 4+4=8, not 4.
    # One 4: 16, not 4.
    # So norm^2=4 is only: four 1s, or one 2 and... 4 = 4, so one 2 and rest 0.
    for i in range(D):
        for si in (2, -2):
            v = [0] * D
            v[i] = si
            add_candidate(v)

    # Norm^2 = 5: one 2, one 1 (4+1=5)
    for i, j in itertools.combinations(range(D), 2):
        for si in (2, -2):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                add_candidate(v)
    # Norm^2 = 5: five 1s
    for i, j, k, l, m in itertools.combinations(range(D), 5):
        for s in itertools.product([1, -1], repeat=5):
            v = [0] * D
            v[i], v[j], v[k], v[l], v[m] = s
            add_candidate(v)

    # Norm^2 = 6: two 2s, one 1... 4+4+1=9 no.
    # Norm^2 = 6: one 2, two 1s: 4+1+1=6
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (2, -2):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    add_candidate(v)

    # Convert to NumPy array
    candidates = np.array(candidates, dtype=np.int64)

    # Sort by norm^2 ascending
    norms_sq = np.sum(candidates * candidates, axis=1)
    sort_idx = np.argsort(norms_sq)
    candidates = candidates[sort_idx]
    norms_sq = norms_sq[sort_idx]

    n_candidates = len(candidates)

    # Pre-allocate selected array (we'll grow it as needed)
    # Start with a reasonable capacity
    capacity = min(n_candidates, 5000)
    selected = np.empty((capacity, D), dtype=np.int64)
    selected_count = 0
    max_norm_sq = 0

    # Pre-compute norms_sq for selected vectors as we go
    # We'll maintain a running list of selected norms_sq

    # For fast checking, we maintain the selected vectors in a contiguous
    # array and compute dot products in batch.

    for i in range(n_candidates):
        if time.time() > deadline:
            break

        v = candidates[i]
        v_norm_sq = norms_sq[i]

        if selected_count == 0:
            selected[0] = v
            selected_count = 1
            max_norm_sq = v_norm_sq
            continue

        threshold = max(max_norm_sq, v_norm_sq)

        # Compute distances to all selected vectors
        # dist^2 = ||v||^2 + ||s||^2 - 2 v.s
        # We need dist^2 >= threshold for all selected s
        # => 2 v.s <= ||v||^2 + ||s||^2 - threshold

        # Get selected vectors
        sel = selected[:selected_count]
        dot_products = sel @ v  # shape (selected_count,)

        # s_norms_sq: we can precompute and store
        # But to avoid recomputing, let's just compute
        s_norms_sq = np.sum(sel * sel, axis=1)

        dist_sqs = v_norm_sq + s_norms_sq - 2 * dot_products
        min_dist_sq = np.min(dist_sqs)

        if min_dist_sq >= threshold:
            if selected_count >= capacity:
                # Grow the array
                new_capacity = capacity * 2
                new_selected = np.empty((new_capacity, D), dtype=np.int64)
                new_selected[:selected_count] = selected[:selected_count]
                selected = new_selected
                capacity = new_capacity

            selected[selected_count] = v
            selected_count += 1
            if v_norm_sq > max_norm_sq:
                max_norm_sq = v_norm_sq

    # Return the selected vectors as a (k, D) array
    if selected_count == 0:
        # Fallback: return a single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    return selected[:selected_count].copy()

# EVOLVE-BLOCK-END
