"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11
import time

# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s (or 150s) hard limit

    def time_left():
        return deadline - time.time()

    # Seed: all weight-2 integer vectors
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)

    base = np.array(vectors, dtype=np.int64)
    k = base.shape[0]

    # Precompute norms
    norms_sq = np.sum(base ** 2, axis=1).astype(np.int64)
    max_norm_sq = int(np.max(norms_sq))
    # max_norm_sq = 2 for weight-2 vectors

    # Greedy: try to add weight-3 vectors
    # Constraint: for new vector w (norm^2=3), we need ||v - w||^2 >= max_norm_sq for all existing v
    # ||v-w||^2 = ||v||^2 + ||w||^2 - 2*v.w >= max_norm_sq
    # ||v||^2 + 3 - 2*v.w >= 2  =>  v.w <= (||v||^2 + 1)/2
    # For weight-2 v: v.w <= (2+1)/2 = 1.5 => v.w <= 1
    # For weight-1 v (if any): v.w <= (1+1)/2 = 1 => v.w <= 1

    # Also need ||w - w'||^2 >= max_norm_sq for all other weight-3 w'
    # ||w-w'||^2 = 3 + 3 - 2*w.w' = 6 - 2*w.w' >= 2 => w.w' <= 2

    # Generate all weight-3 candidates
    candidates = []
    for combo in itertools.combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[combo[0]] = s1
                    v[combo[1]] = s2
                    v[combo[2]] = s3
                    candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)

    # Greedy selection of weight-3 vectors
    added = []
    added_set = set()

    # For efficiency, precompute dot products between candidates and base
    # But n_cand = 1320, k = 220, so 1320*220 = 290400 dots - fast
    # And pairwise among candidates: 1320^2/2 = ~871200 - also fast

    # Greedy: pick candidate with maximum margin (minimum over constraints of slack)
    # Actually, simpler: just try all candidates in order, add if feasible
    # Better: randomize order and try to pack as many as possible

    np.random.seed(42)
    order = np.random.permutation(n_cand)

    # Track current set of added weight-3 vectors
    w3_vectors = []
    w3_set = set()

    # Precompute: for each candidate, check against base
    # base @ candidate^T gives dot products
    # We need max over base of dot <= 1

    # For pairwise among weight-3: need dot <= 2

    for idx in order:
        if time_left() < 5.0:
            break

        w = candidates[idx]
        w_tuple = tuple(w)
        if w_tuple in w3_set:
            continue

        # Check against base: dot product <= 1 for all base vectors
        dots_base = base @ w  # shape (k,)
        if np.any(dots_base > 1):
            continue

        # Check against existing weight-3: dot product <= 2
        if w3_vectors:
            w3_arr = np.array(w3_vectors, dtype=np.int64)
            dots_w3 = w3_arr @ w
            if np.any(dots_w3 > 2):
                continue

        w3_vectors.append(w)
        w3_set.add(w_tuple)

    # Combine
    if w3_vectors:
        w3_arr = np.array(w3_vectors, dtype=np.int64)
        result = np.vstack([base, w3_arr])
    else:
        result = base

    # Verify
    if result.shape[0] > 1:
        norms_sq = np.sum(result ** 2, axis=1).astype(np.int64)
        max_n2 = int(np.max(norms_sq))
        # Compute min pairwise distance squared
        # For large sets this could be expensive, but k is at most ~220+something
        # Use vectorized computation in chunks
        n = result.shape[0]
        min_d2 = max_n2  # initialize to max possible
        # Actually we need min over i<j of ||v_i - v_j||^2
        # = min over i<j of ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
        # Let's compute the Gram matrix
        if n <= 500:
            G = result @ result.T  # Gram matrix
            # d2[i,j] = norms_sq[i] + norms_sq[j] - 2*G[i,j]
            d2 = norms_sq[:, None] + norms_sq[None, :] - 2 * G
            # Set diagonal to infinity
            np.fill_diagonal(d2, np.inf)
            min_d2 = int(np.min(d2))
            if min_d2 < max_n2:
                # Invalid - fall back to base
                result = base

    return result
# EVOLVE-BLOCK-END
