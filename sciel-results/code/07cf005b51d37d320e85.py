"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def _pairwise_min_dist_sq(vectors):
    """Return the minimum squared pairwise distance among rows of vectors (int64)."""
    if len(vectors) < 2:
        return np.iinfo(np.int64).max
    # Compute all pairwise squared differences efficiently in blocks to limit memory
    n = len(vectors)
    # We can compute squared distances via ||a||^2 + ||b||^2 - 2 a.b
    norms_sq = np.einsum('ij,ij->i', vectors, vectors)  # shape (n,)
    G = vectors @ vectors.T  # shape (n, n)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
    np.fill_diagonal(dist_sq, np.iinfo(np.int64).max)
    return int(dist_sq.min())


def _max_norm_sq(vectors):
    if len(vectors) == 0:
        return 0
    norms_sq = np.einsum('ij,ij->i', vectors, vectors)
    return int(norms_sq.max())


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Start with the seed: all vectors with exactly two nonzero entries ±1
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                base.append(v)
    vectors = np.array(base, dtype=np.int64)

    # Also add single-unit vectors? Their norm is 1, which is less than max norm (2 for base).
    # But they might help if we can add more. Let's first check if we can add more 2-norm vectors.

    # Try to greedily add vectors of the form: two coordinates are ±1, possibly with additional ±1 in other coords?
    # Actually, the constraint is: min pairwise dist >= max norm.
    # If all vectors have norm^2 = 2, then max norm^2 = 2, so we need min pairwise dist^2 >= 2.
    # Pairwise dist^2 = ||a-b||^2. For two vectors with entries in {-1,0,1}, this can be computed.

    # Let's try a greedy approach: generate candidate vectors (all with entries in {-1, 0, 1} and norm^2 = 2)
    # and add them if they maintain the constraint.

    # Generate all candidates with exactly two nonzeros ±1
    candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)

    # Shuffle for deterministic but varied order? Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    idx = np.arange(len(candidates))
    rng.shuffle(idx)
    candidates = [candidates[i] for i in idx]

    # Start with empty set, greedily add
    chosen = []
    for c in candidates:
        if time.time() > deadline - 10.0:
            break
        # Check if adding c maintains min dist >= max norm
        # Current max norm^2 is 2 (all have norm^2=2)
        # Need dist(c, v)^2 >= 2 for all v in chosen
        if not chosen:
            chosen.append(c)
            continue
        chosen_arr = np.array(chosen, dtype=np.int64)
        # Compute distances from c to all chosen
        diff = chosen_arr - c
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        if dist_sq.min() >= 2:
            chosen.append(c)

    vectors = np.array(chosen, dtype=np.int64) if chosen else np.array(base, dtype=np.int64)

    return vectors

# EVOLVE-BLOCK-END
