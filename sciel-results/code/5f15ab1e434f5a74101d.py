"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Start with the seed vectors: pairs (i,j) with entries +-1
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(np.array(v, dtype=np.int64))

    vectors = np.array(vectors, dtype=np.int64)
    # Remove duplicates just in case
    vectors = np.unique(vectors, axis=0)

    # Current minimum pairwise distance and max norm
    def compute_min_dist_and_max_norm(vecs):
        if len(vecs) < 2:
            return float('inf'), np.max(np.linalg.norm(vecs, axis=1)) if len(vecs) > 0 else 0
        # Compute pairwise distances efficiently in chunks to avoid memory issues
        min_dist = float('inf')
        max_norm = np.max(np.linalg.norm(vecs, axis=1))
        n = len(vecs)
        chunk_size = 2000
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk = vecs[i:end]
            # distances from chunk to all vectors
            # Use broadcasting but in sub-chunks if needed
            for j in range(i, n, chunk_size):
                end2 = min(j + chunk_size, n)
                chunk2 = vecs[j:end2]
                # pairwise distances between chunk and chunk2
                # Only consider pairs where first index < second index to avoid duplicates
                diff = chunk[:, np.newaxis, :] - chunk2[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                # We need to be careful: this computes all pairs between chunk and chunk2
                # But we want min over all i<j
                if i == j:
                    # upper triangle of this sub-block
                    n_sub = len(chunk)
                    mask = np.triu(np.ones((n_sub, n_sub), dtype=bool), k=1)
                    if mask.any():
                        min_dist = min(min_dist, float(np.min(dists[mask])))
                else:
                    # all pairs are valid since chunk indices < chunk2 indices
                    min_dist = min(min_dist, float(np.min(dists)))
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        return min_dist, max_norm

    min_dist, max_norm = compute_min_dist_and_max_norm(vectors)
    best_count = len(vectors)
    best_vectors = vectors.copy()

    # Try to add new vectors
    # Candidate vectors: integer vectors with small entries
    # Try vectors with entries in {-1, 0, 1} and possibly {-2, -1, 0, 1, 2}
    # But that's too many. Let's be smart.

    # Strategy: try adding vectors of norm 2 (like the seed) or norm sqrt(3), etc.
    # The constraint: for any new vector v, and any existing vector u:
    #   ||v - u|| >= max_norm (which may increase if we add a vector with larger norm)
    # Also ||v + u|| >= max_norm (since distance between v and -u... wait, -u might not be in set)
    # Actually the condition is: for all pairs (a, b) in the set, ||a - b|| >= max_norm
    # where max_norm = max_{x in set} ||x||

    # Let's try to add vectors with the same norm (sqrt(2)) first, then larger norms.

    # Generate candidate vectors: all integer vectors with entries in {-1, 0, 1}
    # and weight (number of nonzero entries) 2 or 3, to keep norm small
    # Weight 2: norm sqrt(2), same as seed
    # Weight 3: norm sqrt(3)

    # But there are many. Let's use a greedy approach:
    # For each candidate, check if it can be added.

    # First, let's try all weight-3 vectors (norm sqrt(3))
    # Number of weight-3 vectors: C(11,3) * 2^3 = 165 * 8 = 1320
    # That's manageable.

    candidates = []

    # Weight 3 vectors
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i], v[j], v[k] = si, sj, sk
                            candidates.append(np.array(v, dtype=np.int64))

    # Also try weight-4 vectors? C(11,4)*2^4 = 330*16 = 5280. Maybe too many.
    # Let's first try weight 3.

    # Sort candidates by some heuristic: maybe by norm (all same for weight 3)
    # Just iterate through them.

    # We need to maintain the constraint. Let's be more efficient:
    # For a new candidate v with norm nv, the new max_norm is max(max_norm, nv)
    # For all existing u, we need ||v - u|| >= new_max_norm
    # Also for all existing u, ||u - v|| >= new_max_norm (same thing)

    # Let's convert vectors to a list for easier management
    vec_list = [vectors[i] for i in range(len(vectors))]

    # Precompute norms
    norms = np.array([np.linalg.norm(v) for v in vec_list])
    current_max_norm = np.max(norms)

    added = 0
    for cand in candidates:
        if time.time() > deadline:
            break

        cand_norm = np.linalg.norm(cand)
        new_max_norm = max(current_max_norm, cand_norm)

        # Check distance to all existing vectors
        # We need ||cand - u|| >= new_max_norm for all u in vec_list
        # This is equivalent to (cand - u)^T (cand - u) >= new_max_norm^2

        # Efficient check: compute min distance
        diff = np.array(vec_list) - cand[np.newaxis, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        min_d = np.min(dists)

        if min_d >= new_max_norm - 1e-9:
            vec_list.append(cand)
            current_max_norm = new_max_norm
            added += 1
            # Update best if we have more vectors
            # But we need to verify the entire set is valid, which we've been doing incrementally

    # Build final array
    if vec_list:
        final_vectors = np.array(vec_list, dtype=np.int64)
    else:
        final_vectors = vectors

    best_count = len(final_vectors)

    return final_vectors
# EVOLVE-BLOCK-END
