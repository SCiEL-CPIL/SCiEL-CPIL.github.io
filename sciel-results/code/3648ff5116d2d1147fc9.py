"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import random


def _generate_candidates(max_norm_sq, max_nonzero=4):
    """Generate nonzero integer vectors in Z^D with sum of squares <= max_norm_sq
    and at most max_nonzero nonzero entries."""
    candidates = []
    # Precompute squared norms for small integers
    # For max_norm_sq up to maybe 10, entries are -3..3
    max_entry = int(np.sqrt(max_norm_sq)) + 1

    # Generate all possible support sets (subsets of indices)
    indices = list(range(D))
    for r in range(1, max_nonzero + 1):
        for support in itertools.combinations(indices, r):
            # For each support, generate all assignments of values to these positions
            # such that sum of squares <= max_norm_sq
            # Values are nonzero integers with |v| <= max_entry
            vals_range = list(range(-max_entry, max_entry + 1))
            vals_range.remove(0)

            for vals in itertools.product(vals_range, repeat=r):
                norm_sq = sum(v * v for v in vals)
                if norm_sq > max_norm_sq:
                    continue
                if norm_sq == 0:
                    continue
                vec = [0] * D
                for idx, val in zip(support, vals):
                    vec[idx] = val
                # Avoid duplicates: normalize by first nonzero sign?
                # Actually, we want both v and -v potentially, but they are distinct vectors.
                # However, for kissing number, we typically consider lines? No, vectors.
                # But note: if v is in the set, -v might also be valid.
                # We just add it.
                candidates.append(tuple(vec))

    # Deduplicate
    candidates = list(set(candidates))
    # Convert to numpy array for efficiency
    return np.array(candidates, dtype=np.int64)


def _check_validity(vectors):
    """Check if the set of vectors satisfies the condition:
    min pairwise distance >= max norm.
    Note: min pairwise distance is over distinct pairs.
    max norm is the maximum L2 norm among all vectors."""
    if len(vectors) == 0:
        return 0
    if len(vectors) == 1:
        # No pairwise distance, so condition is vacuously true?
        # Usually kissing number >= 1. But let's check the definition.
        # "minimum pairwise distance is at least the maximum vector norm"
        # If there's only one vector, there are no pairs. Is it valid?
        # Usually yes, unless k must be >= 2.
        # The seed returns 220, so k can be large.
        # Let's assume k=1 is valid, but score is k.
        return 1

    norms_sq = np.sum(vectors ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)

    # Compute all pairwise distances squared
    # dist_sq[i,j] = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j
    # We need min_{i!=j} dist_sq >= max_norm_sq

    # Efficient check:
    # For each pair, dist_sq >= max_norm_sq
    # This is equivalent to: for all i != j, ||v_i - v_j||^2 >= max_norm_sq

    n = len(vectors)
    # We can check by iterating pairs, but that's O(n^2).
    # With n up to a few hundred, this is fast.
    for i in range(n):
        for j in range(i + 1, n):
            diff = vectors[i] - vectors[j]
            dist_sq = np.sum(diff ** 2)
            if dist_sq < max_norm_sq:
                return 0

    return n


def _greedy_select(candidates, max_norm_sq, num_restarts=50, time_limit=120.0):
    """Greedy selection of a valid set from candidates.
    Returns the best set found."""
    best_count = 0
    best_set = None
    start_time = time.time()

    n_cands = len(candidates)
    if n_cands == 0:
        return np.array([], dtype=np.int64).reshape(0, D)

    # Precompute norms of candidates
    cand_norms_sq = np.sum(candidates ** 2, axis=1)

    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break

        # Random permutation of candidates
        perm = np.random.permutation(n_cands)
        current_set = []
        current_indices = []

        for idx in perm:
            v = candidates[idx]
            v_norm_sq = cand_norms_sq[idx]
            if v_norm_sq > max_norm_sq:
                continue

            # Check compatibility with current set
            compatible = True
            for j in current_indices:
                u = candidates[j]
                # New max norm squared
                new_max_norm_sq = max(max_norm_sq, v_norm_sq, cand_norms_sq[j])
                # Actually, the condition is min pairwise distance >= CURRENT max norm.
                # Wait, the condition is: min pairwise distance >= max vector norm IN THE SET.
                # So as we add vectors, the max norm might increase.
                # If max norm increases, previous pairs might become invalid?
                # No, the condition is global.
                # If we add v, the new set S' = S U {v}.
                # The max norm of S' is max(max_norm_S, ||v||^2).
                # We need min pairwise distance in S' >= max_norm_S'.
                #
                # Let R_S = max norm in S.
                # Let R_S' = max(R_S, ||v||^2).
                # We know min pairwise distance in S >= R_S.
                # We need:
                # 1. min pairwise distance in S >= R_S'
                # 2. min distance from v to S >= R_S'
                #
                # If ||v||^2 > R_S, then R_S' = ||v||^2.
                # Then we need min pairwise distance in S >= ||v||^2.
                # This might not hold!
                # So we cannot just greedily add if compatible with current max norm.
                # We must check if adding v breaks the condition for existing pairs.
                #
                # This makes greedy harder.
                # Alternative: fix R = max_norm_sq.
                # Only select vectors with norm <= R.
                # Then max norm of set is <= R.
                # We need min pairwise distance >= R.
                # This is a fixed constraint!
                # So if we fix R, the condition is:
                # All vectors have norm <= R.
                # All pairwise distances >= R.
                #
                # This is much simpler.
                # So for a fixed R, we are looking for a set S such that:
                # ||v||^2 <= R for all v in S
                # ||u - v||^2 >= R for all u, v in S, u != v
                #
                # So in the greedy loop, we don't need to update max norm.
                # We just check:
                # 1. ||v||^2 <= R
                # 2. For all u in S, ||v - u||^2 >= R

                dist_sq = np.sum((v - u) ** 2)
                if dist_sq < max_norm_sq:
                    compatible = False
                    break

            if compatible:
                current_set.append(v)
                current_indices.append(idx)

        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = np.array(current_set, dtype=np.int64)

    return best_set


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    best_result = None
    best_score = 0

    # Try different max_norm_sq values
    # Start with small values, which allow more vectors potentially?
    # Actually, smaller R means stricter distance constraint (dist >= R) but looser norm constraint (norm <= R).
    # Larger R means looser distance constraint but stricter norm constraint.
    #
    # For R=1:
    # Vectors with norm^2 <= 1 => only +-e_i (22 vectors).
    # Distance between e_i and e_j is sqrt(2) for i!=j, which is > 1.
    # Distance between e_i and -e_i is 2, which is > 1.
    # So all 22 are valid?
    # Let's check:
    # Set S = {e_i, -e_i for i=1..11}.
    # Max norm = 1.
    # Min pairwise distance:
    # ||e_i - e_j|| = sqrt(2) > 1
    # ||e_i - (-e_i)|| = 2 > 1
    # ||e_i - (-e_j)|| = sqrt(2) > 1
    # So yes, 22 is valid for R=1.
    #
    # For R=2:
    # Vectors with norm^2 <= 2.
    # These include e_i, -e_i, and e_i + e_j, e_i - e_j, etc.
    # Actually, e_i + e_j has norm^2 = 2.
    # Distance between e_i and e_j + e_k?
    # Let's just run the greedy.

    for max_norm_sq in [1, 2, 3, 4, 5]:
        if time.time() - start_time > time_limit - 10.0:
            break

        # Generate candidates
        # Limit nonzero entries to keep candidate set manageable
        # For R=1, max_nonzero=1
        # For R=2, max_nonzero=2
        # For R=3, max_nonzero=2 (since 1+1=2<=3, 3 is 1*3+0, but 3^2=9>3, so entries are -1,0,1 mostly)
        # Actually, for R=3, entries can be +-1, and maybe +-sqrt(3) not integer.
        # So entries are in {-1, 0, 1}.
        # Max nonzero entries:
        # If R=3, 1+1+1=3, so up to 3 nonzeros.
        max_nonzero = int(np.sqrt(max_norm_sq)) + 1

        candidates = _generate_candidates(max_norm_sq, max_nonzero)
        if len(candidates) == 0:
            continue

        # Run greedy
        # Number of restarts depends on time
        # Estimate time per restart
        # n_cands = len(candidates)
        # Greedy is O(n_cands * best_count * D)
        # If best_count is ~100, n_cands ~ 1000, D=11
        # 1000 * 100 * 11 = 1.1e6 ops per restart.
        # In Python, this might be ~0.1s.
        # So 100 restarts ~ 10s.
        num_restarts = 50

        result = _greedy_select(candidates, max_norm_sq, num_restarts=num_restarts, time_limit=time_limit)
        if result is not None and len(result) > 0:
            # Verify validity
            count = _check_validity(result)
            if count > best_score:
                best_score = count
                best_result = result

    if best_result is None:
        # Fallback to seed result
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_result = np.array(vectors, dtype=np.int64)
        best_score = len(best_result)

    return best_result

# EVOLVE-BLOCK-END
