"""Constructing kissing configuration in dimension 11 via D11 lattice minimal vectors."""
import numpy as np
import itertools
import time

D = 11


def construct():
    # D11 lattice: integer vectors with even coordinate sum.
    # Shortest vectors have squared norm 4:
    #   - Weight 4: entries in {+1, -1}, exactly 4 non-zeros (sum is even since 4 ones/-ones
    #     with even number of negatives gives even sum; actually any 4 signs give even sum
    #     because 4 is even, so sum of 4 entries each ±1 is even)
    #   - Weight 2: entries {+2, -2} or {+1, -1}? No - weight 1 with value ±2 gives norm 2.
    #   - Weight 1: one coordinate is ±2, rest 0. Sum = ±2, which is even. Norm = 2.
    #   - Weight 2: two coordinates, each ±1. Sum = ±2 or 0, both even. Norm = sqrt(2).
    #     Wait, norm squared = 1+1 = 2, not 4. So these are NOT shortest if we want norm 2.
    #
    # Let me reconsider. The task says "minimum pairwise distance is at least the maximum vector norm."
    # So we need: for all pairs i≠j, ||v_i - v_j|| >= max_k ||v_k||.
    #
    # For D11 minimal vectors of norm 2 (squared norm 4):
    #   Type A: one coordinate ±2, rest 0. There are 2*11 = 22 such vectors.
    #   Type B: four coordinates ±1, rest 0. There are C(11,4)*2^4 = 330*16 = 5280 such vectors.
    #   But wait - for Type B, the sum of 4 signs each ±1: the sum is even (since 4 is even),
    #   so all are in D11. Good.
    #
    # Actually, let me reconsider what gives the largest kissing configuration.
    # The kissing number in 11D is 512 (from the Leech lattice construction / known results).
    #
    # The standard construction for k=512 in 11D uses:
    #   - 22 vectors of type (±2, 0, ..., 0)
    #   - 480 vectors of type with four non-zero entries ±1
    #
    # Total: 22 + 480 = 502. Hmm, not 512.
    #
    # Let me reconsider. The kissing number of 11 is actually 512 per some references,
    # but the known exact value might be different. Let me just construct a valid large set.
    #
    # Key constraint: min pairwise distance >= max norm.
    # If all vectors have the same norm N, then we need ||v_i - v_j|| >= N for all i≠j.
    #
    # For vectors of norm 2 (squared norm 4):
    #   ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j = 4 + 4 - 2*v_i·v_j = 8 - 2*v_i·v_j
    #   We need 8 - 2*v_i·v_j >= 4, i.e., v_i·v_j <= 2.
    #
    # For Type A vectors (±2 at position p): v·v' = 4 if same position same sign, -4 if same pos diff sign, 0 otherwise.
    #   So two Type A vectors at different positions have dot product 0. OK.
    #   Two Type A vectors at same position: dot = ±4. If +4, that violates. So we can only use one sign per position.
    #   Actually, we can use both +2 and -2 at each position? No: +2·+2=4 > 2. Violation.
    #   So for Type A, we pick one sign per position. That gives 11 vectors.
    #
    # For Type A and Type B: v_A has ±2 at position p. v_B has ±1 at 4 positions including possibly p.
    #   Dot product = (±2)*(±1) if p in B's support, else 0. So dot = ±2. OK, |dot|=2 <= 2.
    #
    # For two Type B vectors: dot product depends on overlap of supports and sign agreement.
    #   If supports are disjoint: dot = 0. OK.
    #   If supports overlap in 1 element: dot = ±1. OK.
    #   If supports overlap in 2 elements: dot = ±2 or 0. OK.
    #   If supports overlap in 3 elements: dot = ±3 or ±1. If ±3, violation!
    #   If supports overlap in 4 elements (same support): dot = ±4, ±2, or 0. If ±4, violation.
    #
    # So we need: no two Type B vectors share 3 or more positions with same-sign product summing to ±3.
    #   More precisely, if they share k positions, the dot product is the sum of sign products over those k positions.
    #   For k=3: possible dot products are -3, -1, 1, 3. We need dot <= 2, so we must avoid dot = 3.
    #   For k=4: possible dot products are -4, -2, 0, 2, 4. We must avoid dot = 4.
    #
    # This is getting complex. Let me just use the weight-4 construction with greedy support selection
    # as in the original, but also add the 22 weight-1 vectors (±2 at single position) with appropriate signs.
    #
    # Actually, the simplest large valid configuration: use ONLY weight-4 vectors with carefully chosen supports.
    # The original program found 32 supports * 16 sign variations = 512 vectors. That scored 1.89.
    #
    # Let me try a different approach: use the full D11 minimal vector set but filter for validity.
    # Generate all norm-2 vectors in D11, then greedily select a valid subset.

    start_time = time.time()
    deadline = start_time + 110

    # Generate all candidate vectors of norm 2 (squared norm 4) in D11
    candidates = []

    # Type A: single coordinate ±2
    for pos in range(D):
        for sign in [2, -2]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = sign
            candidates.append(v)

    # Type B: four coordinates ±1
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(support, signs):
                v[pos] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    print(f"Total candidates: {n_cand}")

    # Greedy selection: add vectors that maintain the constraint
    # min pairwise distance >= max norm (which is 2 for all these vectors)
    # So we need ||v_i - v_j||^2 >= 4 for all selected pairs, i.e., v_i·v_j <= 2

    selected = []
    selected_arr = []

    for idx in range(n_cand):
        if time.time() > deadline:
            break

        v = candidates[idx]
        valid = True

        if selected_arr:
            # Compute dot products with all selected vectors
            dots = selected_arr @ v
            if np.any(dots > 2):
                valid = False

        if valid:
            selected.append(idx)
            selected_arr = np.vstack([selected_arr, v.reshape(1, -1)]) if selected_arr else v.reshape(1, -1)

    print(f"Selected {len(selected)} vectors")

    if len(selected) == 0:
        # Fallback
        result = np.zeros((1, D), dtype=np.int64)
        result[0, 0] = 2
        return result

    result = np.array([candidates[i] for i in selected], dtype=np.int64)
    return result
