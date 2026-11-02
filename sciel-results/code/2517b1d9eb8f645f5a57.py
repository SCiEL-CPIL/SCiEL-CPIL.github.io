"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # STRUCTURAL CHANGE: Use the D_11 root system (weight-2 vectors with entries in {+1,-1}).
    # All vectors have norm^2 = 2, so max_norm^2 = 2.
    # Condition: ||u-v||^2 >= 2 for all pairs.
    # ||u-v||^2 = 2 + 2 - 2<u,v> = 4 - 2<u,v> >= 2  =>  <u,v> <= 1.
    # For weight-2 vectors, <u,v> can be: 2 (same vector), 1 (share one coord, same sign),
    #   0 (disjoint or share one coord with opposite signs), -1, -2.
    # So the conflict condition is: <u,v> > 1, i.e., <u,v> == 2 (same vector, excluded by set).
    # Wait: if two distinct weight-2 vectors share one coordinate with the same sign, <u,v>=1.
    # If they share one coordinate with opposite signs, <u,v>=-1.
    # If disjoint, <u,v>=0.
    # So <u,v> <= 1 is automatically satisfied for all DISTINCT weight-2 vectors!
    # That means ALL 2*C(11,2)*4 = 2*55*4 = 440 weight-2 vectors form a valid set.
    #
    # But wait, we need to verify: the condition is min pairwise DISTANCE >= max NORM.
    # max norm = sqrt(2) (since all have norm^2=2).
    # min pairwise distance: for two vectors sharing one coordinate with same sign,
    #   ||u-v||^2 = 4 - 2(1) = 2, so ||u-v|| = sqrt(2) = max norm. OK, equality holds.
    # For disjoint: ||u-v||^2 = 4, ||u-v|| = 2 > sqrt(2). OK.
    # So all 440 D_11 roots work!
    #
    # Can we do better? The kissing number in R^11 is known to be at least 460.
    # Let's try to augment the D_11 set with additional vectors.
    #
    # Strategy: Start with all D_11 roots (440 vectors), then greedily add weight-4
    # vectors (norm^2=4) that are compatible. But mixing norms complicates things:
    # if we have a weight-2 vector (norm^2=2) and a weight-4 vector (norm^2=4),
    # max norm becomes 2 (norm^2=4). We need ||u-v||^2 >= 4 for all pairs.
    # For weight-2 u and weight-4 v: ||u-v||^2 = 2 + 4 - 2<u,v> = 6 - 2<u,v> >= 4 => <u,v> <= 1.
    # For two weight-4 vectors: ||u-v||^2 = 4+4-2<u,v> = 8-2<u,v> >= 4 => <u,v> <= 2.
    #
    # This mixed approach is complex. Let's first just return the 440 D_11 roots,
    # which should give a solid baseline. Then try to augment.

    # Generate all D_11 roots: weight-2 vectors with entries in {+1, -1}
    candidates = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    # All 440 D_11 roots are mutually compatible (verified above: <u,v> <= 1 for distinct)
    selected = candidates.copy()

    # Now try to augment with weight-4 vectors (norm^2 = 4, entries +/-1 in 4 positions)
    # When we add a weight-4 vector, max norm^2 becomes 4.
    # Conditions for a new weight-4 vector w against existing set:
    #   Against D_11 root u (norm^2=2): ||w-u||^2 = 4+2-2<w,u> >= 4 => <w,u> <= 1
    #   Against other weight-4 v (norm^2=4): ||w-v||^2 = 4+4-2<w,v> >= 4 => <w,v> <= 2
    #
    # Generate weight-4 candidates and greedily add compatible ones.
    w4_candidates = []
    for coords in combinations(range(D), 4):
        for signs in range(16):  # 2^4 sign choices
            v = np.zeros(D, dtype=np.int64)
            for k in range(4):
                if (signs >> k) & 1:
                    v[coords[k]] = -1
                else:
                    v[coords[k]] = 1
            w4_candidates.append(v)

    w4_candidates = np.array(w4_candidates, dtype=np.int64)

    # For efficiency, precompute inner products of D_11 roots with each w4 candidate.
    # But that's 440 * 1760 = ~774K inner products of length 11. Doable in batches.
    # Actually, let's just do it greedily with numpy.

    # selected currently has all 440 D_11 roots
    # We need to track which are D_11 (norm^2=2) and which are w4 (norm^2=4)
    # For now, all selected are D_11.

    selected_is_w4 = np.zeros(len(selected), dtype=bool)

    for idx in range(len(w4_candidates)):
        if time.time() > deadline:
            break
        w = w4_candidates[idx]
        # Check against all selected
        # Inner products with selected
        ips = selected @ w  # shape (len(selected),)

        # For D_11 selected (norm^2=2): need ip <= 1
        # For w4 selected (norm^2=4): need ip <= 2
        ok = np.ones(len(selected), dtype=bool)
        d11_mask = ~selected_is_w4
        w4_mask = selected_is_w4
        if np.any(d11_mask):
            ok[d11_mask] = ips[d11_mask] <= 1
        if np.any(w4_mask):
            ok[w4_mask] = ips[w4_mask] <= 2

        if np.all(ok):
            selected = np.vstack([selected, w.reshape(1, -1)])
            selected_is_w4 = np.append(selected_is_w4, True)

    result = selected.astype(np.int64)
    if len(result) == 0:
        # Fallback
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    vec = [0] * D
                    vec[i], vec[j] = si, sj
                    vectors.append(vec)
        result = np.array(vectors, dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
