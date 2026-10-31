import numpy as np
import itertools
import time

def construct():
    D = 11
    deadline = time.time() + 140  # leave 10s safety margin

    # Generate weight-2 vectors: exactly two non-zero entries (±1), rest 0.
    # norm^2 = 2 for each.
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)

    # For weight-2 vectors, norm^2 = 2.
    # Distance squared between v_i and v_j: ||v_i - v_j||^2 = 4 - 2 * v_i · v_j
    # We need ||v_i - v_j|| >= ||v_i|| = sqrt(2), so ||v_i - v_j||^2 >= 2
    # 4 - 2*dot >= 2  =>  dot <= 1
    # Two weight-2 vectors conflict if their dot product > 1.
    # Max dot product between two weight-2 vectors is 2 (same positions, same signs).
    # Dot = 2: identical vectors (only if we had duplicates, which we don't)
    # Dot = 1: share exactly one coordinate with same sign
    # Dot = 0: share one coordinate with opposite signs, or no shared coordinates
    # Dot = -1: share one coordinate, both opposite signs... wait, let's think.
    
    # v_i has entries at positions (a,b) with signs (sa, sb)
    # v_j has entries at positions (c,d) with signs (sc, sd)
    # dot = sa*sc if a==c or a==d, plus sb*sd if b==c or b==d
    # They share at most 2 positions (if {a,b} = {c,d}).
    # If they share 2 positions: dot = sa*sc' + sb*sd' where sc',sd' are signs at those positions
    #   e.g. v_i = (1,1) at (0,1), v_j = (1,1) at (0,1): dot = 1+1 = 2 (identical, not in our set)
    #   v_i = (1,1) at (0,1), v_j = (1,-1) at (0,1): dot = 1-1 = 0
    #   v_i = (1,1) at (0,1), v_j = (-1,1) at (0,1): dot = -1+1 = 0
    #   v_i = (1,1) at (0,1), v_j = (-1,-1) at (0,1): dot = -1-1 = -2
    # If they share 1 position: dot = ±1
    # If they share 0 positions: dot = 0
    
    # So conflict (dot > 1) only happens when dot = 2, which means identical vectors.
    # Since all 220 vectors are distinct, NO conflicts!
    
    # Let's verify: compute dot products to be safe.
    if n <= 300:
        dots = candidates @ candidates.T
        conflict = (dots > 1)
        np.fill_diagonal(conflict, False)
        if not np.any(conflict):
            # All 220 vectors are mutually compatible
            return candidates

    # If there are conflicts (shouldn't be), fall back to greedy selection
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = norms_sq.max()
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)

    best_selected = []
    best_size = 0
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        order = np.random.permutation(n)
        selected = []
        for idx in order:
            if not np.any(conflict[idx, selected]):
                selected.append(idx)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]

    return candidates[best_selected]
