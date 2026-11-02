"""Kissing number in dimension 11 - find max set of integer vectors with min dist >= max norm."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Generate all weight-2 vectors: exactly two non-zero entries, each ±1
    # For symmetry breaking: we'll generate all, but use a canonical ordering
    weight2 = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    weight2.append(v)
    weight2 = [np.array(v, dtype=np.int64) for v in weight2]

    # Generate weight-3 vectors: exactly three non-zero entries, each ±1
    weight3 = []
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            weight3.append(v)
    weight3 = [np.array(v, dtype=np.int64) for v in weight3]

    # Combine all candidates
    all_candidates = weight2 + weight3
    n_cand = len(all_candidates)

    # Precompute: for each candidate, its norm squared
    norms_sq = np.array([np.dot(v, v) for v in all_candidates], dtype=np.int64)

    # Precompute pairwise distance matrix (squared distances)
    # This is expensive: n_cand * n_cand * D operations
    # n_cand = 110 (weight2) + 220 (weight3) = 330
    # 330^2 * 11 = 1.2M operations - very fast

    # Build distance matrix
    dist_mat = np.zeros((n_cand, n_cand), dtype=np.int64)
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            diff = all_candidates[i] - all_candidates[j]
            d2 = int(np.dot(diff, diff))
            dist_mat[i, j] = d2
            dist_mat[j, i] = d2

    # For a set S, the constraint is: for all u,v in S, |u-v|^2 >= max(|u|^2, |v|^2)
    # If we use only weight-2 vectors (norm_sq=2), the constraint becomes:
    #   |u-v|^2 >= 2, i.e., u·v <= 0
    # If we mix weight-2 and weight-3, the max norm_sq is 3, so:
    #   |u-v|^2 >= 3 for ALL pairs (since max is 3)
    # This is stricter! So mixing makes it harder.
    #
    # Better strategy: use ONLY weight-2 vectors with the constraint |u-v|^2 >= 2
    # This is equivalent to u·v <= 0 for weight-2 vectors.
    #
    # Two weight-2 vectors u,v have u·v = 0 if they share no coordinates,
    # u·v = ±1 if they share exactly one coordinate.
    # So u·v <= 0 means they must NOT share a coordinate with the same sign,
    # OR share no coordinates.
    #
    # Actually: u·v = (sum of products at shared positions). If they share 1 position,
    # u·v = ±1. If they share 0 positions, u·v = 0. If they share 2 positions,
    # u·v = ±1 ± 1 which could be 0, ±2.
    #
    # For the seed (all 110 weight-2 vectors), the max pairwise "closeness" is
    # when two vectors share one coordinate: dist_sq = 2 - 2*(±1) = 0 or 4.
    # Wait, that's wrong. Let me recompute.
    #
    # u = e_i + e_j, v = e_i + e_k (share coordinate i)
    # u - v = e_j - e_k, |u-v|^2 = 2. max_norm = 2. So 2 >= 2. OK!
    #
    # u = e_i + e_j, v = e_i - e_j (share both)
    # u - v = 2e_i, |u-v|^2 = 4. OK!
    #
    # u = e_i + e_j, v = -e_i - e_j (opposite)
    # u - v = 2(e_i + e_j), |u-v|^2 = 8. OK!
    #
    # u = e_i + e_j, v = e_k + e_l (disjoint)
    # u - v = e_i + e_j - e_k - e_l, |u-v|^2 = 4. OK!
    #
    # So actually ALL pairs of weight-2 vectors satisfy |u-v|^2 >= 2!
    # The seed of 110 is already valid with max_norm_sq = 2.
    #
    # Can we add weight-3 vectors? If we add a weight-3 vector w (norm_sq=3),
    # then max_norm_sq becomes 3, and ALL pairs must have dist_sq >= 3.
    # For a weight-2 vector u and weight-3 vector w:
    # |u-w|^2 = |u|^2 + |w|^2 - 2u·w = 2 + 3 - 2u·w = 5 - 2u·w
    # Need 5 - 2u·w >= 3, i.e., u·w <= 1.
    #
    # For two weight-3 vectors w1, w2:
    # |w1-w2|^2 = 3 + 3 - 2w1·w2 = 6 - 2w1·w2 >= 3, i.e., w1·w2 <= 1.5, so w1·w2 <= 1.
    #
    # Strategy: Greedily select from all candidates with max_norm_sq = 3 constraint.
    # Use min-degree greedy: at each step, pick the candidate that has the fewest
    # conflicts with already-selected vectors (or equivalently, the one that can
    # coexist with the most remaining candidates).

    # For efficiency, precompute which pairs are "conflicting" under max_norm_sq=3
    # Two candidates i, j conflict if dist_mat[i,j] < 3
    # (since max possible norm_sq in our pool is 3)

    # Actually, the constraint depends on the actual max norm in the selected set.
    # Let's just use max_norm_sq = 3 as the bound (conservative).

    # Conflict: dist_mat[i,j] < max_norm_sq
    # With max_norm_sq = 3: conflict if dist_sq < 3, i.e., dist_sq in {0, 1, 2}

    # Since all our vectors are distinct and non-zero, dist_sq = 0 never happens.
    # dist_sq = 1: can this happen? |u-v|^2 = 1 means they differ in one coordinate by 1.
    # For weight-2 vectors: e.g., e_1+e_2 vs e_1+e_2+e_3 - no, different weights.
    # e_1+e_2 vs e_1+e_2: same vector, dist=0.
    # e_1+e_2 vs e_1-e_2: dist_sq = 4.
    # e_1+e_2 vs e_1+e_3: dist_sq = 2.
    # e_1+e_2 vs -e_1+e_2: dist_sq = 4.
    # For weight-2 vs weight-2: min dist_sq is 2 (when they share one coordinate).
    # For weight-3 vs weight-3: min dist_sq can be 2 (share 2 coords, differ in 1).
    #   e.g., e_1+e_2+e_3 vs e_1+e_2-e_3: dist = 2e_3, dist_sq = 4.
    #   e_1+e_2+e_3 vs e_1+e_2+e_4: dist = e_3-e_4, dist_sq = 2.
    # For weight-2 vs weight-3: 
    #   e_1+e_2 vs e_1+e_2+e_3: dist = -e_3, dist_sq = 1! CONFLICT under max_norm=3.
    #   e_1+e_2 vs e_1+e_3+e_4: dist = e_2-e_3-e_4, dist_sq = 3. OK.
    #   e_1+e_2 vs e_1+e_2-e_3: dist = 2e_3, dist_sq = 4. OK.
    #   e_1+e_2 vs -e_1+e_2+e_3: dist = 2e_1-e_3, dist_sq = 5. OK.

    # So the conflict graph matters. Let's use greedy min-degree.

    # Build conflict list: for each candidate, which other candidates conflict
    # Conflicts under max_norm_sq = 3: dist_sq < 3, i.e., dist_sq in {1, 2}
    # (dist_sq=0 only for identical vectors, which we don't have)

    n_conflicts = np.zeros(n_cand, dtype=np.int64)
    for i in range(n_cand):
        # Count how many j != i have dist_mat[i,j] < 3
        mask = dist_mat[i] < 3
        mask[i] = False  # exclude self
        n_conflicts[i] = int(np.sum(mask))

    # Greedy: repeatedly pick the candidate with minimum current conflict count
    # among remaining candidates
    remaining = list(range(n_cand))
    selected = []
    selected_set = set()

    # Track current conflict count for each remaining candidate
    # When we select candidate s, it conflicts with some remaining candidates,
    # but those candidates are NOT removed - they just can't be selected if they
    # conflict with ANY selected candidate.
    
    # Actually, the constraint is: a candidate v can be added if it does NOT conflict
    # with ANY already-selected candidate. So:
    # eligible(v) = no selected candidate s has dist_mat[v,s] < 3

    # Let's track for each remaining candidate whether it's still eligible
    # and how many selected candidates it conflicts with.

    # For speed, let's precompute the conflict matrix as a set of sets
    conflict_sets = []
    for i in range(n_cand):
        conflicts = set()
        for j in range(n_cand):
            if i != j and dist_mat[i, j] < 3:
                conflicts.add(j)
        conflict_sets.append(conflicts)

    # Greedy min-degree: pick candidate with fewest conflicts among remaining
    while remaining:
        if time.time() > deadline:
            break

        # Find eligible candidates (no conflict with any selected)
        best_candidate = -1
        best_score = -1
        best_conflict_count = n_cand + 1

        for idx in remaining:
            # Check if idx conflicts with any selected
            is_eligible = True
            for s in selected:
                if dist_mat[idx, s] < 3:
                    is_eligible = False
                    break
            if not is_eligible:
                continue
            # Score: prefer candidates with fewer conflicts (more future flexibility)
            # Use n_conflicts as a proxy
            if n_conflicts[idx] < best_conflict_count:
                best_conflict_count = n_conflicts[idx]
                best_candidate = idx

        if best_candidate == -1:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    # Convert to vectors
    selected_vectors = [all_candidates[i] for i in selected]

    # Try to add more via local search: try swapping
    if time.time() < deadline - 5:
        # Local search: try to improve by removing one and adding another
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding any remaining candidate
            for idx in list(remaining):
                if time.time() > deadline - 2:
                    break
                eligible = True
                for s in selected:
                    if dist_mat[idx, s] < 3:
                        eligible = False
                        break
                if eligible:
                    selected.append(idx)
                    remaining.remove(idx)
                    improved = True
                    break

            if not improved:
                # Try swap: remove one selected, add two others? 
                # Or just try swapping one out for one in
                for sel_idx, s in enumerate(selected):
                    if time.time() > deadline - 2:
                        break
                    # Try removing s
                    selected_temp = selected[:sel_idx] + selected[sel_idx + 1:]
                    # Try adding each remaining candidate
                    best_new = -1
                    for idx in remaining:
                        if time.time() > deadline - 2:
                            break
                        eligible = True
                        for t in selected_temp:
                            if dist_mat[idx, t] < 3:
                                eligible = False
                                break
                        if eligible:
                            best_new = idx
                            break
                    if best_new != -1:
                        # Only do swap if it doesn't reduce size (it won't)
                        # Actually swap keeps same size, so no benefit unless
                        # it enables further additions. Skip for now.
                        pass

    result = np.array(selected_vectors, dtype=np.int64)

    # Final verification
    if len(result) > 1:
        norms_sq_final = np.array([np.dot(v, v) for v in result])
        max_ns = int(np.max(norms_sq_final))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                diff = result[i] - result[j]
                if np.dot(diff, diff) < max_ns:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            # Fall back to seed: all weight-2 vectors
            return np.array(weight2, dtype=np.int64)

    return result
# EVOLVE-BLOCK-END
