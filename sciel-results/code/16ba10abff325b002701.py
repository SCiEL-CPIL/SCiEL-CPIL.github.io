D = 11

# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _generate_supports():
    """Generate candidate nonzero integer vectors."""
    supports = []
    # Level 1: single coord (norm 1)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            supports.append(v)
    # Level 2: two coords (norm sqrt(2))  -- the seed set
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                supports.append(v)
    # Level 3: three coords (norm sqrt(3))
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    supports.append(v)
    # Level 4: four coords (norm sqrt(4)=2)
    for i, j, k, m in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sm in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[m] = si, sj, sk, sm
                        supports.append(v)
    # Level 5: five coords (norm sqrt(5))
    for i, j, k, m, n in itertools.combinations(range(D), 5):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sm in (1, -1):
                        for sn in (1, -1):
                            v = [0] * D
                            v[i], v[j], v[k], v[m], v[n] = si, sj, sk, sm, sn
                            supports.append(v)
    # Level 6: six coords (norm sqrt(6))
    for idx in itertools.combinations(range(D), 6):
        for signs in itertools.product((1, -1), repeat=6):
            v = [0] * D
            for pos, s in zip(idx, signs):
                v[pos] = s
            supports.append(v)
    # Level 7: seven coords (norm sqrt(7))
    for idx in itertools.combinations(range(D), 7):
        for signs in itertools.product((1, -1), repeat=7):
            v = [0] * D
            for pos, s in zip(idx, signs):
                v[pos] = s
            supports.append(v)
    # Level 8: eight coords (norm sqrt(8))
    for idx in itertools.combinations(range(D), 8):
        for signs in itertools.product((1, -1), repeat=8):
            v = [0] * D
            for pos, s in zip(idx, signs):
                v[pos] = s
            supports.append(v)
    # Level 9: nine coords (norm sqrt(9)=3)
    for idx in itertools.combinations(range(D), 9):
        for signs in itertools.product((1, -1), repeat=9):
            v = [0] * D
            for pos, s in zip(idx, signs):
                v[pos] = s
            supports.append(v)
    # Level 10: ten coords (norm sqrt(10))
    for idx in itertools.combinations(range(D), 10):
        for signs in itertools.product((1, -1), repeat=10):
            v = [0] * D
            for pos, s in zip(idx, signs):
                v[pos] = s
            supports.append(v)
    # Level 11: eleven coords (norm sqrt(11))
    for signs in itertools.product((1, -1), repeat=11):
        v = [0] * D
        for pos, s in zip(range(D), signs):
            v[pos] = s
        supports.append(v)
    return np.array(supports, dtype=np.int64)


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin under 150s

    S = _generate_supports()
    n = len(S)

    # Vectorized Gram and norms
    G = S @ S.T
    norms_sq = np.einsum('ij,ij->i', S, S)
    max_norm_sq = int(norms_sq.max())

    # Conflict: dist_sq = ||a||^2 + ||b||^2 - 2 a.b < max_norm_sq
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)

    # Greedy: min-degree first, vectorized degree tracking
    remaining = np.ones(n, dtype=bool)
    selected = []

    while remaining.any():
        # degrees among remaining
        deg = conflict[:, remaining].sum(axis=1)
        deg = deg[remaining]
        # pick min degree (tie-break by smaller index for determinism)
        local_idx = np.argmin(deg)
        # map local idx back to global
        remaining_indices = np.where(remaining)[0]
        node = remaining_indices[local_idx]

        selected.append(node)
        # remove node and its neighbors from remaining
        to_remove = conflict[node, remaining]
        to_remove_indices = remaining_indices[to_remove]
        remaining[to_remove_indices] = False
        remaining[node] = False

        if time.time() > deadline - 25.0:
            break

    # Local search: try to add back some of the removed (conflict) nodes
    # Build a list of candidate nodes not selected
    selected_set = set(selected)
    # We want to try adding nodes that were removed due to conflict with selected
    # Strategy: iterate, find a node not selected that if added we can drop the fewest selected
    # Simpler: 1-swap local search

    # For each non-selected node, compute number of selected it conflicts with
    non_sel = [i for i in range(n) if i not in selected_set]
    # sort by (conflict_count_with_selected, -norm_sq) to prefer low conflict, high norm
    sel_arr = np.array(selected) if selected else np.array([], dtype=int)

    # local search with time budget
    sel_mask = np.zeros(n, dtype=bool)
    sel_mask[selected] = True

    improved = True
    while improved and (time.time() < deadline - 10.0):
        improved = False
        # for each candidate, compute conflicts with selected
        if non_sel:
            cand_conf = conflict[non_sel][:, sel_mask].sum(axis=1)
            # try adding the candidate with minimum conflict count
            order = np.argsort(cand_conf, kind='stable')
            for ci in order:
                if time.time() > deadline - 10.0:
                    break
                cand = non_sel[ci]
                if sel_mask[cand]:
                    continue
                cnt = cand_conf[ci]
                # we can add cand if we remove the cnt selected it conflicts with
                # only beneficial if cnt < 1? No, we remove cnt and add 1, net = 1-cnt
                # only do it if cnt == 0 (pure add)
                if cnt == 0:
                    sel_mask[cand] = True
                    improved = True
                    # rebuild non_sel
                    non_sel = [i for i in range(n) if not sel_mask[i]]
                    if not non_sel:
                        break
                    break

    # Convert back to list
    selected = np.where(sel_mask)[0].tolist()

    # If nothing selected, fallback to seed
    if not selected:
        return S[22:224].copy()  # the level-2 subset

    return S[selected].copy()
# EVOLVE-BLOCK-END
