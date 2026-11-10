import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Build mask matrix: M[i, j] = 1 if k-subset i covers t-subset j
    M = np.zeros((n_k, n_t), dtype=np.uint8)
    for i, k_sub in enumerate(all_k_subsets):
        for t_sub in itertools.combinations(k_sub, t):
            j = t_to_idx[t_sub]
            M[i, j] = 1

    # coverers: for each t-subset, list of k-subsets covering it
    coverers = [list(np.where(M[:, j] == 1)[0]) for j in range(n_t)]

    def get_blocks(indices):
        return [list(all_k_subsets[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    best_indices = None

    # Vectorized local search
    def local_search(initial_indices, seed, time_limit=5.0, max_pairs_21=5000, max_pairs_22=1000, max_triples_31=200):
        rng = random.Random(seed)
        S = set(initial_indices)
        if not S:
            return S

        # counts[j] = number of blocks in S covering t-subset j
        counts = np.zeros(n_t, dtype=np.int32)
        for b in S:
            counts += M[b]

        improved = True
        ls_start = time.time()

        while improved and (time.time() - ls_start) < time_limit:
            improved = False

            # Pass 1: Remove Redundant Blocks (vectorized)
            S_list = list(S)
            if S_list:
                sub_counts = counts - M[S_list]  # shape (|S|, n_t)
                redundant_mask = (sub_counts <= 0).all(axis=1)
                to_remove = [S_list[i] for i in range(len(S_list)) if redundant_mask[i]]
                if to_remove:
                    for b in to_remove:
                        S.discard(b)
                        counts -= M[b]
                    improved = True
                    continue

            # Pass 2: 1-1 Swap (vectorized)
            S_list = list(S)
            rng.shuffle(S_list)
            swapped = False
            for b_out in S_list:
                if b_out not in S:
                    continue
                # U = t-subsets only covered by b_out
                U_mask = (counts == 1) & (M[b_out] == 1)
                U = np.where(U_mask)[0]

                if len(U) == 0:
                    # Can remove b_out entirely
                    S.discard(b_out)
                    counts -= M[b_out]
                    improved = True
                    swapped = True
                    break

                # Find candidates covering all of U
                candidates = set(coverers[U[0]])
                for j in U[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(coverers[j])
                candidates -= S

                if candidates:
                    b_in = rng.choice(list(candidates))
                    S.discard(b_out)
                    S.add(b_in)
                    counts -= M[b_out]
                    counts += M[b_in]
                    improved = True
                    swapped = True
                    break

            if swapped:
                continue

            # Pass 3: 2-1 Swap (vectorized, limited pairs)
            if time.time() - ls_start < time_limit - 0.5:
                S_list = list(S)
                rng.shuffle(S_list)
                n_S = len(S_list)
                pairs_checked = 0
                swapped = False

                for i_idx in range(n_S):
                    if pairs_checked > max_pairs_21:
                        break
                    b1 = S_list[i_idx]
                    if b1 not in S:
                        continue
                    for j_idx in range(i_idx + 1, n_S):
                        if pairs_checked > max_pairs_21:
                            break
                        b2 = S_list[j_idx]
                        if b2 not in S:
                            continue
                        pairs_checked += 1

                        # U = t-subsets that would become uncovered
                        new_counts = counts - M[b1] - M[b2]
                        U = np.where((new_counts == 0) & ((M[b1] | M[b2]) == 1))[0]

                        if len(U) == 0:
                            S.discard(b1)
                            S.discard(b2)
                            counts -= M[b1] + M[b2]
                            improved = True
                            swapped = True
                            break

                        # Find candidate covering all of U
                        candidates = set(coverers[U[0]])
                        for jj in U[1:]:
                            if not candidates:
                                break
                            candidates.intersection_update(coverers[jj])
                        candidates -= S

                        if candidates:
                            b_in = rng.choice(list(candidates))
                            S.discard(b1)
                            S.discard(b2)
                            S.add(b_in)
                            counts -= M[b1] + M[b2]
                            counts += M[b_in]
                            improved = True
                            swapped = True
                            break
                    if swapped:
                        break

                if swapped:
                    continue

            # Pass 4: 3-1 Swap (Remove 3, Add 1) - for deeper minima
            if time.time() - ls_start < time_limit - 1.0:
                S_list = list(S)
                rng.shuffle(S_list)
                n_S = len(S_list)
                triples_checked = 0
                swapped = False

                for i_idx in range(n_S):
                    if triples_checked > max_triples_31:
                        break
                    b1 = S_list[i_idx]
                    if b1 not in S:
                        continue
                    for j_idx in range(i_idx + 1, n_S):
                        if triples_checked > max_triples_31:
                            break
                        b2 = S_list[j_idx]
                        if b2 not in S:
                            continue
                        for l_idx in range(j_idx + 1, n_S):
                            if triples_checked > max_triples_31:
                                break
                            b3 = S_list[l_idx]
                            if b3 not in S:
                                continue
                            triples_checked += 1

                            new_counts = counts - M[b1] - M[b2] - M[b3]
                            U = np.where((new_counts == 0) & ((M[b1] | M[b2] | M[b3]) == 1))[0]

                            if len(U) == 0:
                                S.discard(b1)
                                S.discard(b2)
                                S.discard(b3)
                                counts -= M[b1] + M[b2] + M[b3]
                                improved = True
                                swapped = True
                                break

                            candidates = set(coverers[U[0]])
                            for jj in U[1:]:
                                if not candidates:
                                    break
                                candidates.intersection_update(coverers[jj])
                            candidates -= S

                            if candidates:
                                b_in = rng.choice(list(candidates))
                                S.discard(b1)
                                S.discard(b2)
                                S.discard(b3)
                                S.add(b_in)
                                counts -= M[b1] + M[b2] + M[b3]
                                counts += M[b_in]
                                improved = True
                                swapped = True
                                break
                        if swapped:
                            break
                    if swapped:
                        break

                if swapped:
                    continue

            # Pass 5: 2-2 Swap (limited)
            if time.time() - ls_start < time_limit - 1.5:
                S_list = list(S)
                rng.shuffle(S_list)
                n_S = len(S_list)
                pairs_checked_22 = 0
                swapped = False

                for i_idx in range(n_S):
                    if pairs_checked_22 > max_pairs_22:
                        break
                    b1 = S_list[i_idx]
                    if b1 not in S:
                        continue
                    for j_idx in range(i_idx + 1, n_S):
                        if pairs_checked_22 > max_pairs_22:
                            break
                        b2 = S_list[j_idx]
                        if b2 not in S:
                            continue
                        pairs_checked_22 += 1

                        new_counts = counts - M[b1] - M[b2]
                        U = np.where((new_counts == 0) & ((M[b1] | M[b2]) == 1))[0]

                        if len(U) == 0:
                            S.discard(b1)
                            S.discard(b2)
                            counts -= M[b1] + M[b2]
                            improved = True
                            swapped = True
                            break

                        # Find two candidates that together cover U
                        if len(U) > 1:
                            best_c1 = -1
                            best_c1_gain = 0
                            for c1 in coverers[U[0]]:
                                if c1 in S and c1 != b1 and c1 != b2:
                                    continue
                                gain = int(np.sum(M[c1][U] == 1))
                                if gain > best_c1_gain:
                                    best_c1_gain = gain
                                    best_c1 = c1
                            if best_c1 == -1 or best_c1_gain < len(U) // 2:
                                continue

                            remaining = U[M[best_c1][U] == 0]
                            if len(remaining) == 0:
                                all_cands = [c for c in coverers[U[0]] if c not in S and c != best_c1]
                                if not all_cands:
                                    continue
                                best_c2 = rng.choice(all_cands)
                            else:
                                best_c2 = -1
                                best_c2_gain = 0
                                for c2 in coverers[remaining[0]]:
                                    if c2 in S and c2 != b1 and c2 != b2:
                                        continue
                                    if c2 == best_c1:
                                        continue
                                    gain = int(np.sum(M[c2][remaining] == 1))
                                    if gain > best_c2_gain:
                                        best_c2_gain = gain
                                        best_c2 = c2
                                if best_c2 == -1:
                                    continue

                            # Verify coverage
                            cov = M[best_c1][U] | M[best_c2][U]
                            if not np.all(cov == 1):
                                continue

                            S.discard(b1)
                            S.discard(b2)
                            S.add(best_c1)
                            S.add(best_c2)
                            counts -= M[b1] + M[b2]
                            counts += M[best_c1] + M[best_c2]
                            improved = True
                            swapped = True
                            break
                        else:
                            # Single uncovered, just do 2-1 style
                            candidates = set(coverers[U[0]])
                            candidates -= S
                            if candidates:
                                b_in = rng.choice(list(candidates))
                                S.discard(b1)
                                S.discard(b2)
                                S.add(b_in)
                                counts -= M[b1] + M[b2]
                                counts += M[b_in]
                                improved = True
                                swapped = True
                                break

                if swapped:
                    continue

        return S

    # Run multiple randomized greedy attempts
    attempts = 25
    for attempt in range(attempts):
        if time.time() > deadline - 10:
            break

        rng = random.Random(attempt * 12345 + 67890)

        current_uncovered = np.ones(n_t, dtype=np.uint8)
        selected = set()
        selected_union = np.zeros(n_t, dtype=np.uint8)

        while current_uncovered.any():
            gains = np.minimum(M.T @ current_uncovered, 35)  # cap at C(k,t)
            # Exclude already selected
            for s in selected:
                gains[s] = 0

            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]

            if best_gain == 0:
                break

            # Conflict-aware tie-breaking: among top-gain blocks, prefer lower overlap
            top_mask = (gains == best_gain) & (gains > 0)
            top_candidates = np.where(top_mask)[0]
            if len(top_candidates) > 1:
                overlaps = (M[top_candidates] & selected_union).sum(axis=1)
                min_ov = overlaps.min()
                min_ov_cands = top_candidates[overlaps == min_ov]
                best_idx = int(rng.choice(min_ov_cands))

            selected.add(best_idx)
            current_uncovered &= ~M[best_idx]
            selected_union |= M[best_idx]

            if time.time() > deadline - 15:
                break

        if current_uncovered.all() == 0:
            indices = list(selected)

            # Apply Local Search
            ls_indices = local_search(indices, attempt, time_limit=5.0)
            ls_len = len(ls_indices)

            if ls_len < best_len:
                best_len = ls_len
                best_indices = ls_indices
                best_solution = get_blocks(ls_indices)

            if time.time() > deadline - 15:
                break

    # Final Polish: multiple longer local search passes on the best solution
    if best_indices is not None and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 3:
            polished = local_search(best_indices, 99999, time_limit=min(remaining_time * 0.5, 20.0),
                                     max_pairs_21=10000, max_pairs_22=2000, max_triples_31=500)
            if len(polished) < best_len:
                best_len = len(polished)
                best_indices = polished
                best_solution = get_blocks(polished)

            if time.time() < deadline - 3:
                remaining2 = deadline - 3 - time.time()
                if remaining2 > 2:
                    polished2 = local_search(best_indices, 123456, time_limit=min(remaining2, 15.0),
                                              max_pairs_21=20000, max_pairs_22=3000, max_triples_31=1000)
                    if len(polished2) < best_len:
                        best_len = len(polished2)
                        best_indices = polished2
                        best_solution = get_blocks(polished2)

            if time.time() < deadline - 2:
                remaining3 = deadline - 2 - time.time()
                if remaining3 > 2:
                    polished3 = local_search(best_indices, 777777, time_limit=min(remaining3, 10.0),
                                             max_pairs_21=30000, max_pairs_22=5000, max_triples_31=2000)
                    if len(polished3) < best_len:
                        best_len = len(polished3)
                        best_indices = polished3
                        best_solution = get_blocks(polished3)

    if best_solution is None:
        # Fallback: Simple deterministic greedy
        current_uncovered = np.ones(n_t, dtype=np.uint8)
        selected = []
        while current_uncovered.any():
            gains = M.T @ current_uncovered
            for s in selected:
                gains[s] = 0
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            selected.append(best_idx)
            current_uncovered &= ~M[best_idx]
        best_solution = get_blocks(selected)

    return best_solution
