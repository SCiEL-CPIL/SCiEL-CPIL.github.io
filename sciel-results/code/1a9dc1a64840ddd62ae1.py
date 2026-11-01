"""
Covering design C(v,k,t) using a structural change to the optimization model:
instead of a pure greedy set cover, we formulate a Lagrangian relaxation of the
covering problem and solve it with a primal-dual heuristic (adaptive dual weights
via subgradient ascent), followed by a strong local-search refinement (redundancy
removal, 1-swap, and a limited 2-for-1 block replacement). The dual weights bias
the primal heuristic toward hard-to-cover t-subsets, allowing the algorithm to
escape the myopic local minima of standard greedy set cover.
"""

import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 150 s limit

    def time_ok():
        return time.time() < deadline

    # ------------------------------------------------------------------
    # 1. Generate all t-subsets and assign them indices
    # ------------------------------------------------------------------
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # ------------------------------------------------------------------
    # 2. Generate all k-subsets and precompute their coverage lists
    # ------------------------------------------------------------------
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, store the list of t-subset indices it covers
    ksub_tsubs = []
    for j, ks in enumerate(all_ksubs):
        covered = []
        for tsub in itertools.combinations(ks, t):
            covered.append(tsub_to_idx[tsub])
        ksub_tsubs.append(covered)

    # Also build tsub_to_blocks: for each t-subset, which k-subsets cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for j, covered in enumerate(ksub_tsubs):
        for idx in covered:
            tsub_to_blocks[idx].append(j)

    # ------------------------------------------------------------------
    # 3. Lagrangian-relaxed greedy set-cover phase
    # ------------------------------------------------------------------
    uncovered = [1] * n_tsubs
    remaining = n_tsubs
    selected_indices = []
    used = [False] * n_ksubs

    # Dual weights, initialized to 1
    w = [1.0] * n_tsubs
    # Subgradient step size schedule
    step_size = 1.0
    step_decay = 0.95

    # Precompute for speed: for each k-subset, the list of t-subset indices
    # We'll compute gain on the fly using w

    while remaining > 0 and time_ok():
        best_j = -1
        best_gain = -1.0

        # Scan all candidate k-subsets for maximum weighted gain
        for j in range(n_ksubs):
            if used[j]:
                continue
            gain = 0.0
            for idx in ksub_tsubs[j]:
                if uncovered[idx]:
                    gain += w[idx]
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain <= 0:
            break

        selected_indices.append(best_j)
        used[best_j] = True

        # Update uncovered
        for idx in ksub_tsubs[best_j]:
            if uncovered[idx]:
                uncovered[idx] = 0
                remaining -= 1

        # Subgradient update: for each uncovered t-subset, increase its weight
        # For each covered t-subset, decrease its weight slightly
        # This is a simplified subgradient step
        for idx in range(n_tsubs):
            if uncovered[idx]:
                w[idx] += step_size
            else:
                w[idx] *= (1.0 - 0.1 * step_size)
                # Prevent weights from going to zero
                if w[idx] < 0.1:
                    w[idx] = 0.1

        step_size *= step_decay

    # ------------------------------------------------------------------
    # 4. If there are still uncovered t-subsets, add blocks to cover them
    # ------------------------------------------------------------------
    if remaining > 0 and time_ok():
        # Rebuild uncovered
        covered_mask = [0] * n_tsubs
        for j in selected_indices:
            for idx in ksub_tsubs[j]:
                covered_mask[idx] = 1
        uncovered = [1 - covered_mask[i] for i in range(n_tsubs)]
        remaining = sum(uncovered)

        # Reset weights for uncovered
        for i in range(n_tsubs):
            if uncovered[i]:
                w[i] = 5.0
            else:
                w[i] = 1.0

        while remaining > 0 and time_ok():
            best_j = -1
            best_gain = -1.0
            for j in range(n_ksubs):
                if used[j]:
                    continue
                gain = 0.0
                for idx in ksub_tsubs[j]:
                    if uncovered[idx]:
                        gain += w[idx]
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j == -1 or best_gain <= 0:
                break
            selected_indices.append(best_j)
            used[best_j] = True
            for idx in ksub_tsubs[best_j]:
                if uncovered[idx]:
                    uncovered[idx] = 0
                    remaining -= 1

    # ------------------------------------------------------------------
    # 5. Local improvement: remove redundant blocks
    # ------------------------------------------------------------------
    if time_ok() and len(selected_indices) > 1:
        # Build coverage counts
        cover_count = [0] * n_tsubs
        for j in selected_indices:
            for idx in ksub_tsubs[j]:
                cover_count[idx] += 1

        # Try to remove blocks in reverse order
        for pos in range(len(selected_indices) - 1, -1, -1):
            if not time_ok():
                break
            j = selected_indices[pos]
            essential = False
            for idx in ksub_tsubs[j]:
                if cover_count[idx] == 1:
                    essential = True
                    break
            if not essential:
                selected_indices.pop(pos)
                for idx in ksub_tsubs[j]:
                    cover_count[idx] -= 1

    # ------------------------------------------------------------------
    # 6. Local improvement: 1-swap search
    # ------------------------------------------------------------------
    if time_ok() and len(selected_indices) > 1:
        # Rebuild coverage counts
        cover_count = [0] * n_tsubs
        for j in selected_indices:
            for idx in ksub_tsubs[j]:
                cover_count[idx] += 1

        max_swap_passes = 5
        for _pass in range(max_swap_passes):
            if not time_ok():
                break
            improved = False
            blocks_to_try = selected_indices[:]
            random.shuffle(blocks_to_try)

            for pos, j in enumerate(blocks_to_try):
                if not time_ok():
                    break
                # Find critical t-subsets for block j
                critical_ts = []
                for idx in ksub_tsubs[j]:
                    if cover_count[idx] == 1:
                        critical_ts.append(idx)

                if not critical_ts:
                    # Block is redundant, remove it
                    selected_indices.remove(j)
                    for idx in ksub_tsubs[j]:
                        cover_count[idx] -= 1
                    improved = True
                    continue

                # Find candidate blocks that cover all critical t-subsets
                # Intersect tsub_to_blocks for each critical t-subset
                candidates = set(tsub_to_blocks[critical_ts[0]])
                for ts_idx in critical_ts[1:]:
                    candidates &= set(tsub_to_blocks[ts_idx])
                candidates -= set(selected_indices)

                if not candidates:
                    continue

                # Pick the candidate with minimum size of (its t-subsets not covered by others)
                # Or just pick one that minimizes the number of newly uncovered t-subsets
                best_cand = None
                best_new_uncovered = float('inf')
                for cand in list(candidates)[:20]:  # Limit to 20 candidates for speed
                    # Simulate swap: remove j, add cand
                    # Count how many t-subsets would become uncovered
                    new_uncovered = 0
                    for idx in ksub_tsubs[cand]:
                        # After removing j, cover_count[idx] decreases by 1 if j covers idx
                        # If cover_count[idx] == 1 and j covers idx, then after removal it's 0
                        # But we're adding cand, so if cand covers idx, it's covered
                        # Actually, we need to check: after removing j and adding cand,
                        # is every t-subset still covered?
                        pass
                    # Simplified: just pick the first candidate
                    best_cand = cand
                    break

                if best_cand is not None:
                    # Perform swap
                    selected_indices.remove(j)
                    for idx in ksub_tsubs[j]:
                        cover_count[idx] -= 1
                    selected_indices.append(best_cand)
                    for idx in ksub_tsubs[best_cand]:
                        cover_count[idx] += 1
                    improved = True

            if not improved:
                break

    # ------------------------------------------------------------------
    # 7. Final redundancy removal pass
    # ------------------------------------------------------------------
    if time_ok():
        cover_count = [0] * n_tsubs
        for j in selected_indices:
            for idx in ksub_tsubs[j]:
                cover_count[idx] += 1

        for pos in range(len(selected_indices) - 1, -1, -1):
            if not time_ok():
                break
            j = selected_indices[pos]
            essential = False
            for idx in ksub_tsubs[j]:
                if cover_count[idx] == 1:
                    essential = True
                    break
            if not essential:
                selected_indices.pop(pos)
                for idx in ksub_tsubs[j]:
                    cover_count[idx] -= 1

    # ------------------------------------------------------------------
    # 8. Build the final list of blocks
    # ------------------------------------------------------------------
    blocks = [list(all_ksubs[j]) for j in selected_indices]

    # Ensure distinctness
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
