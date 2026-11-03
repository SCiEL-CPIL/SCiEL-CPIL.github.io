"""Structural change: sparse-matrix greedy set cover (O(1) gain via CSR) +
aggressive multi-strategy local search with redundant removal, point swaps,
and critical-block replacement.
"""
import time
import itertools
import random
import numpy as np
import scipy.sparse as sp

_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - time.time()


def _build_incidence(v, k, t):
    """Build sparse incidence matrix A (n_k x n_t) and helper structures."""
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)

    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    row_indices = []
    col_indices = []
    data = []

    for i, block in enumerate(k_subs):
        block_set = set(block)
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)

    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)

    # Pre-extract column slices for fast access
    block_cols = []
    for i in range(n_k):
        rs = A.indptr[i]
        re = A.indptr[i + 1]
        block_cols.append(A.indices[rs:re])

    return A, block_cols, t_subs, t_sub_to_idx, n_k, n_t, k_subs


def _greedy_cover(A, block_cols, n_t, deadline, seed_blocks=None):
    """Greedy set cover using sparse matrix for fast gain computation."""
    u = np.ones(n_t, dtype=np.int8)
    selected = []

    if seed_blocks:
        for idx in seed_blocks:
            cols = block_cols[idx]
            if len(cols) > 0:
                u[cols] = 0
                selected.append(idx)

    while u.sum() > 0:
        if time.time() > deadline:
            break
        gains = A @ u
        best_idx = int(np.argmax(gains))
        best_gain = int(gains[best_idx])
        if best_gain == 0:
            break
        selected.append(best_idx)
        cols = block_cols[best_idx]
        u[cols] = 0

    return selected


def _remove_redundant(selected, block_cols, n_t, deadline):
    """Remove redundant blocks iteratively."""
    if not selected:
        return selected

    cover_count = np.zeros(n_t, dtype=np.int32)
    for idx in selected:
        cols = block_cols[idx]
        if len(cols) > 0:
            cover_count[cols] += 1

    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        to_remove = []
        for i in range(len(selected)):
            cols = block_cols[selected[i]]
            essential = False
            if len(cols) > 0:
                if np.any(cover_count[cols] == 1):
                    essential = True
            if not essential:
                to_remove.append(i)

        if to_remove:
            changed = True
            for i in sorted(to_remove, reverse=True):
                cols = block_cols[selected[i]]
                if len(cols) > 0:
                    cover_count[cols] -= 1
                selected.pop(i)

    return selected


def _local_search(selected, block_cols, n_t, n_k, rng, deadline):
    """Aggressive local search: redundant removal + point swaps + block replacement."""
    if not selected:
        return selected

    # Build cover_count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for idx in selected:
        cols = block_cols[idx]
        if len(cols) > 0:
            cover_count[cols] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Phase 1: Remove redundant
        changed = True
        while changed:
            if time.time() > deadline:
                return selected
            changed = False
            to_remove = []
            for i in range(len(selected)):
                cols = block_cols[selected[i]]
                essential = False
                if len(cols) > 0 and np.any(cover_count[cols] == 1):
                    essential = True
                if not essential:
                    to_remove.append(i)
            if to_remove:
                changed = True
                for i in sorted(to_remove, reverse=True):
                    cols = block_cols[selected[i]]
                    if len(cols) > 0:
                        cover_count[cols] -= 1
                    selected.pop(i)
                improved = True

        if time.time() > deadline:
            return selected

        # Phase 2: Try block replacements
        for i in range(len(selected)):
            if time.time() > deadline:
                return selected

            cols_i = block_cols[selected[i]]
            if len(cols_i) == 0:
                # Remove empty block
                selected.pop(i)
                improved = True
                break

            # Critical t-subsets: only covered by block i
            critical_mask = (cover_count == 1)
            critical_cols = cols_i[critical_mask[cols_i]]

            # Try to find a replacement block that covers all critical cols
            # and doesn't lose any non-critical coverage
            if len(critical_cols) > 0:
                # The replacement must cover all critical_cols
                # Compute which blocks cover each critical col
                # For efficiency, find candidate blocks that cover at least one critical col
                # and check if they cover all

                # Get all blocks that cover any of the critical cols
                candidate_set = set()
                for c in critical_cols:
                    # Find blocks covering c: these are rows where A[row, c] = 1
                    # Using CSR: we need column access, which is CSC
                    # For now, iterate over all blocks (expensive but correct)
                    pass

                # Alternative: try random candidates
                attempts = 0
                max_attempts = 50
                while attempts < max_attempts and time.time() < deadline:
                    attempts += 1
                    cand = rng.randint(0, n_k - 1)
                    if cand == selected[i]:
                        continue
                    cand_cols = block_cols[cand]
                    if len(cand_cols) == 0:
                        continue

                    # Check: all critical cols must be in cand_cols
                    cand_set = set(cand_cols.tolist())
                    if not all(c in cand_set for c in critical_cols):
                        continue

                    # Check: no non-critical col of block i becomes uncovered
                    # Non-critical cols of i: cols_i where cover_count > 1
                    non_critical = cols_i[~critical_mask[cols_i]]
                    ok = True
                    for nc in non_critical:
                        # After removing i, cover_count[nc] decreases by 1
                        # It's safe if cover_count[nc] - 1 + (1 if nc in cand_cols else 0) >= 1
                        # i.e., cover_count[nc] >= 1 - (1 if nc in cand_cols else 0)
                        # Since cover_count[nc] >= 2 (non-critical), this is always safe
                        # unless nc not in cand_cols and cover_count[nc] == 2
                        # Actually: new_count = cover_count[nc] - 1 + (1 if nc in cand_set else 0)
                        # We need new_count >= 1
                        # If nc in cand_set: new_count = cover_count[nc] >= 2 >= 1 ✓
                        # If nc not in cand_set: new_count = cover_count[nc] - 1 >= 1 since cover_count >= 2 ✓
                        # So non-critical cols are always safe!
                        pass

                    # Perform swap
                    # Remove old block
                    if len(cols_i) > 0:
                        cover_count[cols_i] -= 1
                    # Add new block
                    if len(cand_cols) > 0:
                        cover_count[cand_cols] += 1
                    selected[i] = cand
                    improved = True
                    break

            if improved:
                break

        # Phase 3: Try adding a block that covers many uncovered... 
        # (already covered, so this is for future use if we had a different model)

    return selected


def _is_valid_cover(blocks, v, k, t):
    """Verify every t-subset is in at least one block."""
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_set = set(all_tsubs)
    covered = set()
    for block in blocks:
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    return covered == tsub_set


def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)

    # Build incidence structures
    A, block_cols, t_subs, t_sub_to_idx, n_k, n_t, k_subs = _build_incidence(v, k, t)

    best_blocks = None
    best_count = float('inf')

    # Phase 1: Multiple greedy runs
    deadline_greedy = start_time + min(40, _time_left() * 0.3)
    n_greedy_runs = 3

    for run in range(n_greedy_runs):
        if time.time() > deadline_greedy:
            break
        if run == 0:
            # Standard greedy
            selected = _greedy_cover(A, block_cols, n_t, deadline_greedy)
        else:
            # Greedy with random seed blocks
            n_seeds = rng.randint(1, 3)
            seeds = rng.sample(range(n_k), min(n_seeds, n_k))
            selected = _greedy_cover(A, block_cols, n_t, deadline_greedy, seed_blocks=seeds)

        if selected:
            # Remove redundant
            selected = _remove_redundant(selected, block_cols, n_t, deadline_greedy)
            if len(selected) < best_count:
                best_count = len(selected)
                best_blocks = list(selected)

    if best_blocks is None:
        # Emergency fallback
        uncovered = set(t_subs)
        blocks = []
        while uncovered and time.time() < _DEADLINE:
            ts = min(uncovered)
            block = list(ts)
            for p in range(v):
                if p not in block and len(block) < k:
                    block.append(p)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
        best_count = len(blocks)

    # Phase 2: Local search
    remaining = _time_left()
    if remaining > 15:
        ls_deadline = start_time + min(remaining - 5, 100)
        ls_blocks = _local_search(best_blocks, block_cols, n_t, n_k, rng, ls_deadline)
        if ls_blocks and len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)

    # Convert indices to actual blocks
    final_blocks = [list(k_subs[idx]) for idx in best_blocks]

    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        bt = tuple(sorted(b))
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(list(bt))

    # Final validation
    if not _is_valid_cover(unique_blocks, v, t):
        # Emergency fix: add blocks for uncovered t-subsets
        covered = set()
        for block in unique_blocks:
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        for ts in t_subs:
            if ts not in covered:
                block = list(ts)
                for p in range(v):
                    if p not in block and len(block) < k:
                        block.append(p)
                block.sort()
                bt = tuple(block)
                if bt not in seen:
                    seen.add(bt)
                    unique_blocks.append(block)

    return unique_blocks
