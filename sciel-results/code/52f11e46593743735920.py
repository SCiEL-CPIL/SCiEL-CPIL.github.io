import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse


def _remove_redundant_blocks(v, k, t, blocks, counts=None):
    """Remove blocks that do not uniquely cover any t-subset."""
    if not blocks:
        return [], {}

    if counts is None:
        counts = {}
        for b in blocks:
            for tsub in itertools.combinations(b, t):
                counts[tsub] = counts.get(tsub, 0) + 1

    redundant_indices = set()
    for i, b in enumerate(blocks):
        is_redundant = True
        for tsub in itertools.combinations(b, t):
            if counts.get(tsub, 0) <= 1:
                is_redundant = False
                break
        if is_redundant:
            redundant_indices.add(i)

    if not redundant_indices:
        return blocks, counts

    new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_indices]

    for i in redundant_indices:
        b = blocks[i]
        for tsub in itertools.combinations(b, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]

    while True:
        if not new_blocks:
            return [], counts

        redundant_indices = set()
        for i, b in enumerate(new_blocks):
            is_redundant = True
            for tsub in itertools.combinations(b, t):
                if counts.get(tsub, 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.add(i)

        if not redundant_indices:
            break

        for i in redundant_indices:
            b = new_blocks[i]
            for tsub in itertools.combinations(b, t):
                counts[tsub] -= 1
                if counts[tsub] == 0:
                    del counts[tsub]

        new_blocks = [b for i, b in enumerate(new_blocks) if i not in redundant_indices]

    return new_blocks, counts


def _build_greedy_bitmask(v, k, t, all_tsubs, block_tsub_lists, block_masks, rng, deadline):
    """Fast greedy using precomputed bitmask coverage."""
    num_tsubs = len(all_tsubs)
    full_mask = (1 << num_tsubs) - 1
    uncovered_mask = full_mask
    selected = []

    while uncovered_mask:
        if time.monotonic() > deadline:
            break
        best_gain = 0
        best_block = -1
        for j in range(len(block_masks)):
            common = block_masks[j] & uncovered_mask
            if common:
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = j
        if best_block == -1:
            break
        selected.append(best_block)
        uncovered_mask &= ~block_masks[best_block]

    return selected


def _local_search(v, k, t, blocks, all_kblocks, rng, deadline):
    """Local search: try removing redundant blocks and pair-replacement moves."""
    counts = {}
    for b in blocks:
        for tsub in itertools.combinations(b, t):
            counts[tsub] = counts.get(tsub, 0) + 1

    while time.monotonic() < deadline:
        if len(blocks) < 2:
            break

        idx = rng.randrange(len(blocks))
        B = blocks[idx]

        # Temporarily remove B
        for tsub in itertools.combinations(B, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]

        uncovered_U = []
        for tsub in itertools.combinations(B, t):
            if tsub not in counts:
                uncovered_U.append(tsub)

        if not uncovered_U:
            blocks.pop(idx)
            continue

        # Restore B
        for tsub in itertools.combinations(B, t):
            counts[tsub] = counts.get(tsub, 0) + 1

        # Try pair replacement
        idx2 = rng.randrange(len(blocks))
        if idx2 == idx:
            continue
        B2 = blocks[idx2]

        for tsub in itertools.combinations(B, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]
        for tsub in itertools.combinations(B2, t):
            counts[tsub] -= 1
            if counts[tsub] == 0:
                del counts[tsub]

        U = []
        for tsub in itertools.combinations(B, t):
            if tsub not in counts:
                U.append(tsub)
        for tsub in itertools.combinations(B2, t):
            if tsub not in counts:
                U.append(tsub)

        if not U:
            if idx > idx2:
                idx, idx2 = idx2, idx
            blocks.pop(idx2)
            blocks.pop(idx)
            continue

        S_U = set()
        for u in U:
            S_U.update(u)

        if len(S_U) > k:
            for tsub in itertools.combinations(B, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            for tsub in itertools.combinations(B2, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            continue

        available_points = [x for x in range(v) if x not in S_U]
        needed = k - len(S_U)

        found_better = False
        for _ in range(15):
            if needed == 0:
                B3 = sorted(list(S_U))
            else:
                extra = rng.sample(available_points, needed)
                B3 = sorted(list(S_U) + extra)

            b3_tsubs = list(itertools.combinations(B3, t))
            for tsub in b3_tsubs:
                counts[tsub] = counts.get(tsub, 0) + 1

            new_blocks = []
            remove_set = {idx, idx2}
            for i, b in enumerate(blocks):
                if i not in remove_set:
                    new_blocks.append(b)
            new_blocks.append(B3)

            blocks = new_blocks
            found_better = True
            break

        if not found_better:
            for tsub in itertools.combinations(B, t):
                counts[tsub] = counts.get(tsub, 0) + 1
            for tsub in itertools.combinations(B2, t):
                counts[tsub] = counts.get(tsub, 0) + 1

    blocks, counts = _remove_redundant_blocks(v, k, t, blocks, counts)
    return blocks


def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 145.0

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build sparse incidence matrix and per-block t-subset index lists
    rows = []
    cols = []
    for j, b in enumerate(all_kblocks):
        for ts in itertools.combinations(b, t):
            i = tsub_to_idx[ts]
            rows.append(i)
            cols.append(j)

    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                          shape=(num_tsubs, num_kblocks))
    A_csc = A.tocsc()

    block_tsub_lists = [None] * num_kblocks
    for j in range(num_kblocks):
        s = A_csc.indptr[j]
        e = A_csc.indptr[j + 1]
        block_tsub_lists[j] = A_csc.indices[s:e].tolist()

    # Precompute bitmask coverage per block
    block_masks = [0] * num_kblocks
    for j in range(num_kblocks):
        mask = 0
        for ti in block_tsub_lists[j]:
            mask |= (1 << ti)
        block_masks[j] = mask

    best_blocks = None
    best_len = float('inf')

    # Phase 1: Greedy warm-start with multiple restarts
    greedy_deadline = deadline - 45.0
    restart_num = 0
    while time.monotonic() < greedy_deadline:
        restart_num += 1
        rng = random.Random(1000 + restart_num)

        selected = _build_greedy_bitmask(v, k, t, all_tsubs, block_tsub_lists,
                                         block_masks, rng, deadline - 10.0)
        if not selected:
            continue

        blocks = [list(all_kblocks[j]) for j in selected]

        # Local search refinement
        blocks = _local_search(v, k, t, blocks, all_kblocks, rng, deadline - 8.0)

        # Redundancy pruning
        blocks, _ = _remove_redundant_blocks(v, k, t, blocks)

        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    # Phase 2: ILP refinement on best solution found
    remaining = deadline - time.monotonic() - 3.0
    if remaining > 10.0 and best_blocks is not None:
        c = np.ones(num_kblocks)
        constraint = LinearConstraint(A, lb=1, ub=np.inf)
        bound = Bounds(lb=0, ub=1)
        x0 = np.zeros(num_kblocks)
        for b in best_blocks:
            for j, blk in enumerate(all_kblocks):
                if blk == tuple(b):
                    x0[j] = 1.0
                    break
        try:
            result = milp(c=c, constraints=constraint,
                          integrality=np.ones(num_kblocks),
                          bounds=bound, x0=x0,
                          options={"time_limit": remaining, "mip_rel_gap": 0.0})
            if result.success and result.x is not None:
                ilp_sel = [j for j in range(num_kblocks) if result.x[j] > 0.5]
                if len(ilp_sel) <= len(best_blocks):
                    cov = np.zeros(num_tsubs, dtype=np.int8)
                    for j in ilp_sel:
                        for ti in block_tsub_lists[j]:
                            cov[ti] += 1
                    if cov.min() >= 1:
                        ilp_blocks = [list(all_kblocks[j]) for j in ilp_sel]
                        ilp_blocks, _ = _remove_redundant_blocks(v, k, t, ilp_blocks)
                        if len(ilp_blocks) < best_len:
                            best_blocks = ilp_blocks
                            best_len = len(ilp_blocks)
        except Exception:
            pass

    if best_blocks is None:
        rng = random.Random(0)
        selected = _build_greedy_bitmask(v, k, t, all_tsubs, block_tsub_lists,
                                         block_masks, rng, deadline)
        best_blocks = [list(all_kblocks[j]) for j in selected]
        best_blocks, _ = _remove_redundant_blocks(v, k, t, best_blocks)

    return best_blocks
