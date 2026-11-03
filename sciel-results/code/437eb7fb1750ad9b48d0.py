"""Combined LP-guided greedy with bitmask local search for C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import heapq
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using LP-guided greedy initialization
    followed by bitmask-based local search (1-for-1 swaps and block removal).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all t-subsets as bitmasks
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_t = len(all_tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    idx_to_tsub = {i: ts for i, ts in enumerate(all_tsubsets)}

    # All t-subsets as bitmasks
    tsub_masks = []
    for ts in all_tsubsets:
        mask = 0
        for x in ts:
            mask |= (1 << x)
        tsub_masks.append(mask)
    all_tsubsets_set = set(tsub_masks)

    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the t-subsets it covers (as indices)
    block_tsub_indices = []
    rows = []
    cols = []
    data = []
    for bi, block in enumerate(all_blocks):
        tsub_idxs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            tsub_idxs.append(idx)
            rows.append(idx)
            cols.append(bi)
            data.append(1.0)
        block_tsub_indices.append(tsub_idxs)

    # Build sparse incidence matrix M: M[ts_idx, block_idx] = 1
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_blocks))

    # LP relaxation to get dual weights
    dual_weights = None
    try:
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_blocks
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and hasattr(res, 'ineqlin') and res.ineqlin is not None:
            if hasattr(res.ineqlin, 'marginals') and res.ineqlin.marginals is not None:
                dual_weights = np.abs(res.ineqlin.marginals)
                if np.all(dual_weights == 0):
                    dual_weights = None
    except Exception:
        dual_weights = None

    if dual_weights is None:
        dual_weights = np.ones(n_t) / n_t

    # LP-guided greedy construction
    blocks = _lp_greedy_construct(v, k, t, all_blocks, block_tsub_indices,
                                  tsub_to_idx, all_tsubsets_set, dual_weights,
                                  deadline)

    # Local search: try to remove blocks and do 1-for-1 swaps
    if time.time() < deadline - 5:
        blocks = _local_search(blocks, v, k, t, all_tsubsets_set,
                               tsub_to_idx, idx_to_tsub, deadline)

    # Final verification and deduplication
    blocks = _verify_and_dedup(blocks, v, k, t, all_tsubsets_set)

    return blocks


def _lp_greedy_construct(v, k, t, all_blocks, block_tsub_indices,
                         tsub_to_idx, all_tsubsets_set, dual_weights, deadline):
    """
    Greedy construction guided by LP dual weights.
    At each step, pick the block that covers the most 'weighted' uncovered t-subsets.
    """
    covered = set()
    picked = [False] * len(all_blocks)
    blocks = []

    # Weight of each block = sum of dual weights of uncovered t-subsets it covers
    block_weights = np.zeros(len(all_blocks))
    for bi, tsub_idxs in enumerate(block_tsub_indices):
        if tsub_idxs:
            block_weights[bi] = np.sum(dual_weights[tsub_idxs])

    # Max-heap: (-weight, block_index)
    heap = [(-block_weights[bi], bi) for bi in range(len(all_blocks)) if block_weights[bi] > 1e-9]
    heapq.heapify(heap)

    # Map: tsub_mask -> list of block indices that cover it
    tsub_to_blocks = {}
    for bi, tsub_idxs in enumerate(block_tsub_indices):
        for ts_idx in tsub_idxs:
            mask = 0
            for x in all_blocks[bi]:
                # We need the actual t-subset mask, not the block mask
                pass
            # Actually, let's build this differently
            pass

    # Rebuild: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(len(all_tsubsets_set))]
    # We need to map tsub index to mask for lookup
    # Let's use a different approach: for each t-subset index, find blocks covering it
    # This is expensive, so let's use the sparse matrix structure
    # Actually, let's just use the block_tsub_indices in reverse

    # Build reverse map: tsub_idx -> list of block_idx
    tsub_to_blocks = [[] for _ in range(len(block_tsub_indices))]  # wrong size
    tsub_to_blocks = [[] for _ in range(max(len(tsub_to_idx), 1))]
    for bi, tsub_idxs in enumerate(block_tsub_indices):
        for ts_idx in tsub_idxs:
            tsub_to_blocks[ts_idx].append(bi)

    while heap:
        if time.time() > deadline - 10:
            break

        neg_w, bi = heapq.heappop(heap)
        w = -neg_w

        if picked[bi]:
            continue
        if w < block_weights[bi] - 1e-9:
            continue
        if w <= 1e-9:
            break

        # Select this block
        picked[bi] = True
        block = list(all_blocks[bi])
        blocks.append(block)

        # Update coverage
        new_covered = []
        for ts_idx in block_tsub_indices[bi]:
            mask = 0
            for x in all_blocks[bi]:
                # Need to compute mask of t-subset, not block
                pass
            # We need the actual t-subset. Let's precompute tsub masks properly.
            # Actually, tsub_to_idx maps tuple to index. We can get the tuple back.
            # But we need the mask. Let's precompute tsub index -> mask.

        # Simpler approach: just recompute what's covered
        # For efficiency, track covered t-subset indices
        for ts_idx in block_tsub_indices[bi]:
            if ts_idx not in covered:
                covered.add(ts_idx)
                y_s = dual_weights[ts_idx]
                if y_s > 0:
                    for b_prime in tsub_to_blocks[ts_idx]:
                        if not picked[b_prime]:
                            block_weights[b_prime] -= y_s
                            heapq.heappush(heap, (-block_weights[b_prime], b_prime))

    # Check if all covered
    if len(covered) < len(tsub_to_idx):
        # Greedy fallback for remaining
        remaining = set(tsub_to_idx.keys()) - set(all_tsubsets_set)  # wrong
        # Actually, covered is set of indices, tsub_to_idx has all indices
        all_indices = set(range(len(tsub_to_idx)))
        remaining_indices = all_indices - covered
        if remaining_indices:
            # Simple greedy for remaining
            remaining_masks = set()
            for idx in remaining_indices:
                ts = tsub_to_idx  # this is a dict, need reverse
                pass

    return blocks


def _local_search(blocks, v, k, t, all_tsubsets_set, tsub_to_idx, idx_to_tsub, deadline):
    """
    Local search: try to remove redundant blocks, then 1-for-1 swaps.
    """
    # Build coverage count
    coverage = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            mask = 0
            for x in ts:
                mask |= (1 << x)
            coverage[mask] = coverage.get(mask, 0) + 1

    # Try to remove blocks that don't uniquely cover any t-subset
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for bi in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            block = blocks[bi]
            # Check if removing this block uncovers any t-subset
            can_remove = True
            for ts in itertools.combinations(block, t):
                mask = 0
                for x in ts:
                    mask |= (1 << x)
                if coverage.get(mask, 0) <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove block
                for ts in itertools.combinations(block, t):
                    mask = 0
                    for x in ts:
                        mask |= (1 << x)
                    coverage[mask] -= 1
                    if coverage[mask] == 0:
                        del coverage[mask]
                blocks.pop(bi)
                improved = True
                break  # Restart loop since indices changed

    # 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for bi, block in enumerate(blocks):
            if time.time() > deadline - 3:
                break
            block_set = set(block)
            for old_x in block:
                new_block = [x for x in block if x != old_x]
                best_gain = 0
                best_new_x = -1
                for new_x in range(v):
                    if new_x in block_set:
                        continue
                    trial_block = new_block + [new_x]
                    ok = True
                    gain = 0
                    for ts in itertools.combinations(trial_block, t):
                        mask = 0
                        for x in ts:
                            mask |= (1 << x)
                        if mask in coverage:
                            if coverage[mask] == 1 and (1 << old_x) & mask:
                                ok = False
                                break
                        else:
                            gain += 1
                    if not ok:
                        continue
                    if gain > best_gain:
                        best_gain = gain
                        best_new_x = new_x
                if best_new_x != -1 and best_gain > 0:
                    # Perform swap
                    blocks[bi] = sorted(new_block + [best_new_x])
                    # Update coverage
                    for ts in itertools.combinations(block, t):
                        mask = 0
                        for x in ts:
                            mask |= (1 << x)
                        coverage[mask] -= 1
                        if coverage[mask] == 0:
                            del coverage[mask]
                    for ts in itertools.combinations(blocks[bi], t):
                        mask = 0
                        for x in ts:
                            mask |= (1 << x)
                        coverage[mask] = coverage.get(mask, 0) + 1
                    improved = True
                    break
            if improved:
                break

    return blocks


def _verify_and_dedup(blocks, v, k, t, all_tsubsets_set):
    """Verify coverage and remove duplicates."""
    seen = set()
    unique_blocks = []
    covered = set()
    for block in blocks:
        key = tuple(sorted(block))
        if key in seen:
            continue
        seen.add(key)
        unique_blocks.append(list(key))
        for ts in itertools.combinations(key, t):
            mask = 0
            for x in ts:
                mask |= (1 << x)
            covered.add(mask)

    # If not all covered, add emergency blocks
    if covered != all_tsubsets_set:
        uncovered = all_tsubsets_set - covered
        while uncovered:
            seed_mask = next(iter(uncovered))
            block = []
            for i in range(v):
                if seed_mask & (1 << i):
                    block.append(i)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    new_block = block + [x]
                    for ts in itertools.combinations(new_block, t):
                        mask = 0
                        for xx in ts:
                            mask |= (1 << xx)
                        if mask in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            block.sort()
            unique_blocks.append(block)
            for ts in itertools.combinations(block, t):
                mask = 0
                for x in ts:
                    mask |= (1 << x)
                uncovered.discard(mask)
                covered.add(mask)

    return unique_blocks
# EVOLVE-BLOCK-END
