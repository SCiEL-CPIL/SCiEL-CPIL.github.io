"""Covering design C(19,7,4) - optimized greedy with local search and MILP polish."""
import itertools
import time
import random
import heapq
from collections import defaultdict

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    _HAS_MILP = True
except Exception:
    _HAS_MILP = False


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map block to its t-subset indices
    block_tsubs = [None] * n_blocks
    # Map t-subset index to list of block indices covering it
    tsub_blocks = [[] for _ in range(n_t)]

    for i, block in enumerate(all_blocks):
        idxs = [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_tsubs[i] = idxs
        for idx in idxs:
            tsub_blocks[idx].append(i)

    # Precompute for faster greedy: use numpy arrays for coverage
    covered_arr = np.zeros(n_t, dtype=np.uint8)

    best_blocks = None
    best_count = float('inf')

    # Run multiple restarts of the heap-based greedy
    n_restarts = 8

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        # Lazy Heap Greedy
        gains = np.array([len(block_tsubs[i]) for i in range(n_blocks)], dtype=np.int32)
        covered = np.zeros(n_t, dtype=np.uint8)
        covered_count = 0
        selected_blocks = []

        # Heap stores (-gain, block_id)
        heap = [(-int(gains[i]), i) for i in range(n_blocks)]
        heapq.heapify(heap)

        while covered_count < n_t and time_ok():
            neg_gain, b_idx = heapq.heappop(heap)
            current_gain = -neg_gain

            # Check if this gain is still valid (lazy deletion)
            if current_gain != int(gains[b_idx]):
                continue

            if current_gain == 0:
                break

            # Select this block
            selected_blocks.append(b_idx)

            # Update coverage
            for idx in block_tsubs[b_idx]:
                if not covered[idx]:
                    covered[idx] = 1
                    covered_count += 1
                    # Update gains of all blocks covering this t-subset
                    for other_b_idx in tsub_blocks[idx]:
                        if gains[other_b_idx] > 0:
                            gains[other_b_idx] -= 1
                            heapq.heappush(heap, (-int(gains[other_b_idx]), other_b_idx))

        if covered_count == n_t:
            # Deletion phase: remove redundant blocks
            if time_ok():
                tsub_block_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in selected_blocks:
                    for idx in block_tsubs[b_idx]:
                        tsub_block_count[idx] += 1

                changed = True
                while changed and time_ok():
                    changed = False
                    for i in range(len(selected_blocks)):
                        if not time_ok():
                            break
                        removable = True
                        for idx in block_tsubs[selected_blocks[i]]:
                            if tsub_block_count[idx] <= 1:
                                removable = False
                                break
                        if removable:
                            for idx in block_tsubs[selected_blocks[i]]:
                                tsub_block_count[idx] -= 1
                            selected_blocks.pop(i)
                            changed = True
                            break

            # Local search: try to swap or remove blocks
            if time_ok():
                selected_blocks = _local_search(
                    selected_blocks, block_tsubs, n_t, all_blocks, n_blocks,
                    tsub_blocks, t_sub_to_idx, deadline
                )

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [all_blocks[i] for i in selected_blocks]

            if best_count <= 145:
                break

    if best_blocks is None:
        return []

    # MILP polish
    if _HAS_MILP and time_ok():
        remaining_time = deadline - time.time()
        if remaining_time > 20:
            best_blocks = _milp_polish(
                v, k, t, best_blocks, t_sub_to_idx, all_t_subsets,
                block_tsubs, deadline, all_blocks, n_blocks, tsub_blocks
            )

    return best_blocks


def _local_search(selected_blocks, block_tsubs, n_t, all_blocks, n_blocks,
                  tsub_blocks, t_sub_to_idx, deadline):
    """Try to reduce block count via removal and swap operations."""

    def time_ok():
        return time.time() < deadline

    if not selected_blocks:
        return selected_blocks

    # Build coverage counts
    tsub_count = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_blocks:
        for idx in block_tsubs[b_idx]:
            tsub_count[idx] += 1

    # First pass: try removing any block that is redundant
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(selected_blocks)):
            if not time_ok():
                break
            b_idx = selected_blocks[i]
            removable = True
            for idx in block_tsubs[b_idx]:
                if tsub_count[idx] <= 1:
                    removable = False
                    break
            if removable:
                for idx in block_tsubs[b_idx]:
                    tsub_count[idx] -= 1
                selected_blocks.pop(i)
                changed = True
                break

    # Second pass: try swapping one block for another that allows removal
    improved = True
    while improved and time_ok():
        improved = False
        for i in range(len(selected_blocks)):
            if not time_ok():
                break
            b_idx = selected_blocks[i]
            # Find t-subsets uniquely covered by this block
            unique_tsubs = []
            for idx in block_tsubs[b_idx]:
                if tsub_count[idx] == 1:
                    unique_tsubs.append(idx)

            if not unique_tsubs:
                continue

            # Try to find a replacement block that covers all unique_tsubs
            # and when added, allows removal of b_idx
            found_swap = False
            for other_b_idx in range(n_blocks):
                if not time_ok():
                    break
                if other_b_idx == b_idx:
                    continue
                if other_b_idx in selected_blocks:
                    continue

                # Check if other_b covers all unique_tsubs
                other_tsubs = set(block_tsubs[other_b_idx])
                covers_all = all(idx in other_tsubs for idx in unique_tsubs)
                if not covers_all:
                    continue

                # Tentatively add other_b and remove b
                new_tsub_count = tsub_count.copy()
                for idx in block_tsubs[other_b_idx]:
                    new_tsub_count[idx] += 1
                for idx in block_tsubs[b_idx]:
                    new_tsub_count[idx] -= 1

                # Check validity: all t-subsets must have count >= 1
                if np.all(new_tsub_count >= 1):
                    # Check if this swap allows further reduction
                    # (i.e., some other block becomes redundant)
                    tsub_count = new_tsub_count
                    selected_blocks[i] = other_b_idx
                    found_swap = True
                    improved = True
                    break

            if found_swap:
                # Try to remove any now-redundant blocks
                for j in range(len(selected_blocks)):
                    if not time_ok():
                        break
                    b_idx2 = selected_blocks[j]
                    removable = True
                    for idx in block_tsubs[b_idx2]:
                        if tsub_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for idx in block_tsubs[b_idx2]:
                            tsub_count[idx] -= 1
                        selected_blocks.pop(j)
                        improved = True
                        break

    return selected_blocks


def _milp_polish(v, k, t, blocks, t_sub_to_idx, all_t_subsets, block_tsubs,
                 deadline, all_blocks, n_blocks, tsub_blocks):
    n_t = len(all_t_subsets)
    n_blocks_in = len(blocks)
    if n_blocks_in == 0:
        return blocks

    # Build the constraint matrix A (n_t x n_blocks_in)
    from scipy.sparse import lil_matrix
    A = lil_matrix((n_t, n_blocks_in), dtype=np.int8)
    for j in range(n_blocks_in):
        b = blocks[j]
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            A[idx, j] = 1

    A = A.tocsr()

    c = np.ones(n_blocks_in)
    constraint = LinearConstraint(A, 1, None)
    bounds = Bounds(0, 1)

    remaining = deadline - time.time()
    if remaining < 15:
        return blocks

    try:
        res = milp(c, integrality=np.ones(n_blocks_in), constraints=constraint,
                   bounds=bounds, options={'time_limit': int(remaining - 5),
                                           'mip_rel_gap': 0.0,
                                           'mip_absgap': 0.0, 'verbose': 0})
    except Exception:
        return blocks

    if res.x is not None:
        sel = [j for j in range(n_blocks_in) if res.x[j] > 0.5]
        new_blocks = [blocks[i] for i in sel]

        # Verify coverage
        if len(new_blocks) < len(blocks):
            cov = np.zeros(n_t, dtype=np.uint8)
            for b in new_blocks:
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    cov[idx] = 1
            if np.all(cov):
                return new_blocks

    return blocks
