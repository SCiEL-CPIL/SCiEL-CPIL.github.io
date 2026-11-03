"""
Combined strategy: Bitset-based precomputation for fast gain updates,
vectorized scoring with bias for greedy construction, and local search
(removal + swaps) for refinement.
"""
import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Data structures
    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    # Bitsets for fast set operations
    # tsub_mask[b] has bits for tsubs covered by block b
    # tsub_blocks_mask[ts] has bits for blocks covering ts
    tsub_mask = [0] * n_blocks
    tsub_blocks_mask = [0] * n_tsubsets

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask

    # Precompute sets for faster intersection in local search
    tsub_block_sets = [set(lst) for lst in tsub_blocks]

    full_tsub_mask = (1 << n_tsubsets) - 1

    # Precompute conflict masks for Packing LB (independent sets)
    # Two t-subsets conflict if they are covered by the same block
    tsub_conflict_mask = [0] * n_tsubsets
    for b_idx in range(n_blocks):
        mask = tsub_mask[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_conflict_mask[idx] |= mask
            m ^= lsb

    def packing_lb(uncovered_mask):
        """
        Calculate lower bound using maximal independent set of uncovered t-subsets.
        Two t-subsets are independent if no single block covers both.
        """
        count = 0
        remaining = uncovered_mask
        while remaining:
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            count += 1
            remaining &= ~tsub_conflict_mask[idx]
        return count

    def greedy_cover(seed):
        """
        Greedy set cover using a max-heap with vectorized gain updates.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        selected_mask = 0

        # Initial gains: number of tsubs covered by each block
        gains = [0] * n_blocks
        for b_idx in range(n_blocks):
            gains[b_idx] = tsub_mask[b_idx].bit_count()

        heap = [(-gains[b_idx], b_idx) for b_idx in range(n_blocks)]
        heapq.heapify(heap)

        while uncovered_mask:
            if time.time() > deadline - 10.0:
                break

            best_block = -1
            best_gain = 0

            # Pop stale entries
            while heap:
                neg_g, b_idx = heapq.heappop(heap)
                g = -neg_g
                if g == gains[b_idx] and not (selected_mask & (1 << b_idx)):
                    best_block = b_idx
                    best_gain = g
                    break
                # Stale or already selected, discard

            if best_block == -1:
                break

            selected.append(best_block)
            selected_mask |= (1 << best_block)
            gains[best_block] = -1  # Mark as selected

            # Update gains for blocks covering newly covered tsubs
            new_covered = tsub_mask[best_block] & uncovered_mask
            uncovered_mask &= ~tsub_mask[best_block]

            m = new_covered
            while m:
                lsb = m & -m
                ts_idx = lsb.bit_length() - 1
                # For every block covering this ts_idx, decrease its gain
                cm = tsub_blocks_mask[ts_idx]
                while cm:
                    lsb2 = cm & -cm
                    b_idx2 = lsb2.bit_length() - 1
                    if not (selected_mask & (1 << b_idx2)):
                        gains[b_idx2] -= 1
                        heapq.heappush(heap, (-gains[b_idx2], b_idx2))
                    cm ^= lsb2
                m ^= lsb

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 5.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                # Find candidates using set intersection
                # Start with the most constrained unique tsub
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
                candidates = set(tsub_block_sets[min_ts])

                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]

                if not candidates:
                    continue

                candidates.discard(b)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)

                # Limit candidates to avoid too much work per swap attempt
                if len(cand_list) > 20:
                    cand_list = cand_list[:20]

                swapped = False
                for r in cand_list:
                    if r in block_set:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    # Main execution
    best_blocks = None
    best_len = float('inf')

    # Run multiple greedy restarts
    # Since precomputation is done, greedy is fast.
    # We can afford several restarts.
    num_restarts = 10

    for i in range(num_restarts):
        if time.time() > deadline - 30:  # Keep 30s for local search
            break

        seed = i * 12345 + 7
        blocks = greedy_cover(seed)

        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    if not best_blocks:
        return []

    # Apply local search to the best solution
    if time.time() < deadline - 2:
        best_blocks = local_search(best_blocks, 42)

    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]

    return final_blocks
