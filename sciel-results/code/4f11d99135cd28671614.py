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
        Local search with swaps, deletions, and 2-for-2 exchanges.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        # Precompute: for each block, the set of tsubs it covers (as a set for fast ops)
        # Already have block_tsubs as lists.

        def try_remove(block_set, cover_count, b):
            """Try to remove block b if redundant. Returns True if removed."""
            for ts_idx in block_tsubs[b]:
                if cover_count[ts_idx] == 1:
                    return False
            block_set.remove(b)
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] -= 1
            return True

        def try_1for1_swap(block_set, cover_count, b, rng):
            """Try to swap block b with another block. Returns True if swapped."""
            unique_tsubs = []
            for ts_idx in block_tsubs[b]:
                if cover_count[ts_idx] == 1:
                    unique_tsubs.append(ts_idx)

            if not unique_tsubs:
                return try_remove(block_set, cover_count, b)

            # Find candidates using set intersection
            min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
            candidates = set(tsub_block_sets[min_ts])

            for ts_idx in unique_tsubs[1:]:
                if not candidates:
                    break
                candidates &= tsub_block_sets[ts_idx]

            if not candidates:
                return False

            candidates.discard(b)
            if not candidates:
                return False

            cand_list = list(candidates)
            rng.shuffle(cand_list)

            if len(cand_list) > 30:
                cand_list = cand_list[:30]

            for r in cand_list:
                if r in block_set:
                    continue

                block_set.remove(b)
                for ts_idx in block_tsubs[b]:
                    cover_count[ts_idx] -= 1

                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1

                return True

            return False

        def try_2for2_exchange(block_set, cover_count, b1, b2, rng):
            """
            Try to replace blocks b1 and b2 with two other blocks.
            The new pair must cover all tsubs that were uniquely covered by b1 or b2.
            """
            # Find tsubs uniquely covered by b1 or b2
            unique_tsubs = []
            for ts_idx in block_tsubs[b1]:
                if cover_count[ts_idx] == 1:
                    unique_tsubs.append(ts_idx)
            for ts_idx in block_tsubs[b2]:
                if cover_count[ts_idx] == 1:
                    unique_tsubs.append(ts_idx)

            if not unique_tsubs:
                # Both are redundant
                try_remove(block_set, cover_count, b1)
                try_remove(block_set, cover_count, b2)
                return True

            # Find candidate blocks that cover all unique_tsubs
            min_ts = min(unique_tsubs, key=lambda x: len(tsub_block_sets[x]))
            candidates = set(tsub_block_sets[min_ts])

            for ts_idx in unique_tsubs[1:]:
                if not candidates:
                    break
                candidates &= tsub_block_sets[ts_idx]

            if not candidates:
                return False

            candidates.discard(b1)
            candidates.discard(b2)

            if len(candidates) < 2:
                return False

            cand_list = list(candidates)
            rng.shuffle(cand_list)

            # Limit to avoid combinatorial explosion
            max_cands = min(len(cand_list), 15)
            cand_list = cand_list[:max_cands]

            # Try pairs of candidates
            for i in range(max_cands):
                if time.time() > deadline - 5.0:
                    return False
                r1 = cand_list[i]
                if r1 in block_set:
                    continue
                for j in range(i + 1, max_cands):
                    r2 = cand_list[j]
                    if r2 in block_set:
                        continue
                    # Check that r1 and r2 together cover all unique_tsubs
                    # (They individually cover all, so together they do too)
                    # But we need to make sure no tsub becomes uncovered
                    # Since r1 and r2 both cover all unique_tsubs, and unique_tsubs
                    # are the only ones at risk, this should be safe.
                    # However, we should verify that removing b1, b2 and adding r1, r2
                    # doesn't leave any tsub uncovered.
                    # The unique_tsubs are covered by both r1 and r2, so they're fine.
                    # Other tsubs had cover_count >= 2 before, so removing b1, b2
                    # reduces by at most 2, leaving >= 0. But we need >= 1.
                    # Actually, tsubs with cover_count == 2 that are covered by b1 or b2
                    # would drop to 1, which is still covered.
                    # Tsubs with cover_count == 1 are in unique_tsubs, covered by r1, r2.
                    # So it should be safe.
                    block_set.remove(b1)
                    for ts_idx in block_tsubs[b1]:
                        cover_count[ts_idx] -= 1
                    block_set.remove(b2)
                    for ts_idx in block_tsubs[b2]:
                        cover_count[ts_idx] -= 1
                    block_set.add(r1)
                    for ts_idx in block_tsubs[r1]:
                        cover_count[ts_idx] += 1
                    block_set.add(r2)
                    for ts_idx in block_tsubs[r2]:
                        cover_count[ts_idx] += 1
                    return True

            return False

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in list(block_set):
                if try_remove(block_set, cover_count, b):
                    to_remove.append(b)

            if to_remove:
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
                if try_1for1_swap(block_set, cover_count, b, rng):
                    improved = True
                    break

            if improved:
                continue

            # Phase 3: Try 2-for-2 exchanges
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for i in range(len(current_blocks)):
                if time.time() > deadline - 5.0:
                    break
                b1 = current_blocks[i]
                if b1 not in block_set:
                    continue
                for j in range(i + 1, len(current_blocks)):
                    b2 = current_blocks[j]
                    if b2 not in block_set:
                        continue
                    if try_2for2_exchange(block_set, cover_count, b1, b2, rng):
                        improved = True
                        break
                if improved:
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
        if time.time() > deadline - 40:  # Keep 40s for local search
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

        # If local search improved, try another round with a different seed
        if time.time() < deadline - 2 and len(best_blocks) < best_len:
            best_len = len(best_blocks)
            best_blocks = local_search(best_blocks, 137)

    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]

    return final_blocks
