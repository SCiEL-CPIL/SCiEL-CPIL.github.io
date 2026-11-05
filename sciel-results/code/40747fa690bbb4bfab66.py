import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map each block to the list of triplet IDs it covers (vectorized precompute)
    # Each block of size k covers C(k,t) triplets
    block_triples = []
    for block in all_blocks:
        tids = []
        for triplet in itertools.combinations(block, t):
            tids.append(triplet_to_id[triplet])
        block_triples.append(tids)

    # Map each triplet to the list of block indices covering it
    blocks_covering = [[] for _ in range(n_triplets)]
    for b_idx, tids in enumerate(block_triples):
        for tid in tids:
            blocks_covering[tid].append(b_idx)

    def greedy_set_cover(seed):
        rng = random.Random(seed)
        # gain[b] = number of currently uncovered triplets covered by block b
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(n_triplets, dtype=np.bool_)

        selected_blocks = []

        while is_uncovered.any():
            if time.time() > deadline - 25:
                break

            max_gain = np.max(gain)
            if max_gain == 0:
                break

            # Pick random among max-gain blocks for diversity
            candidates = np.where(gain == max_gain)[0]
            b_idx = rng.choice(candidates.tolist())

            selected_blocks.append(b_idx)

            # Update: for each triplet in this block that was uncovered,
            # mark covered and decrement gain for all blocks covering it
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b in blocks_covering[t_idx]:
                        gain[other_b] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        """Remove redundant blocks (all their triplets covered by other blocks)."""
        if not selected_blocks:
            return selected_blocks

        cover_count = np.zeros(n_triplets, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        active = [True] * len(selected_blocks)
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(selected_blocks):
                if not active[i]:
                    continue
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True

        return [b_idx for i, b_idx in enumerate(selected_blocks) if active[i]]

    # Phase 1: Multiple greedy restarts
    best_cover = None
    best_count = float('inf')
    seed = 0
    greedy_time_budget = 55.0  # spend ~55s on greedy restarts

    while time.time() < start_time + greedy_time_budget:
        if time.time() > deadline - 30:
            break
        cover = greedy_set_cover(seed)
        cover = reduce_cover(cover)
        if len(cover) < best_count:
            best_count = len(cover)
            best_cover = cover
        seed += 1

    if best_cover is None:
        best_cover = greedy_set_cover(0)
        best_cover = reduce_cover(best_cover)

    # Phase 2: Local search - 2-for-1 swap
    # Convert to working with block indices
    current_blocks = list(best_cover)
    n_blocks_cur = len(current_blocks)

    # Precompute triplet ID sets for each current block
    block_triplet_sets = []
    for b_idx in current_blocks:
        block_triplet_sets.append(set(block_triples[b_idx]))

    # Cover count
    cover_count = np.zeros(n_triplets, dtype=np.int32)
    for b_idx in current_blocks:
        for t_idx in block_triples[b_idx]:
            cover_count[t_idx] += 1

    def try_remove_redundant():
        nonlocal current_blocks, block_triplet_sets, cover_count
        changed = False
        new_blocks = []
        new_sets = []
        for i, b_idx in enumerate(current_blocks):
            redundant = True
            for t_idx in block_triples[b_idx]:
                if cover_count[t_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for t_idx in block_triples[b_idx]:
                    cover_count[t_idx] -= 1
                changed = True
            else:
                new_blocks.append(b_idx)
                new_sets.append(block_triplet_sets[i])
        if changed:
            current_blocks = new_blocks
            block_triplet_sets = new_sets
        return changed

    def try_swap_2_for_1():
        nonlocal current_blocks, block_triplet_sets, cover_count
        n = len(current_blocks)
        if n < 2:
            return False

        for i in range(n):
            if time.time() > deadline - 5:
                return False
            ids_i = block_triplet_sets[i]

            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return False
                ids_j = block_triplet_sets[j]

                # Find critical triplets: covered only by block i and/or block j
                critical_points = set()

                for tid in ids_i:
                    c = cover_count[tid]
                    cov_i = 1
                    cov_j = 1 if tid in ids_j else 0
                    if c == cov_i + cov_j:
                        critical_points.update(all_triplets[tid])

                for tid in ids_j:
                    if tid in ids_i:
                        continue
                    c = cover_count[tid]
                    if c == 1:
                        critical_points.update(all_triplets[tid])

                if not critical_points:
                    # Both redundant
                    for t_idx in block_triples[current_blocks[i]]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_triples[current_blocks[j]]:
                        cover_count[t_idx] -= 1
                    current_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                    block_triplet_sets = [s for idx, s in enumerate(block_triplet_sets) if idx != i and idx != j]
                    return True

                if len(critical_points) <= k:
                    new_block_set = set(critical_points)
                    # Fill remaining slots greedily by point frequency
                    point_freq = np.zeros(v, dtype=np.int32)
                    for idx, b_idx in enumerate(current_blocks):
                        if idx == i or idx == j:
                            continue
                        for p in all_blocks[b_idx]:
                            point_freq[p] += 1

                    candidates = [p for p in range(v) if p not in new_block_set]
                    candidates.sort(key=lambda p: point_freq[p], reverse=True)

                    for p in candidates:
                        if len(new_block_set) >= k:
                            break
                        new_block_set.add(p)

                    new_block = tuple(sorted(new_block_set))

                    # Verify: new_block must cover all critical triplets
                    # All critical points are in new_block, so all critical triplets (subsets of critical_points) are covered
                    valid = True
                    for tid in ids_i | ids_j:
                        if cover_count[tid] <= 1:
                            # This triplet is critical, check if it's in new_block
                            trip = all_triplets[tid]
                            if not all(p in new_block_set for p in trip):
                                valid = False
                                break
                    if not valid:
                        continue

                    # Remove blocks i and j, add new_block
                    for t_idx in block_triples[current_blocks[i]]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_triples[current_blocks[j]]:
                        cover_count[t_idx] -= 1

                    # Find or create new block index
                    # Check if this block already exists in all_blocks
                    new_b_idx = None
                    # Use a set for O(1) lookup - but we don't have one precomputed
                    # Just scan (rare case)
                    for b_idx in range(n_blocks):
                        if all_blocks[b_idx] == new_block:
                            new_b_idx = b_idx
                            break
                    if new_b_idx is None:
                        # Should not happen since new_block is a valid k-subset
                        continue

                    for t_idx in block_triples[new_b_idx]:
                        cover_count[t_idx] += 1

                    current_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                    block_triplet_sets = [s for idx, s in enumerate(block_triplet_sets) if idx != i and idx != j]
                    current_blocks.append(new_b_idx)
                    block_triplet_sets.append(set(block_triples[new_b_idx]))
                    return True

        return False

    # Main local search loop
    for iteration in range(300):
        if time.time() > deadline - 3:
            break

        progress = False

        # Remove redundants
        while try_remove_redundant():
            progress = True

        # Try 2-for-1 swap
        if not progress:
            if try_swap_2_for_1():
                progress = True

        if not progress:
            break

    # Final redundancy pass
    try_remove_redundant()

    # Convert back to list of blocks
    result = [list(all_blocks[b_idx]) for b_idx in current_blocks]

    # Verify validity (exhaustive check)
    covered = set()
    for block in result:
        for triplet in itertools.combinations(block, t):
            covered.add(triplet)

    if len(covered) < n_triplets:
        # Fallback: return the best greedy result without local search
        return [list(all_blocks[b_idx]) for b_idx in best_cover]

    return result
