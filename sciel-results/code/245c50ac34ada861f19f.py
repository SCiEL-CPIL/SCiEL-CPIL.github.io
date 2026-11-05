import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Safety margin before 120s limit

    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    # Generate all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Map block index to list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[triple] for triple in itertools.combinations(b, t)]
        block_triples.append(ts)

    # Map triple index to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    def greedy_randomized(seed):
        random.seed(seed)
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)

        selected_blocks = []

        while is_uncovered.any():
            max_gain = np.max(gain)
            if max_gain == 0:
                break

            candidates = np.where(gain == max_gain)[0]
            b_idx = random.choice(candidates.tolist())

            selected_blocks.append(b_idx)

            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1

        return selected_blocks

    def reduce_cover(selected_blocks):
        cover_count = np.zeros(num_triples, dtype=np.int32)
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

    def local_search(cover, seed, time_budget=10.0):
        """
        Local search: try to swap one block for another to reduce size or improve coverage.
        Here we focus on removing redundant blocks first, then trying 1-for-1 swaps.
        """
        random.seed(seed)
        if not cover:
            return cover

        cover_set = set(cover)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        # First, remove redundant blocks
        cover = reduce_cover(cover)
        cover_set = set(cover)
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in cover:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1

        # Try swaps: remove one block, add another
        # To make this efficient, we'll try random pairs
        best_cover = list(cover)
        best_len = len(best_cover)

        # Precompute for each block in cover, which triples it uniquely covers
        # This helps in deciding which block to remove

        # Simple approach: iterate over blocks in cover, try to remove each one,
        # then see if we can add a block that covers the newly uncovered triples
        # This is expensive, so we do it selectively

        # Alternative: try to find a block that covers many uncovered triples
        # after removing a block

        # Let's do a simpler local search: try to replace a block with a better one
        # that covers more currently uncovered triples (if any exist after removal)

        # Actually, a better approach for covering designs:
        # 1. Remove a block
        # 2. If coverage is still complete, keep it removed
        # 3. If coverage is broken, try to add a single block that fixes it
        # 4. If no single block fixes it, restore the original block

        # Let's implement this
        max_iterations = 500
        for _ in range(max_iterations):
            if time.time() - start_time > timeout - time_budget:
                break

            if not cover:
                break

            # Pick a random block to try removing
            idx = random.randint(0, len(cover) - 1)
            b_remove = cover[idx]

            # Remove it
            cover.remove(b_remove)

            # Check coverage
            cover_count = np.zeros(num_triples, dtype=np.int32)
            for b_idx in cover:
                for t_idx in block_triples[b_idx]:
                    cover_count[t_idx] += 1

            uncovered = np.where(cover_count == 0)[0]

            if len(uncovered) == 0:
                # Coverage is still complete, keep the removal
                best_cover = list(cover)
                best_len = len(cover)
                continue

            # Coverage is broken, try to add a block that covers all uncovered triples
            # Find blocks that cover all uncovered triples
            # This is expensive, so we try a few random candidates

            found_fix = False
            # Get the set of uncovered triples
            uncovered_set = set(uncovered.tolist())

            # Try random blocks
            for _ in range(20):
                b_add = random.randint(0, num_blocks - 1)
                if b_add in cover:
                    continue

                # Check if b_add covers all uncovered triples
                covers_all = True
                for t_idx in block_triples[b_add]:
                    if t_idx in uncovered_set:
                        uncovered_set.discard(t_idx)
                    if not uncovered_set:
                        break

                if not uncovered_set:
                    # b_add covers all uncovered triples
                    cover.append(b_add)
                    found_fix = True
                    break

            if not found_fix:
                # Restore the removed block
                cover.append(b_remove)

        return cover

    best_cover = None
    seed = 0
    num_restarts = 0

    while True:
        if time.time() - start_time > timeout:
            break

        # Run greedy
        cover = greedy_randomized(seed)
        # Reduce
        cover = reduce_cover(cover)

        if best_cover is None or len(cover) < len(best_cover):
            best_cover = cover

        # Occasionally do local search on the best cover
        if num_restarts > 0 and num_restarts % 3 == 0 and best_cover is not None:
            ls_cover = local_search(best_cover, seed + 1000)
            if len(ls_cover) < len(best_cover):
                best_cover = ls_cover

        seed += 1
        num_restarts += 1

        # If we've been running for a while and haven't improved, increase local search effort
        if num_restarts > 50:
            break

    # Final local search on the best cover
    if best_cover is not None:
        ls_cover = local_search(best_cover, seed + 2000, time_budget=5.0)
        if len(ls_cover) < len(best_cover):
            best_cover = ls_cover

    # Convert indices back to blocks
    if best_cover is None:
        cover = greedy_randomized(0)
        return [list(all_blocks[b_idx]) for b_idx in cover]

    return [list(all_blocks[b_idx]) for b_idx in best_cover]
