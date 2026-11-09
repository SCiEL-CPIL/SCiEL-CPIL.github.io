"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin before 150s kill
    rng = random.Random(42)

    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # For each point, list of t-subset indices containing that point
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_tsubs[p].append(idx)

    # For each t-subset, list of points in it
    tsub_points = [list(ts) for ts in all_tsubs]

    def greedy_cover():
        """Randomized greedy set cover with reverse indexing."""
        coverage = [0] * n_tsubs
        blocks = []

        while True:
            if time.time() > deadline:
                return None

            # Find an uncovered t-subset
            uncovered_idx = None
            for i in range(n_tsubs):
                if coverage[i] == 0:
                    uncovered_idx = i
                    break
            if uncovered_idx is None:
                break

            # Build a block that covers this t-subset and maximizes new coverage
            seed_tsub = all_tsubs[uncovered_idx]
            block_set = set(seed_tsub)

            # Greedily add points
            for _ in range(k - t):
                if time.time() > deadline:
                    return None

                best_point = None
                best_gain = -1
                candidates = [p for p in range(v) if p not in block_set]

                # Shuffle for randomization
                rng.shuffle(candidates)

                for p in candidates:
                    # Calculate gain: how many new t-subsets this point covers
                    # given the current block
                    gain = 0
                    for tsub_idx in point_tsubs[p]:
                        if coverage[tsub_idx] == 0:
                            # Check if this t-subset would be covered by adding p
                            # i.e., all other elements are already in block_set
                            ts = all_tsubs[tsub_idx]
                            others = [x for x in ts if x != p]
                            if all(o in block_set for o in others):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                    if gain == n_tsubs:  # max possible
                        break

                if best_point is not None:
                    block_set.add(best_point)

            block = sorted(block_set)
            blocks.append(block)

            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage[idx] += 1

        return blocks

    def local_search_remove(blocks):
        """Try to remove redundant blocks."""
        if not blocks:
            return blocks

        # Build coverage map: t-subset -> list of block indices covering it
        coverage_count = [0] * n_tsubs
        block_tsubs = []  # for each block, list of t-subset indices it covers

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        # Try to remove each block if it doesn't break coverage
        keep = [True] * len(blocks)
        removed = 0

        for i in range(len(blocks)):
            if not keep[i]:
                continue
            # Check if all t-subsets in this block are covered by other blocks
            can_remove = True
            for tsub_idx in block_tsubs[i]:
                if coverage_count[tsub_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                keep[i] = False
                removed += 1
                # Update coverage
                for tsub_idx in block_tsubs[i]:
                    coverage_count[tsub_idx] -= 1

        return [blocks[i] for i in range(len(blocks)) if keep[i]]

    def local_search_replace(blocks):
        """Try to replace blocks with better ones to reduce total count.
        
        For each block, try removing it and then greedily covering any newly
        uncovered t-subsets. If the net change is negative (fewer blocks), keep
        the replacement.
        """
        if not blocks:
            return blocks

        coverage_count = [0] * n_tsubs
        block_tsubs = []

        for block in blocks:
            tsubs_in_block = []
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage_count[idx] += 1
                tsubs_in_block.append(idx)
            block_tsubs.append(tsubs_in_block)

        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks

                # Temporarily remove block i
                for tsub_idx in block_tsubs[i]:
                    coverage_count[tsub_idx] -= 1

                # Find newly uncovered t-subsets
                uncovered = [j for j in range(n_tsubs) if coverage_count[j] == 0]

                if not uncovered:
                    # Block i was redundant, remove it
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    improved = True
                    # Re-index: need to rebuild block_tsubs for remaining blocks
                    # Actually, let's just rebuild from scratch for simplicity
                    break

                # Try to cover uncovered t-subsets with minimal new blocks
                new_blocks = []
                covered_set = set(uncovered)

                while covered_set:
                    if time.time() > deadline:
                        # Revert: add block i back
                        for tsub_idx in block_tsubs[i]:
                            coverage_count[tsub_idx] += 1
                        return blocks

                    # Pick an uncovered t-subset
                    seed_idx = min(covered_set)
                    seed_tsub = all_tsubs[seed_idx]
                    block_set = set(seed_tsub)

                    # Greedily add points to maximize coverage of uncovered
                    for _ in range(k - t):
                        best_point = None
                        best_gain = -1
                        candidates = [p for p in range(v) if p not in block_set]
                        rng.shuffle(candidates)

                        for p in candidates:
                            gain = 0
                            for tsub_idx in point_tsubs[p]:
                                if tsub_idx in covered_set:
                                    ts = all_tsubs[tsub_idx]
                                    others = [x for x in ts if x != p]
                                    if all(o in block_set for o in others):
                                        gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = p

                        if best_point is not None:
                            block_set.add(best_point)

                    new_block = sorted(block_set)
                    new_blocks.append(new_block)

                    # Update covered_set
                    for ts in itertools.combinations(new_block, t):
                        idx = tsub_to_idx[ts]
                        if idx in covered_set:
                            covered_set.discard(idx)

                # Check if new_blocks is smaller than the removed block count (1)
                if len(new_blocks) < 1:
                    # This shouldn't happen since we had uncovered, but just in case
                    pass
                else:
                    # Replace block i with new_blocks
                    # But we need to check if this actually helps
                    # Net change: -1 + len(new_blocks)
                    if len(new_blocks) >= 1:
                        # No improvement or worse, revert
                        for tsub_idx in block_tsubs[i]:
                            coverage_count[tsub_idx] += 1
                        continue
                    else:
                        # Should not reach here
                        continue

                # If we get here, new_blocks is empty (all covered by removing block i)
                # This is already handled above
                break

            if improved:
                # Rebuild coverage_count and block_tsubs
                coverage_count = [0] * n_tsubs
                block_tsubs = []
                for block in blocks:
                    tsubs_in_block = []
                    for ts in itertools.combinations(block, t):
                        idx = tsub_to_idx[ts]
                        coverage_count[idx] += 1
                        tsubs_in_block.append(idx)
                    block_tsubs.append(tsubs_in_block)

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Run greedy with randomization and local search
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        blocks = greedy_cover()
        if blocks is None:
            continue

        # Local search to remove redundant blocks
        blocks = local_search_remove(blocks)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()

    # Try local search replace on the best solution
    if best_blocks is not None and time.time() < deadline:
        improved_blocks = local_search_replace(best_blocks.copy())
        if improved_blocks is not None and len(improved_blocks) < best_count:
            best_count = len(improved_blocks)
            best_blocks = improved_blocks

    if best_blocks is not None:
        return best_blocks

    # Fallback: simple greedy without local search
    blocks = greedy_cover()
    if blocks is not None:
        return blocks

    # Last resort: original greedy
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
# EVOLVE-BLOCK-END
