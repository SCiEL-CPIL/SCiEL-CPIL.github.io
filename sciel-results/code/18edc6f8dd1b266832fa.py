import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering of t-subsets of v points using k-element blocks.
    Strategy: fast NumPy-based greedy set cover with local search (redundancy
    removal and 1-swaps) to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    random.seed(42)

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {s: i for i, s in enumerate(all_tsubsets)}

    # State: covered[t] = number of blocks covering t-subset t
    covered = [0] * n_tsubsets
    blocks = []

    def add_block(block):
        """Add a block and update coverage."""
        blocks.append(tuple(sorted(block)))
        for ts in itertools.combinations(block, t):
            covered[tsubset_index[ts]] += 1

    def remove_block(block_idx):
        """Remove a block and update coverage."""
        block = blocks[block_idx]
        for ts in itertools.combinations(block, t):
            covered[tsubset_index[ts]] -= 1
        del blocks[block_idx]

    def get_uncovered_count():
        return sum(1 for c in covered if c == 0)

    def get_uncovered_set():
        return {i for i, c in enumerate(covered) if c == 0}

    # Phase 1: Greedy construction
    # Use a faster greedy: for each uncovered t-subset, pick the block that covers
    # the most uncovered t-subsets, prioritizing blocks that share points.

    uncovered = set(range(n_tsubsets))

    while uncovered and time.time() < deadline - 30:
        # Pick the first uncovered t-subset as seed
        seed_ts = next(iter(uncovered))
        seed_points = all_tsubsets[seed_ts]

        # Build a candidate block by greedily adding points that cover most uncovered t-subsets
        block = list(seed_points[:t])
        remaining_need = k - len(block)

        if remaining_need > 0:
            # Score each candidate point by how many uncovered t-subsets it helps cover
            # A point x helps cover a t-subset s if s contains (block_elements ∪ {x}) in some way
            # More precisely: adding x to block creates new (t)-subsets that include x and (t-1) from block
            best_candidates = []
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                # These are t-subsets that contain x and (t-1) points from current block
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(combo + (x,)))
                    if ts in tsubset_index:
                        idx = tsubset_index[ts]
                        if idx in uncovered:
                            gain += 1
                best_candidates.append((gain, x))

            best_candidates.sort(reverse=True)
            for _, x in best_candidates[:remaining_need]:
                if x not in block:
                    block.append(x)

        block = tuple(sorted(block))

        # Update coverage
        newly_covered = 0
        for ts in itertools.combinations(block, t):
            idx = tsubset_index[ts]
            if idx in uncovered:
                uncovered.discard(idx)
                newly_covered += 1
            covered[idx] += 1

        blocks.append(block)

    # Phase 2: Local search - redundancy removal
    # Try to remove blocks that are redundant (all their t-subsets are covered by other blocks)
    removed_any = True
    while removed_any and time.time() < deadline - 15:
        removed_any = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            block = blocks[i]
            # Check if all t-subsets of this block are covered by other blocks
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[ts]
                if covered[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                remove_block(i)
                removed_any = True
                break  # Restart scan after removal

    # Phase 3: Local search - 1-swap improvement
    # Try replacing one block with another that covers more uncovered t-subsets
    # or enables removal of another block
    for iteration in range(20):
        if time.time() > deadline - 5:
            break

        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 5:
                break

            current_block = blocks[i]
            # Temporarily remove this block
            for ts in itertools.combinations(current_block, t):
                covered[tsubset_index[ts]] -= 1

            uncovered_after_removal = set()
            for idx, c in enumerate(covered):
                if c == 0:
                    uncovered_after_removal.add(idx)

            if not uncovered_after_removal:
                # Block was redundant, keep it removed
                remove_block(i)
                improved = True
                break

            # Try to find a replacement block that covers the most uncovered t-subsets
            best_gain = len(uncovered_after_removal)  # Minimum gain to be better
            best_new_block = None

            # Try blocks that share some points with current block
            candidate_points = set()
            for ts in list(uncovered_after_removal)[:100]:  # Sample
                candidate_points.update(all_tsubsets[ts])

            # Also try current block's points
            candidate_points.update(current_block)

            if candidate_points:
                # Try adding one different point
                for x in list(candidate_points):
                    if x in current_block:
                        continue
                    new_block = list(current_block)
                    new_block.remove(random.choice([p for p in new_block]))
                    new_block.append(x)
                    new_block = tuple(sorted(new_block))

                    # Count how many uncovered t-subsets this would cover
                    gain = 0
                    for ts in itertools.combinations(new_block, t):
                        idx = tsubset_index[ts]
                        if idx in uncovered_after_removal:
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_new_block = new_block

            if best_new_block is not None:
                # Restore coverage of old block
                for ts in itertools.combinations(current_block, t):
                    covered[tsubset_index[ts]] += 1
                # Add new block
                for ts in itertools.combinations(best_new_block, t):
                    covered[tsubset_index[ts]] += 1
                blocks[i] = best_new_block
                improved = True
                break
            else:
                # Restore coverage of old block
                for ts in itertools.combinations(current_block, t):
                    covered[tsubset_index[ts]] += 1

        if not improved:
            break

    # Phase 4: Final greedy fill for any remaining uncovered t-subsets
    uncovered = set(i for i, c in enumerate(covered) if c == 0)
    while uncovered and time.time() < deadline - 2:
        seed_ts = next(iter(uncovered))
        seed_points = all_tsubsets[seed_ts]
        block = list(seed_points)

        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(combo + (x,)))
                    if ts in tsubset_index and tsubset_index[ts] in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        block = tuple(sorted(block))
        for ts in itertools.combinations(block, t):
            idx = tsubset_index[ts]
            if idx in uncovered:
                uncovered.discard(idx)
            covered[idx] += 1
        blocks.append(block)

    # Final redundancy removal
    for i in range(len(blocks) - 1, -1, -1):
        if time.time() > deadline - 1:
            break
        block = blocks[i]
        redundant = True
        for ts in itertools.combinations(block, t):
            idx = tsubset_index[ts]
            if covered[idx] <= 1:
                redundant = False
                break
        if redundant:
            remove_block(i)

    # Convert to list of lists
    return [list(b) for b in blocks]
