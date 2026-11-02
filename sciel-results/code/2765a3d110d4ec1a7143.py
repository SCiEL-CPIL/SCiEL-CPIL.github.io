"""Greedy covering with fast gain computation, then pairwise merge local search."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Precompute all t-subsets as tuples
    all_tsubsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for fast lookup
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsets = len(all_tsubsets)

    # Track which t-subsets are uncovered: use a boolean array
    uncovered_mask = bytearray([1] * n_tsets)  # 1 = uncovered
    uncovered_count = n_tsets

    blocks = []

    # Greedy phase: repeatedly build a block
    while uncovered_count > 0:
        if time.time() > deadline - 30.0:
            # Not enough time for more blocks; break and rely on what we have
            # But we need a valid cover, so we must continue greedily until covered
            # Actually, we must ensure validity. Let's just continue but be fast.
            pass

        # Find an uncovered t-subset to seed the block
        # Pick the first uncovered one (arbitrary but deterministic)
        seed_idx = -1
        for i in range(n_tsets):
            if uncovered_mask[i]:
                seed_idx = i
                break
        if seed_idx == -1:
            break
        block = list(all_tsubsets[seed_idx])

        # Build the block by adding points one at a time
        # For each candidate point not in block, compute how many new t-subsets it covers
        # A new t-subset covered by adding point x to block B is a subset S of B∪{x} of size t
        # that contains x and the other t-1 elements from B, and is currently uncovered.

        while len(block) < k:
            # Compute gain for each candidate
            best_gain = -1
            best_point = -1
            # We can use combinations of block of size t-1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets that contain x and t-1 elements from block
                gain = 0
                if t - 1 == 0:
                    # t=1: adding x covers the t-subset (x,) if uncovered
                    idx = tset_index.get((x,))
                    if idx is not None and uncovered_mask[idx]:
                        gain = 1
                else:
                    for rest in itertools.combinations(block, t - 1):
                        # Form the t-subset: rest + (x,)
                        ts = tuple(sorted(rest + (x,)))
                        idx = tset_index.get(ts)
                        if idx is not None and uncovered_mask[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                # No candidate improves; pick any
                for x in range(v):
                    if x not in block_set:
                        best_point = x
                        break
            block.append(best_point)

        block.sort()
        blocks.append(block)

        # Update uncovered mask: mark all t-subsets of this block as covered
        for ts in itertools.combinations(block, t):
            idx = tset_index.get(ts)
            if idx is not None and uncovered_mask[idx]:
                uncovered_mask[idx] = 0
                uncovered_count -= 1

    # Local search: pairwise merge
    # For each pair of blocks, try to merge them into one block that covers all
    # t-subsets that were uniquely covered by one of the two.
    # If successful, replace the two blocks with the merged one.

    # Precompute for each block which t-subsets it covers
    def block_tsets(block):
        return [tset_index[ts] for ts in itertools.combinations(block, t)]

    # Count how many times each t-subset is covered
    cover_count = [0] * n_tsets
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = tset_index[ts]
            cover_count[idx] += 1

    # Function to try merging two blocks
    def try_merge(b1, b2):
        """Try to find a k-subset that covers all t-subsets uniquely covered by b1 or b2."""
        # Collect t-subsets uniquely covered by b1 or b2
        required = set()
        for ts in itertools.combinations(b1, t):
            idx = tset_index[ts]
            if cover_count[idx] == 1:
                required.add(idx)
        for ts in itertools.combinations(b2, t):
            idx = tset_index[ts]
            if cover_count[idx] == 1:
                required.add(idx)

        if not required:
            # No unique t-subsets; we can just use any k-subset, but we need to cover
            # all t-subsets that were covered by b1 or b2 that are still needed.
            # Actually, if no unique ones, the t-subsets are covered by others too.
            # We can try to find a block that covers as many as possible.
            # For simplicity, skip if no required sets.
            return None

        # The merged block must contain all elements from required t-subsets? No,
        # it must contain each required t-subset as a subset.
        # So the merged block must be a k-subset that contains every required t-subset.
        # This means the union of all required t-subsets must have size <= k.
        union = set()
        for idx in required:
            union.update(all_tsubsets[idx])
        if len(union) > k:
            return None

        # If union size <= k, we can form a block from union plus extra points.
        # But we also need to make sure we don't break other covers.
        # Since we're only removing b1 and b2, we need to ensure all t-subsets
        # that were covered by b1 or b2 are still covered by the new block or others.
        # The required set handles the unique ones. For non-unique ones, they're
        # covered by at least one other block, so we're fine.
        # But wait: some t-subsets covered by b1 might be covered by b2 and no other.
        # Those would have cover_count == 2, and after removing both, they'd be uncovered.
        # We need to handle this: the merged block must cover all t-subsets that
        # have cover_count <= 2 and are covered by b1 or b2.

        # Let's redo: collect all t-subsets covered by b1 or b2 that have cover_count <= 2
        required2 = set()
        for ts in itertools.combinations(b1, t):
            idx = tset_index[ts]
            if cover_count[idx] <= 2:
                required2.add(idx)
        for ts in itertools.combinations(b2, t):
            idx = tset_index[ts]
            if cover_count[idx] <= 2:
                required2.add(idx)

        if not required2:
            return None

        union2 = set()
        for idx in required2:
            union2.update(all_tsubsets[idx])
        if len(union2) > k:
            return None

        # Form the merged block
        merged = sorted(union2)
        if len(merged) > k:
            return None
        # Pad with arbitrary points if needed
        if len(merged) < k:
            for x in range(v):
                if x not in union2:
                    merged.append(x)
                if len(merged) == k:
                    break
        merged.sort()

        # Verify that the merged block covers all required2 t-subsets
        merged_set = set(merged)
        for idx in required2:
            if not all(x in merged_set for x in all_tsubsets[idx]):
                return None

        return merged

    # Perform merges
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Recompute cover_count
        cover_count = [0] * n_tsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = tset_index[ts]
                cover_count[idx] += 1

        # Try all pairs
        n_blocks = len(blocks)
        for i in range(n_blocks):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, n_blocks):
                merged = try_merge(blocks[i], blocks[j])
                if merged is not None:
                    # Replace blocks[i] and blocks[j] with merged
                    # Remove j first, then i
                    del blocks[j]
                    blocks[i] = merged
                    improved = True
                    break
            if improved:
                break

    return blocks
