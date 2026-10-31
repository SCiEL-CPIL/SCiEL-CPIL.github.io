"""Structural rewrite: greedy + fast local search (redundancy removal, 1-for-1 swaps)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    deadline = time.time() + 110  # safety margin before 120s kill
    rng = random.Random(12345)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Greedy construction
    def greedy_cover():
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered:
            # Start with a random uncovered t-subset
            start = min(uncovered)
            start_tsub = all_tsubs[start]
            block = set(start_tsub)
            while len(block) < k:
                # Choose point that covers most uncovered t-subsets
                best_x = None
                best_gain = -1
                # Candidate points: those in uncovered t-subsets that extend block
                candidates = set()
                for idx in uncovered:
                    ts = all_tsubs[idx]
                    if len(set(ts) & block) >= t - 1:
                        for p in ts:
                            if p not in block:
                                candidates.add(p)
                if not candidates:
                    break
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(rest + (x,)))
                        if new_ts in tsub_to_idx and tsub_to_idx[new_ts] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.add(best_x)
            block = sorted(block)
            blocks.append(block)
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                uncovered.discard(idx)
        return blocks

    blocks = greedy_cover()

    # Local search: redundancy removal and 1-for-1 swaps
    def build_cover_map(block_list):
        """Map each t-subset to list of block indices covering it."""
        cover_map = {}
        for bi, block in enumerate(block_list):
            for ts in itertools.combinations(block, t):
                cover_map.setdefault(ts, []).append(bi)
        return cover_map

    def is_valid(block_list):
        """Check if every t-subset is covered by at least one block."""
        cover_map = build_cover_map(block_list)
        return len(cover_map) == n_tsubs

    def try_remove(blocks, bi):
        """Try to remove block bi if it's redundant."""
        block = blocks[bi]
        # Count how many t-subsets are uniquely covered by this block
        unique_count = 0
        for ts in itertools.combinations(block, t):
            covering = [j for j, b in enumerate(blocks) if j != bi and set(ts) <= set(b)]
            if not covering:
                unique_count += 1
                break
        return unique_count == 0

    def try_swap(blocks, bi, bj):
        """Try to replace block bj with a new block that covers unique t-subsets of bj."""
        # Find t-subsets uniquely covered by bj
        unique_tsubs = []
        for ts in itertools.combinations(blocks[bj], t):
            covering = [j for j, b in enumerate(blocks) if j != bj and set(ts) <= set(b)]
            if not covering:
                unique_tsubs.append(ts)
        if not unique_tsubs:
            return None
        # Try to find a new block covering all unique_tsubs (or as many as possible)
        # Start with union of points in unique_tsubs
        candidate = set()
        for ts in unique_tsubs:
            candidate.update(ts)
        if len(candidate) > k:
            # Need to prune; pick k points that cover most unique_tsubs
            # Greedy set cover on unique_tsubs
            remaining = list(unique_tsubs)
            selected = set()
            while len(selected) < k and remaining:
                best_p = None
                best_cov = 0
                for p in candidate:
                    if p in selected:
                        continue
                    cov = sum(1 for ts in remaining if p in ts)
                    if cov > best_cov:
                        best_cov = cov
                        best_p = p
                if best_p is None:
                    break
                selected.add(best_p)
                remaining = [ts for ts in remaining if best_p not in ts]
            candidate = selected
        if len(candidate) < k:
            # Pad with random points
            for p in range(v):
                if len(candidate) >= k:
                    break
                if p not in candidate:
                    candidate.add(p)
        new_block = sorted(candidate)
        # Verify new block covers all unique_tsubs
        for ts in unique_tsubs:
            if not set(ts) <= set(new_block):
                return None
        # Check new block is distinct
        if new_block in blocks:
            return None
        return new_block

    # Main local search loop
    best_blocks = blocks
    best_count = len(blocks)
    no_improve_streak = 0
    max_no_improve = 200

    while time.time() < deadline - 5:
        if no_improve_streak >= max_no_improve:
            break

        improved = False
        n_blocks = len(best_blocks)

        # Phase 1: Remove redundant blocks
        for bi in range(len(best_blocks)):
            if time.time() > deadline - 5:
                break
            if try_remove(best_blocks, bi):
                best_blocks = [b for j, b in enumerate(best_blocks) if j != bi]
                improved = True
                no_improve_streak = 0
                break
        else:
            # Phase 2: Try 1-for-1 swaps
            for bi in range(len(best_blocks)):
                if time.time() > deadline - 5:
                    break
                for bj in range(len(best_blocks)):
                    if bi == bj:
                        continue
                    new_block = try_swap(best_blocks, bi, bj)
                    if new_block is not None:
                        # Apply swap: replace block bj with new_block
                        test_blocks = list(best_blocks)
                        test_blocks[bj] = new_block
                        if is_valid(test_blocks) and len(test_blocks) < len(best_blocks):
                            best_blocks = test_blocks
                            improved = True
                            no_improve_streak = 0
                            break
                if improved:
                    break
            if not improved:
                no_improve_streak += 1

        if improved and len(best_blocks) < best_count:
            best_count = len(best_blocks)
            no_improve_streak = 0

    return best_blocks
# EVOLVE-BLOCK-END
