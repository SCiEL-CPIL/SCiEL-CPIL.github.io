"""Rewritten program: greedy construction + local search with redundancy pruning."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 110  # safety margin under 120s
    rng = random.Random(12345)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for fast counting
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # count[i] = number of blocks covering t-subset i
    count = [0] * n_tsubs

    def block_tsubs(block):
        """Return list of t-subset indices covered by this block."""
        return [tsub_index[ts] for ts in itertools.combinations(block, t)]

    # ---- Phase 1: Greedy construction ----
    blocks = []
    # uncovered set: t-subsets with count == 0
    uncovered = set(range(n_tsubs))

    # Precompute for greedy: for each point x, and given a partial block,
    # we need the marginal gain. We'll compute on the fly but cache per block-build.

    while uncovered and time.time() < deadline - 30:
        # Pick the "hardest" uncovered t-subset: one with fewest candidate blocks
        # For simplicity, pick one at random from uncovered (or min by some heuristic).
        # Use: pick the uncovered t-subset with the smallest number of points
        # that can extend it to a full block covering many uncovered subsets.
        # Simplified: pick random uncovered t-subset to start the block.
        start_ts = next(iter(uncovered))
        start_points = list(all_tsubs[start_ts])
        block = list(start_points)

        # Greedily add points to maximize marginal gain (number of newly covered t-subsets)
        while len(block) < k and time.time() < deadline - 30:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: number of uncovered t-subsets that include x and
                # (block minus one point) such that the t-subset is in uncovered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_index.get(ts, -1)
                    if idx != -1 and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        block.sort()
        # Update counts and uncovered
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            count[idx] += 1
            if count[idx] == 1:
                uncovered.discard(idx)
        blocks.append(block)

    # ---- Phase 2: Local search with redundancy elimination ----
    # Precompute for each block the list of t-subset indices it covers
    block_tsub_lists = [block_tsubs(b) for b in blocks]

    def is_redundant(bi):
        """Check if block bi is redundant (all its t-subsets covered by other blocks)."""
        for idx in block_tsub_lists[bi]:
            if count[idx] <= 1:
                return False
        return True

    def remove_block(bi):
        """Remove block bi from the solution, updating counts."""
        for idx in block_tsub_lists[bi]:
            count[idx] -= 1
        del blocks[bi]
        del block_tsub_lists[bi]

    def add_block(block):
        """Add a block to the solution, updating counts."""
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            count[idx] += 1
        blocks.append(block)
        block_tsub_lists.append(block_tsubs(block))

    # Iteratively try to remove redundant blocks
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        n = len(blocks)
        # Try removing redundant blocks
        for bi in range(n - 1, -1, -1):
            if is_redundant(bi):
                remove_block(bi)
                improved = True
        # Try swapping: replace a block with a better one
        if time.time() < deadline - 10:
            for bi in range(len(blocks)):
                if time.time() > deadline - 10:
                    break
                # Try to find a replacement block that covers at least the same
                # "critical" t-subsets (those with count == 1)
                critical = [idx for idx in block_tsub_lists[bi] if count[idx] == 1]
                if not critical:
                    continue
                # Try a few random candidate blocks
                for _ in range(5):
                    if time.time() > deadline - 10:
                        break
                    # Random block
                    cand = sorted(rng.sample(range(v), k))
                    cand_ts = set(block_tsub_lists and block_tsubs(cand) or [tsub_index[ts] for ts in itertools.combinations(cand, t)])
                    # Check if cand covers all critical t-subsets
                    if all(c in cand_ts for c in critical):
                        # Also check if cand is "better" (covers more total or same)
                        # Remove old, add new
                        old_list = block_tsub_lists[bi]
                        for idx in old_list:
                            count[idx] -= 1
                        for idx in cand_ts:
                            count[idx] += 1
                        blocks[bi] = cand
                        block_tsub_lists[bi] = list(cand_ts)
                        improved = True
                        break

    # Final pass: remove any remaining redundant blocks
    for bi in range(len(blocks) - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        if is_redundant(bi):
            remove_block(bi)

    return blocks
# EVOLVE-BLOCK-END
