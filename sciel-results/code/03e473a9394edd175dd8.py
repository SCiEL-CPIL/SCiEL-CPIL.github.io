"""
Covering design C(24,7,3) - greedy construction with local search.

Strategy:
1. Fast greedy construction using bitmasks for t-subsets.
2. Local search: try to remove blocks that are redundant (their t-subsets are
   all covered by other blocks). Iterate until no more removals possible.
3. If time permits, try a few restarts with different greedy orderings.
"""
import time
import itertools
import random
from collections import defaultdict

# Precompute t-subsets as bitmasks for fast membership testing
def _build_tsubsets(v, t):
    """Return list of all t-subsets as frozensets and a dict mapping each to its index."""
    subs = list(itertools.combinations(range(v), t))
    return subs

def _block_tsubsets(block, t, v):
    """Return list of t-subsets (as tuples) contained in block."""
    return list(itertools.combinations(block, t))


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsub_set = set(all_tsubs)
    n_tsubs = len(all_tsubs)

    # Map each t-subset to a bit index for fast coverage tracking
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Greedy construction
    def greedy_construct(seed_order=None):
        """Greedy: repeatedly pick the block that covers the most uncovered t-subsets."""
        # Track which t-subsets are uncovered using a set of indices
        uncovered = set(range(n_tsubs))
        blocks = []
        seen_blocks = set()

        # Precompute for each point, which t-subsets contain it
        point_to_tsubs = defaultdict(list)
        for i, ts in enumerate(all_tsubs):
            for p in ts:
                point_to_tsubs[p].append(i)

        # For greedy: maintain coverage count per t-subset
        cover_count = [0] * n_tsubs

        while uncovered:
            if time.time() > deadline:
                break

            # Find the block that maximizes coverage of uncovered t-subsets.
            # Heuristic: start from an uncovered t-subset, build a block around it.
            # Pick the uncovered t-subset with the "most constrained" points? 
            # Simpler: pick first uncovered, then greedily add points.
            
            # Pick an uncovered t-subset to seed the block
            # Use a heuristic: pick the uncovered t-subset whose points have 
            # the fewest available "partners" - or just pick one at random for diversity
            if seed_order is not None:
                # Try to find an uncovered t-subset in the given order
                seed_idx = None
                for i in seed_order:
                    if i in uncovered:
                        seed_idx = i
                        break
                if seed_idx is None:
                    seed_idx = min(uncovered)
            else:
                seed_idx = min(uncovered)
            
            seed_tsub = all_tsubs[seed_idx]
            block = list(seed_tsub)
            block_set = set(block)

            # Greedily add points to maximize new coverage
            while len(block) < k:
                if time.time() > deadline:
                    break
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    # A t-subset is covered if it's a subset of block ∪ {x}
                    # i.e., it contains x and (t-1) elements from block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in tsub_to_idx:
                            idx = tsub_to_idx[ts]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            block_tuple = tuple(block)
            if block_tuple in seen_blocks:
                continue
            seen_blocks.add(block_tuple)
            blocks.append(block)

            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                cover_count[idx] += 1
                uncovered.discard(idx)

        return blocks, cover_count

    # Local search: remove redundant blocks
    def local_search(blocks, cover_count):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        n_tsubs = len(all_tsubs)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_blocks = []
            new_cover_count = list(cover_count)
            for bi, block in enumerate(blocks):
                # Check if this block is redundant
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx[ts]
                    if new_cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for ts in itertools.combinations(block, t):
                        idx = tsub_to_idx[ts]
                        new_cover_count[idx] -= 1
                    improved = True
                else:
                    new_blocks.append(block)
            if improved:
                blocks = new_blocks
                cover_count = new_cover_count
        return blocks, cover_count

    # Try multiple strategies
    best_blocks = None
    best_count = float('inf')

    # Strategy 1: Standard greedy
    blocks, cc = greedy_construct()
    blocks, cc = local_search(blocks, cc)
    if len(blocks) < best_count:
        best_blocks = blocks
        best_count = len(blocks)

    # Strategy 2: Greedy with random seed ordering (a few restarts)
    random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        order = list(range(n_tsubs))
        random.shuffle(order)
        blocks, cc = greedy_construct(seed_order=order)
        blocks, cc = local_search(blocks, cc)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)

    # Strategy 3: Greedy but pick seed t-subset by "rarity" heuristic
    # Points that appear in fewer t-subsets are more constrained
    if time.time() < deadline - 10:
        # Compute point frequencies in all t-subsets (all equal for uniform case, 
        # so use a different heuristic: pick uncovered t-subset with max number 
        # of points that are in fewest blocks so far)
        # For simplicity, try a few more random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            order = list(range(n_tsubs))
            random.shuffle(order)
            blocks, cc = greedy_construct(seed_order=order)
            blocks, cc = local_search(blocks, cc)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)

    # Final safety: if we ran out of time and best_blocks is empty, do a minimal valid covering
    if best_blocks is None or not best_blocks:
        # Fallback: simple greedy without time check
        uncovered = set(range(n_tsubs))
        blocks = []
        seen_blocks = set()
        while uncovered:
            seed_idx = min(uncovered)
            seed_tsub = all_tsubs[seed_idx]
            block = list(seed_tsub)
            block_set = set(block)
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in tsub_to_idx:
                            idx = tsub_to_idx[ts]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
                block_set.add(best_point)
            block.sort()
            bt = tuple(block)
            if bt in seen_blocks:
                continue
            seen_blocks.add(bt)
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(tsub_to_idx[ts])
        best_blocks = blocks

    # Verify coverage and return
    return best_blocks
