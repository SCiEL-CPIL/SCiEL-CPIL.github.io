"""Covering design C(v,k,t) via fast greedy + local improvement with a strict deadline guard."""
import itertools
import time

# EVOLVE-BLOCK-START


def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))


def _block_covers(block, t):
    return list(itertools.combinations(block, t))


def _greedy_build(v, k, t, deadline, seed_order=None):
    """
    Fast greedy: maintain an array of uncovered counts per t-subset index.
    We map each t-subset to an integer index for O(1) lookup.
    """
    all_subs = _all_tsubsets(v, t)
    n_subs = len(all_subs)
    # Map tuple -> index
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    uncovered = [1] * n_subs  # 1 = uncovered, 0 = covered

    # Precompute for each point x and each (t-1)-subset of other points, which t-subsets are formed
    # But that's expensive. Instead, for a candidate point x given current partial block B,
    # the new t-subsets covered are those (b1,...,b_{t-1}, x) where {b1,...,b_{t-1}} is a (t-1)-subset of B.
    # We need to check if each such t-subset is still uncovered.

    blocks = []
    # For speed, precompute all (t-1)-subsets of each possible (t-1)-tuple? No, too many.
    # Instead, when building a block, for each candidate x, we iterate over all (t-1)-subsets of the current block.

    # To speed up the "is this t-subset uncovered?" check, we use the sub_to_idx dict.
    # But dict lookup in Python is slow. Let's try a different approach:
    # Represent uncovered as a set of tuples? That's also slow for large v.
    # For v=24, t=3, n_subs = C(24,3) = 2024. A list of 2024 ints is fine.
    # The bottleneck is the gain computation: for each candidate x (up to 24), for each (t-1)-subset of block (up to C(6,2)=15 if block size 6), check if tuple is uncovered.

    # Let's precompute a lookup: for each t-subset tuple, its index. We already have sub_to_idx.
    # To avoid tuple creation overhead, we can precompute for each point x and each pair of points (for t=3), the index.
    # But t is general. Let's keep it general but optimize the inner loop.

    # Alternative: use a boolean numpy array? NumPy is available.
    # Let's try with a set of uncovered tuples first for simplicity, then optimize if needed.

    # Actually, let's use a list of ints for uncovered, and sub_to_idx for mapping.
    # The gain function:
    def gain(x, block):
        g = 0
        for rest in itertools.combinations(block, t - 1):
            # Form the t-subset containing x and rest
            # We need to check if this t-subset is uncovered
            # Create the tuple and look up its index
            # To avoid creating a new tuple each time, we can sort rest + (x,)
            # But rest is already sorted (from combinations), and x may be out of order.
            # Let's just create the tuple and use sub_to_idx.
            full = tuple(sorted(rest + (x,)))
            idx = sub_to_idx[full]
            if uncovered[idx]:
                g += 1
        return g

    # To further speed up, precompute for each x and each possible (t-1)-tuple, the index of the t-subset.
    # But (t-1)-tuples vary. Let's just try the direct approach first and see if it's fast enough.

    # Greedy loop
    while any(uncovered):
        if time.time() > deadline:
            break
        # Find an uncovered t-subset to start a new block
        # Find the first uncovered one
        start_idx = -1
        for i in range(n_subs):
            if uncovered[i]:
                start_idx = i
                break
        if start_idx == -1:
            break

        start_sub = all_subs[start_idx]
        block = list(start_sub)

        while len(block) < k:
            if time.time() > deadline:
                break
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = gain(x, block)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        block.sort()
        blocks.append(block)
        # Mark all t-subsets in this block as covered
        for sub in itertools.combinations(block, t):
            uncovered[sub_to_idx[sub]] = 0

    return blocks


def _local_improve(blocks, v, k, t, deadline):
    """
    Try to remove redundant blocks and replace blocks to improve coverage.
    """
    all_subs = _all_tsubsets(v, t)
    n_subs = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}

    def coverage_count(blocks_list):
        counts = [0] * n_subs
        for block in blocks_list:
            for sub in itertools.combinations(block, t):
                counts[sub_to_idx[sub]] += 1
        return counts

    def is_valid(blocks_list):
        counts = coverage_count(blocks_list)
        return all(c >= 1 for c in counts)

    # First pass: remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Try removing block i
            trial = blocks[:i] + blocks[i+1:]
            if is_valid(trial):
                blocks = trial
                changed = True
                break  # Restart the loop since indices changed

    # Second pass: try replacing blocks
    # For each block, try to find a better replacement that covers more uncovered t-subsets
    # This is complex. Let's try a simpler approach: for each pair of blocks, try swapping.
    # Actually, let's try: for each block i, try removing it and then greedily adding a new block
    # that covers the most newly-uncovered t-subsets.

    for i in range(len(blocks)):
        if time.time() > deadline:
            break
        # Remove block i
        without_i = blocks[:i] + blocks[i+1:]
        counts = coverage_count(without_i)
        uncovered_subs = [all_subs[j] for j in range(n_subs) if counts[j] == 0]
        if not uncovered_subs:
            # Block i is redundant, remove it
            blocks = without_i
            continue

        # Try to find a new block that covers the most uncovered t-subsets
        best_block = None
        best_gain = 0
        # Candidate blocks: start from each uncovered t-subset
        for start_sub in uncovered_subs:
            if time.time() > deadline:
                break
            block = list(start_sub)
            while len(block) < k:
                if time.time() > deadline:
                    break
                best_x = -1
                best_g = -1
                for x in range(v):
                    if x in block:
                        continue
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        full = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx[full]
                        if counts[idx] == 0:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            block.sort()
            # Count how many uncovered t-subsets this block covers
            gain = 0
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx[sub]
                if counts[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is not None and best_gain > 0:
            blocks = without_i + [best_block]

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Initial greedy
    blocks = _greedy_build(v, k, t, deadline)

    # Local improvement
    if time.time() < deadline - 5.0:
        blocks = _local_improve(blocks, v, k, t, deadline)

    # Try a few more greedy runs with different starting points if time allows
    if time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            alt_blocks = _greedy_build(v, k, t, deadline - 5.0)
            if len(alt_blocks) < len(blocks):
                blocks = alt_blocks
                blocks = _local_improve(blocks, v, k, t, deadline)

    return blocks
# EVOLVE-BLOCK-END
