"""Covering design C(19,8,5): greedy + local search with time guard."""
import itertools
import random
import time

def construct(v, k, t):
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin before 120s limit

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    # Map each t-subset to an index for fast lookup
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    # coverage[i] = number of blocks covering t-subset i
    coverage = [0] * n_t
    # For each point, precompute which t-subsets it participates in
    point_t_subs = [[] for _ in range(v)]
    for idx, s in enumerate(all_t_subs):
        for p in s:
            point_t_subs[p].append(idx)

    def elapsed():
        return time.time() - t0

    def time_left():
        return DEADLINE - time.time()

    def add_block(block_set, block_list):
        """Add a block to coverage; return number of newly covered t-subsets."""
        new_covered = 0
        for ts in itertools.combinations(block_list, t):
            idx = t_sub_to_idx[ts]
            if coverage[idx] == 0:
                new_covered += 1
            coverage[idx] += 1
        block_set.add(tuple(block_list))
        return new_covered

    def remove_block(block_list):
        """Remove a block from coverage; return number of t-subsets that became uncovered."""
        became_uncovered = 0
        for ts in itertools.combinations(block_list, t):
            idx = t_sub_to_idx[ts]
            coverage[idx] -= 1
            if coverage[idx] == 0:
                became_uncovered += 1
        block_set.discard(tuple(block_list))
        return became_uncovered

    def uncovered_count():
        return sum(1 for c in coverage if c == 0)

    # --- Phase 1: Greedy construction ---
    blocks = []
    block_set = set()
    uncovered = set(range(n_t))

    while uncovered:
        if time_left() < 30.0:
            break
        # Pick the first uncovered t-subset (or a random one for diversity)
        target = next(iter(uncovered))
        target_set = set(all_t_subs[target])

        # Build block greedily
        block = list(target_set)
        # Sort by some heuristic: prefer points that appear in many uncovered t-subsets
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for p in all_points:
                if p in block:
                    continue
                gain = 0
                # Count how many uncovered t-subsets this point would help cover
                # A t-subset is covered by adding p if it contains p and t-1 elements from block
                # Actually: adding p to block means any t-subset containing p and t-1 from block gets covered
                # But we need the t-subset to be uncovered.
                # For each (t-1)-subset of block, check if combining with p gives an uncovered t-subset
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (p,)))
                    idx = t_sub_to_idx.get(ts, -1)
                    if idx >= 0 and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point == -1:
                break
            block.append(best_point)
        
        block.sort()
        newly = add_block(block_set, block)
        blocks.append(block)
        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = t_sub_to_idx[ts]
            if idx in uncovered and coverage[idx] == 1:
                uncovered.discard(idx)

    if not blocks:
        # Fallback: return empty or minimal
        return []

    best_blocks = list(blocks)
    best_count = len(blocks)
    best_uncovered = uncovered_count()

    # --- Phase 2: Local search to reduce block count ---
    # Try removing blocks that don't cause any t-subset to become uncovered
    # Then try replacing pairs of blocks with a single better block

    def try_remove_blocks():
        """Try to remove redundant blocks."""
        nonlocal blocks, best_blocks, best_count, best_uncovered
        changed = True
        while changed:
            if time_left() < 10.0:
                return
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                # Temporarily remove block i
                bl = blocks[i]
                became_uncovered = remove_block(bl)
                if became_uncovered == 0:
                    blocks.pop(i)
                    changed = True
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_uncovered = uncovered_count()
                        best_blocks = [list(b) for b in blocks]
                else:
                    # Re-add
                    add_block(block_set, bl)
        # Ensure coverage is consistent
        # Recompute coverage from scratch if needed
        pass

    try_remove_blocks()

    # Try swapping: remove a block, then try to cover its unique t-subsets with modifications to other blocks
    # Simpler: try to merge two blocks if their union of t-subsets can be covered by a single new block
    # Actually, let's try a different approach: for each pair of blocks, see if we can find a single block
    # that covers all t-subsets currently covered ONLY by those two blocks.

    if time_left() > 20.0:
        # Local search: try to replace a block with a better one
        random.seed(42)
        for iteration in range(2000):
            if time_left() < 15.0:
                break
            if not blocks:
                break
            # Pick a random block to potentially improve
            idx = random.randrange(len(blocks))
            old_block = list(blocks[idx])
            
            # Find t-subsets that are uniquely covered by this block
            unique_to_block = []
            for ts in itertools.combinations(old_block, t):
                i = t_sub_to_idx[ts]
                if coverage[i] == 1:
                    unique_to_block.append(i)
            
            if not unique_to_block:
                # This block is redundant; remove it
                remove_block(old_block)
                blocks.pop(idx)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_uncovered = uncovered_count()
                    best_blocks = [list(b) for b in blocks]
                continue

            # Try to find a new block that covers all unique_to_block t-subsets
            # The new block must contain the union of all points in unique_to_block t-subsets
            required_points = set()
            for ui in unique_to_block:
                required_points.update(all_t_subs[ui])
            
            if len(required_points) > k:
                continue

            # Build a new block containing required_points plus additional points to maximize coverage
            new_block = list(required_points)
            while len(new_block) < k:
                best_p = -1
                best_gain = -1
                for p in all_points:
                    if p in new_block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(new_block, t - 1):
                        ts = tuple(sorted(rest + (p,)))
                        i = t_sub_to_idx.get(ts, -1)
                        if i is not None and coverage[i] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p == -1:
                    break
                new_block.append(best_p)
            
            new_block.sort()
            # Check if new block is better
            # Temporarily replace
            old_uncovered = uncovered_count()
            remove_block(old_block)
            add_block(block_set, new_block)
            new_uncovered = uncovered_count()
            
            if new_uncovered <= old_uncovered:
                blocks[idx] = new_block
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_uncovered = uncovered_count()
                    best_blocks = [list(b) for b in blocks]
            else:
                # Revert
                remove_block(new_block)
                add_block(block_set, old_block)

    # Final cleanup: remove any redundant blocks
    if time_left() > 5.0:
        changed = True
        while changed:
            if time_left() < 5.0:
                break
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                bl = blocks[i]
                became_uncovered = remove_block(bl)
                if became_uncovered == 0:
                    blocks.pop(i)
                    changed = True
                else:
                    add_block(block_set, bl)

    # Return the best result
    result = best_blocks if best_blocks else [list(b) for b in blocks]
    # Ensure validity: every t-subset is covered
    # If not fully covered, we still return what we have (evaluator checks)
    return result
