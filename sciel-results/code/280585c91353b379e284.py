"""Rewritten covering design solver for C(v,k,t)."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """Construct a covering of t-subsets by k-subsets of {0,...,v-1}.

    Uses a vectorized greedy approach with randomized restarts and a
    deadline guard. Returns a list of blocks (each a list of k distinct ints).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Precompute all t-subsets as bitmasks for fast membership
    all_subsets = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_subsets.append(mask)

    num_subsets = len(all_subsets)

    # Build a mapping from mask to index for fast lookup
    mask_to_idx = {m: i for i, m in enumerate(all_subsets)}

    # For each point, precompute which t-subsets contain it
    point_subsets = [[] for _ in range(v)]
    for i, mask in enumerate(all_subsets):
        for p in range(v):
            if mask & (1 << p):
                point_subsets[p].append(i)

    # For each pair of points, precompute which t-subsets contain both
    # (used for gain computation when block has t-1 points)
    # This is expensive for large v; we'll compute on the fly instead

    def greedy_cover(uncovered_set, rng):
        """Greedy: repeatedly pick the k-subset that covers the most uncovered t-subsets."""
        blocks = []
        uncovered = set(uncovered_set)  # set of indices into all_subsets

        while uncovered and time.time() < deadline:
            # Find the best block: try all C(v,k) is too expensive.
            # Instead, use a heuristic: start from an uncovered subset and expand.
            # Pick a random uncovered subset as seed
            seed_idx = rng.choice(list(uncovered))
            seed_mask = all_subsets[seed_idx]
            seed_points = [p for p in range(v) if seed_mask & (1 << p)]

            block = list(seed_points)

            # Greedily add points to maximize coverage of uncovered subsets
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Compute gain: number of uncovered t-subsets that contain x
                    # and also contain t-1 points from the current block
                    gain = 0
                    block_set = set(block)
                    for idx in uncovered:
                        m = all_subsets[idx]
                        # Check if m contains x
                        if not (m & (1 << x)):
                            continue
                        # Count how many of block's points are in m
                        count = 0
                        for b in block:
                            if m & (1 << b):
                                count += 1
                                if count >= t - 1:
                                    break
                        if count >= t - 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and gain > 0:
                        # Tie-breaking: prefer points that appear in more uncovered subsets overall
                        pass
                if best_point == -1 or best_gain <= 0:
                    # Can't improve; pick a random unused point
                    remaining = [x for x in range(v) if x not in block]
                    if not remaining:
                        break
                    best_point = rng.choice(remaining)
                block.append(best_point)

            block.sort()
            block_set = frozenset(block)

            # Remove covered subsets
            for idx in list(uncovered):
                m = all_subsets[idx]
                # Check if all t-subset points are in block
                contained = True
                for p in range(v):
                    if m & (1 << p) and p not in block_set:
                        contained = False
                        break
                if contained:
                    uncovered.discard(idx)

            blocks.append(block)

        return blocks, uncovered

    def greedy_cover_fast(uncovered_set, rng):
        """Faster greedy using bit operations for containment checks."""
        blocks = []
        uncovered = set(uncovered_set)

        # Precompute for each subset index the set of points
        subset_points = []
        for mask in all_subsets:
            pts = [p for p in range(v) if mask & (1 << p)]
            subset_points.append(pts)

        while uncovered and time.time() < deadline:
            # Pick seed from uncovered
            seed_idx = rng.choice(list(uncovered))
            block = list(subset_points[seed_idx])
            block_set = set(block)

            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count uncovered subsets that contain x and t-1 points from block
                    gain = 0
                    for idx in uncovered:
                        pts = subset_points[idx]
                        if x not in pts:
                            continue
                        # Count intersection with block
                        inter = 0
                        for b in block:
                            if b in pts:
                                inter += 1
                                if inter >= t - 1:
                                    break
                        if inter >= t - 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining:
                        break
                    best_point = rng.choice(remaining)
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            block_set = frozenset(block)

            # Remove covered
            for idx in list(uncovered):
                pts = subset_points[idx]
                if all(p in block_set for p in pts):
                    uncovered.discard(idx)

            blocks.append(block)

        return blocks, uncovered

    def greedy_cover_bitmask(uncovered_set, rng):
        """Greedy using bitmask operations for speed."""
        blocks = []
        uncovered = set(uncovered_set)

        # Precompute subset point lists
        subset_points = []
        for mask in all_subsets:
            pts = []
            for p in range(v):
                if mask & (1 << p):
                    pts.append(p)
            subset_points.append(pts)

        # For faster gain computation, precompute for each point the list of subset indices containing it
        point_to_subsets = [[] for _ in range(v)]
        for i, pts in enumerate(subset_points):
            for p in pts:
                point_to_subsets[p].append(i)

        while uncovered and time.time() < deadline:
            seed_idx = rng.choice(list(uncovered))
            block = list(subset_points[seed_idx])
            block_set = set(block)

            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # For each subset containing x, check if it has t-1 points in block
                    gain = 0
                    for idx in point_to_subsets[x]:
                        if idx not in uncovered:
                            continue
                        pts = subset_points[idx]
                        inter = 0
                        for b in block:
                            if b in pts:
                                inter += 1
                                if inter >= t - 1:
                                    break
                        if inter >= t - 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining:
                        break
                    best_point = rng.choice(remaining)
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            block_set = frozenset(block)

            for idx in list(uncovered):
                pts = subset_points[idx]
                if all(p in block_set for p in pts):
                    uncovered.discard(idx)

            blocks.append(block)

        return blocks, uncovered

    # Try multiple strategies with random restarts
    best_blocks = None
    best_count = float('inf')

    rng = random.Random(42)

    # Strategy 1: Deterministic greedy (seed = smallest uncovered)
    try:
        uncovered_init = set(range(num_subsets))
        # Deterministic: always pick the smallest uncovered index as seed
        blocks_det = []
        uncovered = set(uncovered_init)
        subset_points = []
        for mask in all_subsets:
            pts = [p for p in range(v) if mask & (1 << p)]
            subset_points.append(pts)

        point_to_subsets = [[] for _ in range(v)]
        for i, pts in enumerate(subset_points):
            for p in pts:
                point_to_subsets[p].append(i)

        while uncovered and time.time() < deadline:
            seed_idx = min(uncovered)
            block = list(subset_points[seed_idx])
            block_set = set(block)

            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for idx in point_to_subsets[x]:
                        if idx not in uncovered:
                            continue
                        pts = subset_points[idx]
                        inter = 0
                        for b in block:
                            if b in pts:
                                inter += 1
                                if inter >= t - 1:
                                    break
                        if inter >= t - 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining:
                        break
                    best_point = remaining[0]
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            block_set = frozenset(block)

            for idx in list(uncovered):
                pts = subset_points[idx]
                if all(p in block_set for p in pts):
                    uncovered.discard(idx)

            blocks_det.append(block)

        if not uncovered and len(blocks_det) < best_count:
            best_blocks = blocks_det
            best_count = len(blocks_det)
    except Exception:
        pass

    # Strategy 2: Randomized restarts
    for restart in range(20):
        if time.time() >= deadline:
            break
        rng = random.Random(1000 + restart)
        try:
            blocks, uncovered = greedy_cover_bitmask(set(range(num_subsets)), rng)
            if not uncovered and len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        except Exception:
            continue

    # Strategy 3: Local search - swap points in blocks to reduce count
    if best_blocks and time.time() < deadline:
        # Try to remove redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(best_blocks) - 1, -1, -1):
                if time.time() >= deadline:
                    break
                # Check if block i is redundant
                block_i = best_blocks[i]
                block_i_set = set(block_i)
                # Find a t-subset that is ONLY covered by block i
                only_covered = False
                for combo in itertools.combinations(block_i, t):
                    mask = 0
                    for x in combo:
                        mask |= (1 << x)
                    idx = mask_to_idx.get(mask)
                    if idx is None:
                        continue
                    # Check if any other block covers this t-subset
                    covered_elsewhere = False
                    for j, block_j in enumerate(best_blocks):
                        if j == i:
                            continue
                        block_j_set = set(block_j)
                        if all(x in block_j_set for x in combo):
                            covered_elsewhere = True
                            break
                    if not covered_elsewhere:
                        only_covered = True
                        break
                if not only_covered:
                    best_blocks.pop(i)
                    changed = True

    # Fallback: if no valid covering found, use a brute-force-ish approach
    if best_blocks is None:
        # Simple greedy without randomization
        blocks = []
        uncovered = set(range(num_subsets))
        subset_points = []
        for mask in all_subsets:
            pts = [p for p in range(v) if mask & (1 << p)]
            subset_points.append(pts)

        while uncovered and time.time() < deadline:
            seed_idx = min(uncovered)
            block = list(subset_points[seed_idx])
            block_set = set(block)

            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for idx in uncovered:
                        pts = subset_points[idx]
                        if x not in pts:
                            continue
                        inter = 0
                        for b in block:
                            if b in pts:
                                inter += 1
                                if inter >= t - 1:
                                    break
                        if inter >= t - 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining:
                        break
                    best_point = remaining[0]
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            block_set = frozenset(block)

            for idx in list(uncovered):
                pts = subset_points[idx]
                if all(p in block_set for p in pts):
                    uncovered.discard(idx)

            blocks.append(block)

        best_blocks = blocks

    return best_blocks
