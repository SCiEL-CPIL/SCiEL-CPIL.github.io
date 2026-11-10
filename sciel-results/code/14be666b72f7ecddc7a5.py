# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 120s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)

    # Index t-subsets for fast lookup
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # coverage[i] = number of blocks that cover all_t_subsets[i]
    coverage = np.zeros(total_t, dtype=np.int32)

    # uncovered[i] = True if all_t_subsets[i] is not yet covered
    uncovered = np.ones(total_t, dtype=bool)
    uncovered_count = total_t

    blocks = []

    # Precompute for efficient gain computation
    if t == 3:
        # triple_index[pair][x] = index of triple sorted((a,b,x))
        triple_index = {}
        for a in range(v):
            for b in range(a + 1, v):
                pair = (a, b)
                triple_index[pair] = {}
                for x in range(v):
                    if x != a and x != b:
                        triple = tuple(sorted((a, b, x)))
                        triple_index[pair][x] = t_subset_index[triple]
    else:
        triple_index = None

    def get_gain(block, x):
        """Compute how many uncovered t-subsets are covered by adding x to block."""
        if t == 3:
            gain = 0
            n = len(block)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = block[i], block[j]
                    if a > b:
                        a, b = b, a
                    idx = triple_index[(a, b)][x]
                    if uncovered[idx]:
                        gain += 1
            return gain
        else:
            gain = 0
            for rest in itertools.combinations(block, t - 1):
                ts = tuple(sorted(rest + (x,)))
                idx = t_subset_index.get(ts, -1)
                if idx >= 0 and uncovered[idx]:
                    gain += 1
            return gain

    def cover_block(block):
        """Update coverage and uncovered arrays for the given block."""
        nonlocal uncovered_count
        newly_covered = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                newly_covered += 1
            coverage[idx] += 1
        uncovered_count -= newly_covered
        return newly_covered

    def uncover_block(block):
        """Reverse the coverage for the given block (for local search swaps)."""
        nonlocal uncovered_count
        newly_uncovered = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            coverage[idx] -= 1
            if coverage[idx] == 0:
                if not uncovered[idx]:
                    uncovered[idx] = True
                    newly_uncovered += 1
        uncovered_count += newly_uncovered
        return newly_uncovered

    def local_search(block, max_iters=50):
        """Try to improve a completed block by swapping out low-value points."""
        nonlocal uncovered_count
        block_set = set(block)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            # For each point in block, compute its marginal contribution
            # (how many uncovered triples it uniquely covers given the rest of block)
            best_swap = None
            best_delta = 0
            for pos, p in enumerate(block):
                # Temporarily remove p
                remaining = [block[i] for i in range(len(block)) if i != pos]
                # Count uncovered triples involving p and two points from remaining
                p_gain = 0
                if t == 3:
                    for i in range(len(remaining)):
                        for j in range(i + 1, len(remaining)):
                            a, b = remaining[i], remaining[j]
                            if a > b:
                                a, b = b, a
                            idx = triple_index[(a, b)][p]
                            if uncovered[idx] and coverage[idx] == 1:
                                p_gain += 1
                # Try each candidate replacement
                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute gain of x with remaining
                    x_gain = 0
                    if t == 3:
                        for i in range(len(remaining)):
                            for j in range(i + 1, len(remaining)):
                                a, b = remaining[i], remaining[j]
                                if a > b:
                                    a, b = b, a
                                idx = triple_index[(a, b)][x]
                                if uncovered[idx]:
                                    x_gain += 1
                    delta = x_gain - p_gain
                    if delta > best_delta:
                        best_delta = delta
                        best_swap = (pos, p, x)
            if best_swap is not None:
                pos, old_p, new_p = best_swap
                # Perform swap: unblock old, block new
                # Remove old_p from block
                block[pos] = new_p
                block_set.remove(old_p)
                block_set.add(new_p)
                # Update coverage: the triples that change are those involving old_p or new_p
                # Simplest: uncover the whole block, then re-cover
                uncover_block(block)
                # Revert the swap in the array for uncover (need original block)
                block[pos] = old_p
                block_set.remove(new_p)
                block_set.add(old_p)
                uncover_block(block)  # wait, this is getting complicated
                # Let me redo: just uncover the full old block, then cover the new block
                # First restore block to old state
                block[pos] = old_p
                block_set.remove(new_p)
                block_set.add(old_p)
                uncover_block(block)
                # Now apply swap
                block[pos] = new_p
                block_set.remove(old_p)
                block_set.add(new_p)
                cover_block(block)
                improved = True
        return block

    # Greedy construction with smarter start selection
    while uncovered_count > 0:
        if time.time() > deadline:
            break

        # Find the "hardest" uncovered t-subset: one whose points are in the fewest
        # existing blocks (i.e., most isolated). This tends to be in sparse regions.
        best_start = None
        best_score = -1
        # Sample a subset of uncovered t-subsets for efficiency
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        # Check up to 50 random uncovered subsets or all if fewer
        sample_size = min(len(uncovered_indices), 50)
        if len(uncovered_indices) <= sample_size:
            sample = uncovered_indices
        else:
            sample = np.random.choice(uncovered_indices, size=sample_size, replace=False)

        for idx in sample:
            ts = all_t_subsets[idx]
            # Score: sum of coverage of each (t-1)-subset of ts
            # Lower coverage means more "isolated" -> better start
            score = 0
            if t == 3:
                for i in range(t):
                    for j in range(i + 1, t):
                        a, b = ts[i], ts[j]
                        if a > b:
                            a, b = b, a
                        # Count blocks containing pair (a,b)
                        # We don't have pair coverage directly; use point-level proxy
                        score += 0  # placeholder
            # Use a simpler heuristic: pick the uncovered triple with min sum of point degrees
            # Point degree = number of blocks containing that point
            # We'll track point_block_count
            pass

        # Simpler: just pick the first uncovered (fast) but try a few candidates
        # Actually let's use: pick uncovered triple where the sum of (k - len(block) contributions)
        # is maximized. For now, use the first uncovered for speed, but check a small sample.
        # Let's use a weighted approach: prefer uncovered triples in under-covered regions.
        
        # Track which points appear in how many blocks
        # We'll compute this on the fly from blocks (expensive) or maintain incrementally.
        # For now, use the first uncovered as start (proven to work) but add randomization.
        
        # Pick from the sample with some randomness
        start_idx = sample[0]
        if len(sample) > 1 and random.random() < 0.3:
            start_idx = sample[random.randint(1, len(sample) - 1)]
        
        start_ts = all_t_subsets[start_idx]
        block = list(start_ts)

        # Greedily add points to maximize gain
        while len(block) < k:
            if time.time() > deadline:
                break
            # Find best point to add
            best_x = None
            best_gain = -1
            for x in range(v):
                if x not in block:
                    g = get_gain(block, x)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                    elif g == best_gain and best_x is not None:
                        # Tie-breaking: prefer points that appear in fewer existing blocks
                        pass

            if best_x is None:
                break

            # If no point has positive gain, pick one that minimizes redundancy
            if best_gain <= 0:
                available = [x for x in range(v) if x not in block]
                if available:
                    # Pick the point that appears in the fewest existing blocks
                    # Count appearances
                    min_count = float('inf')
                    best_x = available[0]
                    for x in available:
                        cnt = sum(1 for b in blocks if x in b)
                        if cnt < min_count:
                            min_count = cnt
                            best_x = x
                    if min_count >= len(blocks) and len(blocks) > 0:
                        best_x = random.choice(available)

            block.append(best_x)

        if len(block) == k:
            block.sort()
            # Check for duplicate
            if block not in blocks:
                blocks.append(block)
                cover_block(block)
            else:
                # Duplicate found; try local search to differentiate
                pass
        else:
            pass

    # Local search phase: try to improve the covering by swapping points in blocks
    if time.time() < deadline - 5:
        for bi in range(len(blocks)):
            if time.time() > deadline - 3:
                break
            # Try local search on this block
            old_block = list(blocks[bi])
            new_block = local_search(blocks[bi], max_iters=30)
            if new_block != old_block:
                # Apply the change
                uncover_block(old_block)
                blocks[bi] = new_block
                cover_block(new_block)

    return blocks
# EVOLVE-BLOCK-END
