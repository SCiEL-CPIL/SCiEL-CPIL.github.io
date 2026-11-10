"""Optimized covering design C(19,7,4) solver.

Strategy:
1. Precompute all t-subsets and per-point t-1 subset lists for fast gain computation.
2. Fast greedy with vectorized gain scoring.
3. Aggressive iterative removal of redundant blocks.
4. Multiple randomized restarts to escape local optima.
5. Local search: swap points within blocks to maximize coverage density.
6. Deadline guard with safety margin.
"""

import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin for safety

    # Precompute all t-subsets
    all_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}

    # Precompute: for each point x and each (t-1)-tuple of other points,
    # which t-subset index does (x + rest) map to?
    # point_rest_to_idx[(x, rest_tuple)] = idx
    point_rest_to_idx = {}
    for idx, sub in enumerate(all_subs):
        for pos in range(t):
            x = sub[pos]
            rest = sub[:pos] + sub[pos+1:]
            point_rest_to_idx[(x, rest)] = idx

    # Precompute all (t-1)-subsets for each point
    # point_t1_subsets[x] = list of (t-1)-tuples not containing x
    point_t1_subsets = {}
    for x in range(v):
        others = [p for p in range(v) if p != x]
        point_t1_subsets[x] = list(itertools.combinations(others, t - 1))

    def greedy_construct(rng, deadline):
        """Fast greedy covering."""
        uncovered = [1] * n_subs  # 1 = uncovered
        blocks = []

        while True:
            # Check if all covered
            if all(c == 0 for c in uncovered):
                break
            if time.time() > deadline:
                break

            # Find first uncovered subset
            min_idx = -1
            for i in range(n_subs):
                if uncovered[i]:
                    min_idx = i
                    break
            if min_idx == -1:
                break

            min_sub = all_subs[min_idx]
            block = list(min_sub)
            block_set = set(block)

            while len(block) < k:
                best_x = -1
                best_gain = -1

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        idx = point_rest_to_idx.get((x, rest), -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx.get(sub, -1)
                if idx >= 0:
                    uncovered[idx] = 0

        return blocks

    def remove_redundant(blocks, deadline):
        """Iteratively remove blocks that don't cover any unique t-subset."""
        if not blocks:
            return blocks

        while True:
            if time.time() > deadline:
                break

            # Count coverage
            sub_count = defaultdict(int)
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    sub_count[sub] += 1

            # Find and remove redundant blocks
            changed = False
            new_blocks = []
            for block in blocks:
                redundant = True
                for sub in itertools.combinations(block, t):
                    if sub_count[sub] == 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                else:
                    new_blocks.append(block)

            if not changed:
                break
            blocks = new_blocks

        return blocks

    def local_search(blocks, deadline):
        """Local search: try swapping points in blocks to enable more removals."""
        if not blocks:
            return blocks

        best_blocks = blocks[:]
        best_len = len(blocks)

        # Compute sub_count
        def compute_sub_count(blks):
            sc = defaultdict(int)
            for b in blks:
                for sub in itertools.combinations(b, t):
                    sc[sub] += 1
            return sc

        sub_count = compute_sub_count(blocks)

        rng = random.Random()
        max_iter = 50000
        iter_count = 0

        while iter_count < max_iter and time.time() < deadline:
            iter_count += 1

            # Try removing a random block
            idx = rng.randint(0, len(blocks) - 1)
            block = blocks[idx]
            can_remove = True
            for sub in itertools.combinations(block, t):
                if sub_count[sub] <= 1:
                    can_remove = False
                    break

            if can_remove:
                for sub in itertools.combinations(block, t):
                    sub_count[sub] -= 1
                blocks.pop(idx)
                if len(blocks) < best_len:
                    best_blocks = blocks[:]
                    best_len = len(blocks)
                continue

            # Try swapping a point in a block
            if len(blocks) > 1:
                idx = rng.randint(0, len(blocks) - 1)
                block = blocks[idx]
                pos = rng.randint(0, k - 1)
                old_pt = block[pos]

                # Find a replacement point
                best_new_pt = -1
                best_score = -1

                for new_pt in range(v):
                    if new_pt in block:
                        continue

                    # Simulate swap
                    new_block = block[:]
                    new_block[pos] = new_pt
                    new_block.sort()

                    # Compute change in coverage
                    old_subs = set(itertools.combinations(block, t))
                    new_subs = set(itertools.combinations(new_block, t))

                    # Subsets lost
                    lost = old_subs - new_subs
                    # Subsets gained
                    gained = new_subs - old_subs

                    # Check if any lost subset has count <= 1
                    valid = True
                    for sub in lost:
                        if sub_count[sub] <= 1:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Score: number of newly covered subsets (count was 0)
                    score = 0
                    for sub in gained:
                        if sub_count[sub] == 0:
                            score += 1

                    if score > best_score:
                        best_score = score
                        best_new_pt = new_pt

                if best_new_pt != -1:
                    # Apply swap
                    old_subs = set(itertools.combinations(block, t))
                    new_block = block[:]
                    new_block[pos] = best_new_pt
                    new_block.sort()
                    new_subs = set(itertools.combinations(new_block, t))

                    for sub in old_subs:
                        sub_count[sub] -= 1
                    for sub in new_subs:
                        sub_count[sub] += 1

                    blocks[idx] = new_block

                    # Try to remove redundant blocks
                    changed = True
                    while changed:
                        changed = False
                        for j in range(len(blocks)):
                            b = blocks[j]
                            can_rm = True
                            for sub in itertools.combinations(b, t):
                                if sub_count[sub] <= 1:
                                    can_rm = False
                                    break
                            if can_rm:
                                for sub in itertools.combinations(b, t):
                                    sub_count[sub] -= 1
                                blocks.pop(j)
                                changed = True
                                if len(blocks) < best_len:
                                    best_blocks = blocks[:]
                                    best_len = len(blocks)
                                break

        return best_blocks

    def verify_and_fix(blocks, deadline):
        """Verify all t-subsets are covered; add blocks if needed."""
        covered = [0] * n_subs
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = sub_to_idx.get(sub, -1)
                if idx >= 0:
                    covered[idx] += 1

        missing = [i for i in range(n_subs) if covered[i] == 0]
        if not missing:
            return blocks

        # Add blocks to cover missing subsets
        blocks = blocks[:]
        for i in missing:
            if time.time() > deadline:
                break
            sub = all_subs[i]
            remaining = [p for p in range(v) if p not in sub]
            extra = random.sample(remaining, k - t)
            block = sorted(list(sub) + extra)
            if block not in blocks:
                blocks.append(block)

        return blocks

    # Main: try multiple restarts
    best_overall = None
    best_len_overall = float('inf')

    # Number of restarts based on available time
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed

    n_restarts = max(1, min(5, int(remaining / 25)))  # ~25s per restart

    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(42 + restart * 1000)

        # Greedy
        blocks = greedy_construct(rng, deadline)
        if time.time() > deadline:
            break

        # Remove redundant
        blocks = remove_redundant(blocks, deadline)
        if time.time() > deadline:
            break

        # Local search
        blocks = local_search(blocks, deadline)
        if time.time() > deadline:
            break

        # Verify
        blocks = verify_and_fix(blocks, deadline)

        if len(blocks) < best_len_overall:
            best_len_overall = len(blocks)
            best_overall = blocks[:]

        # If we're doing well, we can stop early
        if best_len_overall <= 140:
            break

    if best_overall is None:
        # Fallback: single greedy
        rng = random.Random(42)
        best_overall = greedy_construct(rng, deadline)
        best_overall = remove_redundant(best_overall, deadline)
        best_overall = verify_and_fix(best_overall, deadline)

    return best_overall
