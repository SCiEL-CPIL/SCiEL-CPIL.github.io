"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 135.0
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute: for each point, which t-subsets contain it
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_tsubs[p].append(idx)

    def compute_coverage(blocks):
        """Return coverage count array for a list of blocks."""
        coverage = [0] * n_tsubs
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage[idx] += 1
        return coverage

    def is_valid_cover(blocks):
        coverage = compute_coverage(blocks)
        return all(c > 0 for c in coverage)

    def greedy_set_cover_from_uncovered(uncovered_set, existing_blocks, max_blocks=200):
        """Greedy set cover starting from uncovered t-subsets."""
        coverage = [0] * n_tsubs
        for block in existing_blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage[idx] += 1

        new_blocks = []
        uncovered = set(uncovered_set)

        while uncovered and len(new_blocks) < max_blocks:
            if time.time() > deadline:
                break

            # Find block that covers most uncovered t-subsets
            best_block = None
            best_count = -1

            # Try random candidates
            for _ in range(50):
                candidate = set(rng.sample(range(v), k))
                cnt = 0
                for ts in itertools.combinations(candidate, t):
                    if ts in uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_block = candidate

            # Also try deterministic: pick an uncovered t-subset and extend it
            if uncovered:
                sample_ts = next(iter(uncovered))
                for _ in range(20):
                    candidate = set(sample_ts)
                    for _ in range(k - t):
                        best_p = None
                        best_gain = -1
                        for p in range(v):
                            if p in candidate:
                                continue
                            gain = 0
                            for ts2 in itertools.combinations(candidate | {p}, t):
                                if ts2 in uncovered:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_p = p
                        if best_p is not None:
                            candidate.add(best_p)
                    cnt = 0
                    for ts in itertools.combinations(candidate, t):
                        if ts in uncovered:
                            cnt += 1
                    if cnt > best_count:
                        best_count = cnt
                        best_block = candidate

            if best_block is None or best_count <= 0:
                break

            new_blocks.append(sorted(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts)

        return new_blocks

    def build_packing(v, k, max_packing_size=20):
        """Build a large packing (set of pairwise disjoint k-subsets) using branch and bound."""
        # For v=24, k=7, max packing size is floor(24/7) = 3
        max_size = v // k
        packing = []
        used_points = set()

        # Branch and bound to find max packing
        best_packing = []

        def branch(remaining_points, current_packing, depth):
            nonlocal best_packing
            if time.time() > deadline:
                return
            if len(current_packing) > len(best_packing):
                best_packing = list(current_packing)

            if len(current_packing) == max_size:
                return

            # Lower bound: even with perfect packing, can't exceed remaining/k
            if len(current_packing) + len(remaining_points) // k <= len(best_packing):
                return

            # Choose a point to branch on (first unused point)
            if not remaining_points:
                return

            pivot = min(remaining_points)

            # Option 1: don't include pivot in any more blocks
            new_remaining = set(remaining_points)
            branch(new_remaining, current_packing, depth + 1)

            # Option 2: include pivot in a new block
            remaining_after_pivot = remaining_points - {pivot}
            # Choose k-1 more points from remaining
            candidates = list(remaining_after_pivot)
            rng.shuffle(candidates)
            # Try a subset of combinations for speed
            num_trials = min(100, len(candidates))
            for _ in range(num_trials):
                if len(candidates) < k - 1:
                    break
                trial_idx = rng.sample(range(len(candidates)), k - 1)
                block_points = [pivot] + [candidates[i] for i in sorted(trial_idx)]
                block_set = set(block_points)
                new_remaining = remaining_points - block_set
                current_packing.append(sorted(block_points))
                branch(new_remaining, current_packing, depth + 1)
                current_packing.pop()

        all_points = set(range(v))
        branch(all_points, [], 0)

        return best_packing

    def packing_based_cover(v, k, t, packing):
        """Build a cover starting from a packing, then fill in gaps."""
        blocks = list(packing)
        coverage = compute_coverage(blocks)
        uncovered = [i for i in range(n_tsubs) if coverage[i] == 0]

        if not uncovered:
            return blocks

        # Greedily add blocks to cover remaining t-subsets
        uncovered_set = set(uncovered)
        added = []

        while uncovered_set:
            if time.time() > deadline:
                break

            best_block = None
            best_count = -1

            # Try blocks that include points from uncovered t-subsets
            for _ in range(80):
                # Pick a random uncovered t-subset as seed
                sample_idx = rng.choice(list(uncovered_set))
                seed = all_tsubs[sample_idx]
                candidate = set(seed)
                for _ in range(k - t):
                    best_p = None
                    best_gain = -1
                    for p in range(v):
                        if p in candidate:
                            continue
                        gain = 0
                        for ts in itertools.combinations(candidate | {p}, t):
                            if tsub_to_idx[ts] in uncovered_set:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                    if best_p is not None:
                        candidate.add(best_p)

                cnt = 0
                for ts in itertools.combinations(candidate, t):
                    idx = tsub_to_idx[ts]
                    if idx in uncovered_set:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_block = candidate

            if best_block is None or best_count <= 0:
                break

            added.append(sorted(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered_set.discard(tsub_to_idx[ts])

        return blocks + added

    def local_search_reduce(blocks, max_rounds=10):
        """Try to reduce block count via 1-exchange and removal."""
        if not blocks:
            return blocks

        coverage = compute_coverage(blocks)
        improved = True
        round_num = 0

        while improved and round_num < max_rounds:
            improved = False
            round_num += 1

            # Try removing redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks
                can_remove = True
                for ts in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[ts]
                    if coverage[idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in itertools.combinations(blocks[i], t):
                        idx = tsub_to_idx[ts]
                        coverage[idx] -= 1
                    blocks.pop(i)
                    improved = True

            if improved:
                continue

            # Try 1-exchange: replace block i with a better one
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks

                # Find t-subsets uniquely covered by block i
                unique_ts = set()
                for ts in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[ts]
                    if coverage[idx] == 1:
                        unique_ts.add(ts)

                if not unique_ts:
                    continue

                # Required points from unique t-subsets
                required = set()
                for ts in unique_ts:
                    required.update(ts)

                if len(required) > k:
                    continue

                # Try to find replacement block
                best_new = None
                best_score = -float('inf')

                for trial in range(30):
                    new_block_set = set(required)
                    for _ in range(k - len(new_block_set)):
                        best_p = None
                        best_gain = -1
                        for p in range(v):
                            if p in new_block_set:
                                continue
                            gain = 0
                            test_set = new_block_set | {p}
                            for ts in itertools.combinations(sorted(test_set), t):
                                idx = tsub_to_idx[ts]
                                if coverage[idx] > 1:
                                    gain += 2
                                elif coverage[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_p = p
                        if best_p is not None:
                            new_block_set.add(best_p)

                    # Verify all unique t-subsets are covered
                    valid = True
                    for ts in unique_ts:
                        if not set(ts).issubset(new_block_set):
                            valid = False
                            break
                    if not valid:
                        continue

                    # Score: how many other blocks become redundant
                    # Temporarily remove block i
                    temp_cov = coverage.copy()
                    for ts in itertools.combinations(blocks[i], t):
                        temp_cov[tsub_to_idx[ts]] -= 1
                    for ts in itertools.combinations(new_block_set, t):
                        temp_cov[tsub_to_idx[ts]] += 1

                    # Count blocks that can be removed
                    removable = 0
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        can_rm = True
                        for ts in itertools.combinations(blocks[j], t):
                            if temp_cov[tsub_to_idx[ts]] <= 1:
                                can_rm = False
                                break
                        if can_rm:
                            removable += 1

                    score = removable * 10 - 0.01 * trial
                    if score > best_score:
                        best_score = score
                        best_new = sorted(new_block_set)

                if best_new is not None and best_score > 0:
                    # Apply the exchange
                    for ts in itertools.combinations(blocks[i], t):
                        coverage[tsub_to_idx[ts]] -= 1
                    for ts in itertools.combinations(best_new, t):
                        coverage[tsub_to_idx[ts]] += 1
                    blocks[i] = best_new

                    # Remove any newly redundant blocks
                    for j in range(len(blocks) - 1, -1, -1):
                        can_rm = True
                        for ts in itertools.combinations(blocks[j], t):
                            if coverage[tsub_to_idx[ts]] <= 1:
                                can_rm = False
                                break
                        if can_rm:
                            for ts in itertools.combinations(blocks[j], t):
                                coverage[tsub_to_idx[ts]] -= 1
                            blocks.pop(j)

                    improved = True

        return blocks

    # Main strategy: try packing-based approach and greedy, keep best
    best_blocks = None
    best_count = float('inf')

    # Strategy 1: Pure greedy set cover with multiple restarts
    for restart in range(3):
        if time.time() > deadline - 20:
            break

        blocks = []
        uncovered = set(range(n_tsubs))
        coverage = [0] * n_tsubs

        while uncovered:
            if time.time() > deadline - 10:
                break

            best_block = None
            best_count_uncov = -1

            # Random candidates
            for _ in range(60):
                candidate = set(rng.sample(range(v), k))
                cnt = 0
                for ts in itertools.combinations(candidate, t):
                    if tsub_to_idx[ts] in uncovered:
                        cnt += 1
                if cnt > best_count_uncov:
                    best_count_uncov = cnt
                    best_block = candidate

            # Seeded from uncovered
            if uncovered:
                for _ in range(20):
                    sample_idx = rng.choice(list(uncovered))
                    seed = all_tsubs[sample_idx]
                    candidate = set(seed)
                    for _ in range(k - t):
                        best_p = None
                        best_gain = -1
                        for p in range(v):
                            if p in candidate:
                                continue
                            gain = 0
                            for ts in itertools.combinations(candidate | {p}, t):
                                if tsub_to_idx[ts] in uncovered:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_p = p
                        if best_p is not None:
                            candidate.add(best_p)
                    cnt = 0
                    for ts in itertools.combinations(candidate, t):
                        if tsub_to_idx[ts] in uncovered:
                            cnt += 1
                    if cnt > best_count_uncov:
                        best_count_uncov = cnt
                        best_block = candidate

            if best_block is None or best_count_uncov <= 0:
                break

            blocks.append(sorted(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(tsub_to_idx[ts])

        if uncovered:
            continue

        blocks = local_search_reduce(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()

    # Strategy 2: Packing-based approach
    if time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            packing = build_packing(v, k)
            if len(packing) >= 2:
                blocks = packing_based_cover(v, k, t, packing)
                if is_valid_cover(blocks):
                    blocks = local_search_reduce(blocks)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = blocks.copy()

    if best_blocks is not None:
        return best_blocks

    # Fallback: simple greedy
    blocks = []
    uncovered = set(range(n_tsubs))
    while uncovered:
        sample_idx = min(uncovered)
        seed = all_tsubs[sample_idx]
        candidate = list(seed)
        while len(candidate) < k:
            best_p = None
            best_gain = -1
            for p in range(v):
                if p in candidate:
                    continue
                gain = sum(1 for ts in itertools.combinations(candidate + [p], t)
                           if tsub_to_idx[ts] in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p is not None:
                candidate.append(best_p)
        blocks.append(sorted(candidate))
        for ts in itertools.combinations(candidate, t):
            uncovered.discard(tsub_to_idx[ts])
    return blocks
# EVOLVE-BLOCK-END
