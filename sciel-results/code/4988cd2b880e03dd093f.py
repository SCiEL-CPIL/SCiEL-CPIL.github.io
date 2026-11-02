"""Improved covering design search for C(19,8,5) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def elapsed():
        return time.time() - start_time

    # All t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # For each point, precompute which t-subsets contain it
    point_tsubsets = [[] for _ in range(v)]
    for ts in all_tsubsets:
        for p in ts:
            point_tsubsets[p].append(tsubset_index[ts])

    coverage = [0] * n_tsubsets
    blocks = []

    def update_coverage(block, delta):
        for ts in itertools.combinations(block, t):
            coverage[tsubset_index[ts]] += delta

    def is_complete():
        return all(c > 0 for c in coverage)

    def greedy_block(rng):
        """Build a block greedily maximizing uncovered coverage."""
        # Find an uncovered t-subset to start
        uncovered = [i for i in range(n_tsubsets) if coverage[i] == 0]
        if not uncovered:
            return None
        start_idx = rng.choice(uncovered)
        block = list(all_tsubsets[start_idx])
        block_set = set(block)

        while len(block) < k:
            best_point = None
            best_gain = -1
            for p in range(v):
                if p in block_set:
                    continue
                gain = 0
                for ts_idx in point_tsubsets[p]:
                    if coverage[ts_idx] == 0:
                        ts = all_tsubsets[ts_idx]
                        ts_set = set(ts)
                        if p in ts_set:
                            rest = ts_set - {p}
                            if rest.issubset(block_set):
                                gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point is None:
                break
            block.append(best_point)
            block_set.add(best_point)
        block.sort()
        return block

    # Phase 1: Greedy construction
    rng = random.Random(42)
    while not is_complete():
        if elapsed() > deadline:
            break
        block = greedy_block(rng)
        if block is None:
            break
        update_coverage(block, 1)
        blocks.append(block)

    best_blocks = list(blocks)
    best_count = len(blocks)

    # Phase 2: Try to improve by removing and re-adding blocks
    if elapsed() < deadline - 15:
        current = list(blocks)
        for _ in range(500):
            if elapsed() > deadline:
                break
            if not current:
                break
            idx = rng.randint(0, len(current) - 1)
            removed = current[idx]
            current.pop(idx)
            update_coverage(removed, -1)

            # Re-cover
            new_blocks = []
            while not is_complete():
                if elapsed() > deadline:
                    break
                block = greedy_block(rng)
                if block is None:
                    break
                update_coverage(block, 1)
                new_blocks.append(block)

            current.extend(new_blocks)
            if len(current) < best_count:
                best_count = len(current)
                best_blocks = list(current)

    # Verify and return
    if best_blocks:
        final_cov = [0] * n_tsubsets
        for b in best_blocks:
            for ts in itertools.combinations(b, t):
                final_cov[tsubset_index[ts]] += 1
        if all(c > 0 for c in final_cov):
            return best_blocks

    # Fallback
    final_cov = [0] * n_tsubsets
    for b in blocks:
        for ts in itertools.combinations(b, t):
            final_cov[tsubset_index[ts]] += 1
    if all(c > 0 for c in final_cov):
        return blocks

    return blocks
# EVOLVE-BLOCK-END
