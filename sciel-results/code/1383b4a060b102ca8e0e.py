import time
import random
import itertools
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 143.0  # safety margin before 150s kill
    best_blocks = None
    best_len = float('inf')

    def time_left():
        return deadline - time.time() > 3.0

    # Precompute t-subset tuples once (membership is O(1) on a set)
    all_tsubs = set(itertools.combinations(range(v), t))

    def greedy_cover(seed, strategy):
        rng = random.Random(seed)
        uncovered = set(all_tsubs)
        blocks = []

        # point_gain[p] = number of uncovered t-subsets containing p
        point_gain = [0] * v
        for sub in uncovered:
            for p in sub:
                point_gain[p] += 1

        while uncovered and time_left():
            # Choose starting t-subset based on strategy
            if strategy == 0:
                start = rng.choice(tuple(uncovered))
            elif strategy == 1:
                sample = list(uncovered)[:300]
                best_sub = None
                best_rarity = float('inf')
                for sub in sample:
                    rarity = min(point_gain[p] for p in sub)
                    if rarity < best_rarity:
                        best_rarity = rarity
                        best_sub = sub
                start = best_sub if best_sub else rng.choice(tuple(uncovered))
            else:
                start = min(uncovered)

            block = list(start)
            block_set = set(block)

            while len(block) < k and time_left():
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = point_gain[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and gain > 0 and rng.random() < 0.3:
                        best_point = x

                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = rng.choice(candidates)
                    best_gain = point_gain[best_point]

                block.append(best_point)
                block_set.add(best_point)

                # Update point_gain: any uncovered t-subset containing best_point
                # will be covered, so decrement the count for its other points.
                for rest in itertools.combinations(block, t - 1):
                    cand = tuple(sorted(rest + (best_point,)))
                    if cand in uncovered:
                        for p in rest:
                            point_gain[p] -= 1
                point_gain[best_point] = 0

            block.sort()
            blocks.append(block)

            # Remove all t-subsets covered by this block
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    uncovered.discard(sub)
                    for p in sub:
                        point_gain[p] -= 1

        return blocks, uncovered

    def complete_cover(blocks, uncovered):
        """Finish the covering even if the main loop hit the deadline."""
        if not uncovered:
            return blocks
        # Rebuild point_gain from current uncovered set
        point_gain = [0] * v
        for sub in uncovered:
            for p in sub:
                point_gain[p] += 1

        while uncovered:
            anchor = min(uncovered)
            block = list(anchor)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = point_gain[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = candidates[0]
                    best_gain = point_gain[best_point]
                block.append(best_point)
                block_set.add(best_point)
                for rest in itertools.combinations(block, t - 1):
                    cand = tuple(sorted(rest + (best_point,)))
                    if cand in uncovered:
                        for p in rest:
                            point_gain[p] -= 1
                point_gain[best_point] = 0
            block.sort()
            if block not in blocks:
                blocks.append(block)
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    uncovered.discard(sub)
                    for p in sub:
                        point_gain[p] -= 1
        return blocks

    def remove_redundant(blocks):
        """Drop blocks that are not the unique cover of any t-subset."""
        if len(blocks) <= 1:
            return blocks
        tsub_to_blocks = defaultdict(set)
        for i, block in enumerate(blocks):
            for comb in itertools.combinations(block, t):
                tsub_to_blocks[comb].add(i)
        removable = set(range(len(blocks)))
        for comb, bset in tsub_to_blocks.items():
            if len(bset) == 1:
                removable.discard(bset.pop())
        if not removable:
            return blocks
        return [b for i, b in enumerate(blocks) if i not in removable]

    strategies = [0, 1, 2]
    seed_base = 42

    for strategy in strategies:
        for trial in range(200):
            if not time_left():
                break
            seed = seed_base + strategy * 1000 + trial
            blocks, remaining = greedy_cover(seed, strategy)
            if not blocks:
                continue
            if remaining:
                blocks = complete_cover(blocks, remaining)
            if time_left():
                blocks = remove_redundant(blocks)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
            if not time_left():
                break

    if best_blocks is None:
        # Deterministic fallback
        uncovered = set(all_tsubs)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    candidates = [x for x in range(v) if x not in block_set]
                    best_point = candidates[0]
                block.append(best_point)
                block_set.add(best_point)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks

    return best_blocks
