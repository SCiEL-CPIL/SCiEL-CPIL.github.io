# EVOLVE-BLOCK-START
import itertools
import time

_DEADLINE = 150.0


def construct(v, k, t):
    start = time.time()
    points = list(range(v))

    uncovered = set(frozenset(c) for c in itertools.combinations(points, t))
    freq = [0] * v
    blocks = []

    while uncovered:
        if time.time() - start > _DEADLINE - 15:
            break

        # Seed: lexicographically smallest uncovered t-subset
        seed_tsub = min(uncovered)
        seed_point = seed_tsub[0]

        block = [seed_point]
        freq[seed_point] += 1

        while len(block) < k:
            best_x = None
            best_score = -1
            for x in points:
                if x in block:
                    continue
                gain = 0
                if len(block) >= t - 1:
                    for s in itertools.combinations(block, t - 1):
                        if frozenset(s + (x,)) in uncovered:
                            gain += 1
                score = gain - 0.01 * freq[x]
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            freq[best_x] += 1

        block.sort()
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(frozenset(ts))

    return blocks
# EVOLVE-BLOCK-END
