"""Fresh greedy covering with incremental uncovered-count bookkeeping."""
import time
import itertools


def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin under the 120s kill
    pts = range(v)
    all_tsubs = list(itertools.combinations(pts, t))
    # incidence: point -> set of t-subsets containing it
    inc = {p: set() for p in pts}
    for c in all_tsubs:
        for p in c:
            inc[p].add(c)
    # uncovered t-subset -> set of points that can still cover it
    uncc = {c: set(c) for c in all_tsubs}
    # cnt[p] = number of uncovered t-subsets containing p
    cnt = {p: len(inc[p]) for p in pts}

    blocks = []
    while True:
        if time.time() > deadline:
            break
        if not uncc:
            break

        # 1) open a block on the first uncovered t-subset (cheapest uncovered)
        target = next(iter(uncc))
        block = list(target)
        # 2) grow greedily: add the point covering the most uncovered t-subsets
        while len(block) < k:
            best = None
            best_gain = -1
            for x in pts:
                if x in block:
                    continue
                g = cnt[x]
                if g > best_gain:
                    best_gain = g
                    best = x
            if best is None or best_gain <= 0:
                # cannot improve; fill arbitrarily to satisfy k distinct points
                for x in pts:
                    if x not in block:
                        block.append(x)
                    if len(block) == k:
                        break
                break
            block.append(best)

        block.sort()
        blocks.append(block)

        # 3) update bookkeeping for every t-subset of the new block
        for c in itertools.combinations(block, t):
            if c not in uncc:
                continue
            uncc.pop(c)
            for p in c:
                cnt[p] -= 1
            for x in block:
                if x not in c:
                    uncc[c].discard(x)

    return blocks
