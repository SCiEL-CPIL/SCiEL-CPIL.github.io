"""Covering design C(v,k,t) - fast greedy + polish."""
import itertools
import time


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 150s kill
    points = list(range(v))

    # All t-subsets as frozensets
    all_t = [frozenset(c) for c in itertools.combinations(points, t)]
    uncovered = set(all_t)
    blocks = []

    def gain_of(x, block):
        # number of uncovered t-sets that contain x and t-1 points of block
        c = 0
        for rest in itertools.combinations(block, t - 1):
            if frozenset(rest + (x,)) in uncovered:
                c += 1
        return c

    while uncovered and time.time() < deadline:
        # seed with first uncovered subset
        seed = min(uncovered)
        block = list(seed)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in points:
                if x in block:
                    continue
                g = gain_of(x, block)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
        block.sort()
        blocks.append(block)
        for c in itertools.combinations(block, t):
            uncovered.discard(frozenset(c))

    # ---- Polish pass 1: drop redundant blocks ----
    def coverage_count(block, exclude_idx):
        # count how many t-sets of block are covered by other blocks
        cnt = 0
        for c in itertools.combinations(block, t):
            cs = frozenset(c)
            for j, other in enumerate(blocks):
                if j == exclude_idx:
                    continue
                if cs.issubset(other):
                    cnt += 1
                    break
        return cnt

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if coverage_count(blocks[i], i) == len(list(itertools.combinations(blocks[i], t))):
                blocks.pop(i)
                changed = True

    # ---- Polish pass 2: 1-for-1 swaps reducing block count is not possible,
    #      so instead try to replace a block with a strictly better one that
    #      covers at least as much (keeps count, may enable later drops). ----
    if time.time() < deadline:
        for i in range(len(blocks)):
            cur = blocks[i]
            # try a small perturbation: replace one point to increase unique coverage
            for pos in range(k):
                for x in points:
                    if x in cur:
                        continue
                    cand = cur[:pos] + [x] + cur[pos + 1:]
                    cand.sort()
                    # count unique t-sets covered by cand not covered by others
                    unique = 0
                    for c in itertools.combinations(cand, t):
                        cs = frozenset(c)
                        covered = False
                        for j, other in enumerate(blocks):
                            if j == i:
                                continue
                            if cs.issubset(other):
                                covered = True
                                break
                        if not covered:
                            unique += 1
                    # also count unique of current
                    cur_unique = 0
                    for c in itertools.combinations(cur, t):
                        cs = frozenset(c)
                        covered = False
                        for j, other in enumerate(blocks):
                            if j == i:
                                continue
                            if cs.issubset(other):
                                covered = True
                                break
                        if not covered:
                            cur_unique += 1
                    if unique > cur_unique:
                        blocks[i] = cand

        # re-run drop pass
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if coverage_count(blocks[i], i) == len(list(itertools.combinations(blocks[i], t))):
                    blocks.pop(i)
                    changed = True

    return blocks
