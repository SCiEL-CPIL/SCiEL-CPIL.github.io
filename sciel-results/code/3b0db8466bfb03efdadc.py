"""Greedy covering (C(v,k,t)) with fast inverted-index gain + deletion pass."""
import time
import random
import itertools

_START = time.time()
_DEADLINE = 140.0   # hard kill is 150s; keep a safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _build_cover_index(blocks, v, t):
    """Map each t-subset to the number of blocks covering it."""
    cover = {}
    for blk in blocks:
        for s in itertools.combinations(blk, t):
            cover[s] = cover.get(s, 0) + 1
    return cover


def _deletion_pass(blocks, v, t, deadline_left):
    """Drop any block whose removal leaves every t-subset still covered.
    Repeat until no more deletions are possible."""
    cover = _build_cover_index(blocks, v, t)
    changed = True
    while changed and _time_left() > deadline_left:
        changed = False
        n = len(blocks)
        for i in range(n):
            blk = blocks[i]
            # If every t-subset of this block is covered by >=2 blocks, we can drop it.
            removable = True
            for s in itertools.combinations(blk, t):
                if cover.get(s, 0) <= 1:
                    removable = False
                    break
            if removable:
                for s in itertools.combinations(blk, t):
                    cover[s] -= 1
                blocks.pop(i)
                changed = True
                break  # restart scan after a mutation
    return blocks


def construct(v, k, t):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 140.0

    # ---- Phase 1: fast greedy covering ------------------------------------
    # uncovered: set of t-subsets still needing coverage.
    # by_first: first-element -> set of uncovered t-subsets starting with it.
    # by_last:  last-element  -> set of uncovered t-subsets ending with it.
    # For a block whose first element is fixed, the next point only needs to
    # cover (t-1)-subsets of the prefix, so we only consult by_last.
    uncovered = set(itertools.combinations(range(v), t))
    by_first = {}
    by_last = {}
    for s in uncovered:
        by_first.setdefault(s[0], set()).add(s)
        by_last.setdefault(s[-1], set()).add(s)

    blocks = []
    rng = random.Random(12345)
    while uncovered and _time_left() > 5.0:
        # Pick the first uncovered t-subset (deterministic); start a block with it.
        first = min(uncovered)
        block = [first[0]]
        # Remaining points: only candidates > first[0] are useful (the first
        # element of the block is already the smallest uncovered point).
        cand = [x for x in range(first[0] + 1, v)]
        while len(block) < k and cand and _time_left() > 5.0:
            prefix = block
            # For each candidate, count how many uncovered t-subsets it would cover.
            best = -1
            best_x = None
            for x in cand:
                gain = 0
                for rest in itertools.combinations(prefix, t - 1):
                    gain += 1 if (rest + (x,)) in by_last.get(x, ()) else 0
                if gain > best:
                    best = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            cand.remove(best_x)

        block.sort()
        blocks.append(block)
        # Remove all t-subsets now covered by this block.
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            if s in by_first.get(s[0], ()):
                by_first[s[0]].discard(s)
            if s in by_last.get(s[-1], ()):
                by_last[s[-1]].discard(s)

    # ---- Phase 2: deletion pass (strict improvement, same family) ----------
    if _time_left() > 3.0:
        blocks = _deletion_pass(blocks, v, t, 1.0)

    return blocks
