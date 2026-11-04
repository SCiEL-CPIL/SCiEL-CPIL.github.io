import time
import itertools
from collections import defaultdict

_DEADLINE = 120.0
_START = time.time()


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _covers(block, t):
    return itertools.combinations(block, t)


def _greedy(v, k, t, seed):
    """Fast incremental greedy set cover. Returns list of blocks (may be incomplete)."""
    rng = _rng(seed)
    pts = list(range(v))
    rng.shuffle(pts)

    # uncovered: set of t-subsets still to cover
    uncovered = set(itertools.combinations(range(v), t))
    if not uncovered:
        return []

    # cnt[x] = number of uncovered t-subsets that contain x
    cnt = defaultdict(int)
    for sub in uncovered:
        for x in sub:
            cnt[x] += 1

    blocks = []
    while uncovered:
        if _remaining_time() < 1.0:
            break
        # start block from an uncovered t-subset, choose the one whose points
        # have the largest total remaining coverage (break ties randomly).
        best_sub = None
        best_key = None
        for sub in uncovered:
            key = sum(cnt[x] for x in sub)
            if best_key is None or key > best_key:
                best_key = key
                best_sub = sub
                if rng.random() < 0.02:  # occasional randomization for diversity
                    break
        block = list(best_sub)
        block_set = set(block)

        while len(block) < k:
            if _remaining_time() < 0.5:
                break
            # pick candidate x (not in block) maximizing cnt[x] (tie -> random)
            best_x = None
            best_val = -1
            best_val2 = None
            for x in pts:
                if x in block_set:
                    continue
                val = cnt[x]
                if val > best_val:
                    best_val = val
                    best_x = x
                    best_val2 = None
                elif val == best_val and best_val > 0:
                    # tie: keep a random candidate
                    if best_val2 is None or rng.random() < 0.5:
                        best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)

            # update: for each (t-1)-subset of the current block, the t-subset
            # formed by adding best_x is now covered (if it was uncovered).
            for rest in itertools.combinations(block[:-1], t - 1):
                newsub = tuple(sorted(rest + (best_x,)))
                if newsub in uncovered:
                    uncovered.discard(newsub)
                    for x in newsub:
                        cnt[x] -= 1
                    # cnt[best_x] already decremented via x == best_x in loop
        # Note: cnt[best_x] is decremented inside the loop for each newsub that
        # contains best_x (always true), so it stays consistent.

        block.sort()
        blocks.append(block)
        # final safety: remove any t-subsets of the block still marked uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    return blocks


def _is_valid(blocks, v, k, t):
    seen = set()
    covered = set()
    for b in blocks:
        if len(set(b)) != k:
            return False
        key = tuple(b)
        if key in seen:
            return False
        seen.add(key)
        for sub in itertools.combinations(b, t):
            covered.add(sub)
    total = len(list(itertools.combinations(range(v), t)))
    return len(covered) == total


def _rng(seed):
    import random
    return random.Random(seed)


def construct(v, k, t):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 120.0

    # quick sanity / trivial
    if t <= 0:
        return []
    if k >= v:
        return [list(range(v))] if t <= v else []

    # A single greedy pass as a safe baseline.
    best = _greedy(v, k, t, 0)
    if _is_valid(best, v, k, t):
        best_blocks = list(best)
    else:
        best_blocks = None

    # Multi-restart: try to beat the current best with fresh random seeds.
    seed = 1
    while _remaining_time() > 8.0:
        cand = _greedy(v, k, t, seed)
        seed += 1
        if len(cand) < len(best_blocks or []) and _is_valid(cand, v, k, t):
            best_blocks = list(cand)
        # stop early if we hit a strong lower bound heuristic
        if best_blocks is not None and len(best_blocks) <= _lower_bound(v, k, t):
            break

    if best_blocks is None:
        # Fallback: return whatever we have (may be invalid) to avoid crash.
        return best if best is not None else []
    return best_blocks


def _lower_bound(v, k, t):
    # Simple counting lower bound: ceil(C(v,t) / C(k,t))
    def C(n, r):
        if r < 0 or r > n:
            return 0
        import math
        return math.comb(n, r)
    num = C(v, t)
    den = C(k, t)
    if den == 0:
        return 0
    return -(-num // den)
