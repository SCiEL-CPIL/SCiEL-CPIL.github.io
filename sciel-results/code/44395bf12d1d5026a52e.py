"""Bitmask-accelerated randomized greedy covering with restarts and polish."""
import time
import random
import itertools

_DEADLINE = 118.0  # safety margin before the 150s kill


def _build_masks(v, k, t):
    """Map every t-subset to an index; precompute coverage mask per k-subset."""
    idx = {}
    count = 0
    for c in itertools.combinations(range(v), t):
        idx[c] = count
        count += 1
    total = count
    masks = {}
    for blk in itertools.combinations(range(v), k):
        m = 0
        for sub in itertools.combinations(blk, t):
            m |= 1 << idx[sub]
        masks[blk] = m
    return idx, masks, total


def _greedy(v, k, t, idx, masks, rng, deadline, target=None):
    """One randomized greedy pass. Returns (blocks, uncovered_mask)."""
    full = (1 << len(idx)) - 1
    uncovered = full
    blocks = []
    while uncovered:
        if time.time() > deadline:
            return None, uncovered
        if target is not None:
            # pick the target subset with the fewest covering blocks (tightest)
            best_target = None
            best_cov = None
            for sub, i in idx.items():
                if (uncovered >> i) & 1:
                    cov = bin(masks[sub] & uncovered).count('1')
                    if best_cov is None or cov < best_cov or (cov == best_cov and rng.random() < 0.5):
                        best_cov = cov
                        best_target = sub
            target = best_target
        else:
            # random uncovered target
            sub = None
            for _ in range(64):
                i = rng.randrange(len(idx))
                if (uncovered >> i) & 1:
                    # find the sub for index i
                    for s, j in idx.items():
                        if j == i:
                            sub = s
                            break
                    break
            if sub is None:
                for s, i in idx.items():
                    if (uncovered >> i) & 1:
                        sub = s
                        break
            if sub is None:
                break
        block = list(sub)
        while len(block) < k:
            if time.time() > deadline:
                return None, uncovered
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            # compute gain for each candidate
            best_x = None
            best_gain = -1
            for x in candidates:
                cand = tuple(sorted(block + [x]))
                g = bin(masks[cand] & uncovered).count('1')
                if g > best_gain or (g == best_gain and rng.random() < 0.5):
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        block.sort()
        blocks.append(block)
        uncovered &= ~masks[tuple(block)]
    return blocks, uncovered


def _polish(v, k, t, idx, masks, blocks, deadline):
    """Try to reduce block count by replacing a block with a better one."""
    blocks = [list(b) for b in blocks]
    full = (1 << len(idx)) - 1
    # coverage mask per block
    cov = [masks[tuple(b)] for b in blocks]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # recompute total coverage
        total = 0
        for c in cov:
            total |= c
        if total != full:
            break
        # for each block, see if removing it still leaves full coverage
        for i in range(len(blocks)):
            rest = total & ~cov[i]
            if rest == full:
                blocks.pop(i)
                cov.pop(i)
                total = rest
                improved = True
                break
    # try to merge: replace two blocks with one that covers their union if possible
    # (heuristic: only if union is subset of some single block's coverage)
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)

    idx, masks, total = _build_masks(v, k, t)

    best_blocks = None
    best_count = float('inf')

    # First deterministic-ish pass to get a baseline
    b, unc = _greedy(v, k, t, idx, masks, rng, deadline, target=None)
    if b is not None and unc == 0:
        best_blocks = b
        best_count = len(b)

    # Randomized restarts
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        b, unc = _greedy(v, k, t, idx, masks, rng, deadline, target=None)
        if b is not None and unc == 0:
            if len(b) < best_count:
                best_count = len(b)
                best_blocks = b

    # Polish the best solution
    if best_blocks is not None and time.time() < deadline - 2:
        polished = _polish(v, k, t, idx, masks, best_blocks, deadline)
        # verify polished still covers everything
        full = (1 << total) - 1
        total_cov = 0
        for bl in polished:
            total_cov |= masks[tuple(bl)]
        if total_cov == full and len(polished) < best_count:
            best_blocks = polished

    if best_blocks is None:
        # fallback: exhaustive-ish greedy (should not happen)
        blocks = []
        uncovered = (1 << total) - 1
        while uncovered and time.time() < deadline:
            # find a t-subset that is uncovered
            target = None
            for s, i in idx.items():
                if (uncovered >> i) & 1:
                    target = s
                    break
            if target is None:
                break
            block = list(target)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    cand = tuple(sorted(block + [x]))
                    g = bin(masks[cand] & uncovered).count('1')
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered &= ~masks[tuple(block)]
        best_blocks = blocks

    return best_blocks
