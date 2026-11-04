import itertools
import random
import time

_cache = {}

def _prep(v, t):
    key = (v, t)
    if key in _cache:
        return _cache[key]
    subs = list(itertools.combinations(range(v), t))
    idx = {s: i for i, s in enumerate(subs)}
    m = len(subs)
    _cache[key] = (subs, idx, m)
    return _cache[key]

def _block_cover_indices(block, idx, t):
    out = []
    for sub in itertools.combinations(block, t):
        out.append(idx[sub])
    return out

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 135.0
    subs, idx, m = _prep(v, t)
    rng = random.Random(987654321)

    # Greedy
    uncovered = bytearray([1]) * m
    uncovered_count = m
    blocks = []
    block_covers = []

    while uncovered_count > 0:
        anchor = rng.randrange(m)
        while not uncovered[anchor]:
            anchor = rng.randrange(m)
        block = list(subs[anchor])
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ii = idx[tuple(sorted(rest + (x,)))]
                    if uncovered[ii]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        cov = _block_cover_indices(tuple(block), idx, t)
        for ii in cov:
            if uncovered[ii]:
                uncovered[ii] = 0
                uncovered_count -= 1
        blocks.append(tuple(block))
        block_covers.append(cov)

    cover_count = [0] * m
    for cov in block_covers:
        for ii in cov:
            cover_count[ii] += 1

    def removable(bi):
        for ii in block_covers[bi]:
            if cover_count[ii] < 2:
                return False
        return True

    def do_remove_pass():
        while True:
            n = len(blocks)
            if n == 0:
                return
            order = list(range(n))
            rng.shuffle(order)
            removed = False
            for bi in order:
                if removable(bi):
                    cov = block_covers[bi]
                    for ii in cov:
                        cover_count[ii] -= 1
                    blocks.pop(bi)
                    block_covers.pop(bi)
                    removed = True
                    break
            if not removed:
                return

    def perturb():
        cur = set(blocks)
        for _ in range(50):
            blk = tuple(sorted(rng.sample(range(v), k)))
            if blk not in cur:
                cov = _block_cover_indices(blk, idx, t)
                for ii in cov:
                    cover_count[ii] += 1
                blocks.append(blk)
                block_covers.append(cov)
                cur.add(blk)
                return

    best_n = len(blocks)
    best_snapshot = (list(blocks), [list(c) for c in block_covers], list(cover_count))

    def snapshot():
        return (list(blocks), [list(c) for c in block_covers], list(cover_count))

    def restore(snap):
        nonlocal blocks, block_covers, cover_count
        blocks = [tuple(b) for b in snap[0]]
        block_covers = [list(c) for c in snap[1]]
        cover_count = list(snap[2])

    do_remove_pass()
    if len(blocks) < best_n:
        best_n = len(blocks)
        best_snapshot = snapshot()

    while time.time() < DEADLINE:
        perturb()
        do_remove_pass()
        if len(blocks) < best_n:
            best_n = len(blocks)
            best_snapshot = snapshot()

    restore(best_snapshot)
    return [list(b) for b in blocks]
