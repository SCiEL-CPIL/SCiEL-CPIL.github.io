"""Fast randomized greedy covering with NumPy bitsets + smooth top-k selection."""
import time
import random
import itertools

import numpy as np

# ---- global cache (populated once per process) -----------------------------
_V = _K = _T = None
_TRIPLES = None
_TRIPLE_IDX = None
_COMB = None
_RAND = None
_MASK_CACHE = {}


def _init(v, k, t, seed):
    global _V, _K, _T, _TRIPLES, _TRIPLE_IDX, _COMB, _RAND, _MASK_CACHE
    if _V is None or (_V, _K, _T) != (v, k, t):
        _V, _K, _T = v, k, t
        _MASK_CACHE = {}
        combos = list(itertools.combinations(range(v), t))
        _COMB = combos
        n = len(combos)
        _TRIPLES = np.array(combos, dtype=np.int32)
        _TRIPLE_IDX = np.zeros(n, dtype=np.int64)
        for i, c in enumerate(combos):
            _TRIPLE_IDX[i] = c[0] * (v * (v - 1) // 2) + c[1] * (v - 1) + c[2]
    if _RAND is None:
        _RAND = random.Random(seed)


def _block_mask(block):
    key = tuple(block)
    m = _MASK_CACHE.get(key)
    if m is not None:
        return m
    n = _TRIPLES.shape[0]
    m = np.zeros(n, dtype=bool)
    for i in range(len(block)):
        a = block[i]
        for j in range(i + 1, len(block)):
            b = block[j]
            for c in block[j + 1:]:
                m[_TRIPLE_IDX[a * (_V * (_V - 1) // 2) + b * (_V - 1) + c]] = True
    if len(_MASK_CACHE) < 300000:
        _MASK_CACHE[key] = m
    return m


def _greedy_restart(v, k, t, deadline, temp):
    """One randomized greedy pass. Returns (n_blocks, blocks) or (None, None)."""
    n = _TRIPLES.shape[0]
    covered = np.zeros(n, dtype=bool)
    blocks = []
    rng = _RAND
    while not covered.all():
        if time.time() >= deadline:
            return None, None
        # seed block from the first (lexicographically) still-uncovered triple
        first = int(np.argmax(~covered))
        block = list(_TRIPLES[first])
        inb = set(block)
        while len(block) < k:
            if time.time() >= deadline:
                return None, None
            gains = np.empty(v, dtype=np.float64)
            for x in range(v):
                if x in inb:
                    gains[x] = -1e9
                    continue
                g = 0
                for i in range(len(block)):
                    a = block[i]
                    for j in range(i + 1, len(block)):
                        b = block[j]
                        g += int(~covered[_TRIPLE_IDX[a * (_V * (_V - 1) // 2) + b * (_V - 1) + x]])
                gains[x] = g
            mx = gains.max()
            if mx <= 0:
                # no gain; fill with a random unused point to keep block size k
                cand = [x for x in range(v) if x not in inb]
                x = cand[rng.randrange(len(cand))] if cand else None
                if x is None:
                    break
                block.append(x)
                inb.add(x)
                continue
            # smooth top-k: softmax over the top candidates -> soft-argmax
            order = np.argsort(gains)[::-1]
            top = gains[order[:5]]
            z = np.exp((top - top.max()) / temp)
            p = z / z.sum()
            x = int(order[rng.choice(5, p=p)])
            block.append(x)
            inb.add(x)
        block.sort()
        covered |= _block_mask(block)
        blocks.append(block)
    return len(blocks), blocks


def _is_valid(blocks, v, t):
    n = _TRIPLES.shape[0]
    covered = np.zeros(n, dtype=bool)
    for b in blocks:
        covered |= _block_mask(b)
    return bool(covered.all())


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0          # safety margin below the 120 s kill
    _init(v, k, t, 0xC0FFEE)
    rng = _RAND

    # quick deterministic warm-start (temp=0 -> pure greedy)
    best_n, best_blocks = _greedy_restart(v, k, t, deadline, 0.0)
    if best_n is None:
        best_n, best_blocks = 0, []

    temps = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    it = 0
    while time.time() < deadline - 1.5:
        temp = temps[it % len(temps)]
        n, blocks = _greedy_restart(v, k, t, deadline, temp)
        if n is not None and (best_n is None or n < best_n):
            best_n, best_blocks = n, blocks
        it += 1

    if best_blocks is None:
        best_blocks = []
    if not _is_valid(best_blocks, v, t):
        # last-resort: rebuild deterministically if we somehow have a partial cover
        covered = np.zeros(_TRIPLES.shape[0], dtype=bool)
        for b in best_blocks:
            covered |= _block_mask(b)
        blocks = list(best_blocks)
        while not covered.all():
            if time.time() >= deadline - 0.2:
                break
            first = int(np.argmax(~covered))
            block = list(_TRIPLES[first])
            inb = set(block)
            while len(block) < k:
                gains = np.empty(v, dtype=np.float64)
                for x in range(v):
                    if x in inb:
                        gains[x] = -1e9
                        continue
                    g = 0
                    for i in range(len(block)):
                        a = block[i]
                        for j in range(i + 1, len(block)):
                            b = block[j]
                            g += int(~covered[_TRIPLE_IDX[a * (_V * (_V - 1) // 2) + b * (_V - 1) + x]])
                    gains[x] = g
                mx = gains.max()
                if mx <= 0:
                    cand = [x for x in range(v) if x not in inb]
                    x = cand[0] if cand else None
                    if x is None:
                        break
                    block.append(x)
                    inb.add(x)
                    continue
                x = int(np.argmax(gains))
                block.append(x)
                inb.add(x)
            block.sort()
            covered |= _block_mask(block)
            blocks.append(block)
        best_blocks = blocks
    return best_blocks
