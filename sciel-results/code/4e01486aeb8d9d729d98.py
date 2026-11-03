"""Optimized greedy covering for C(v,k,t): bitset acceleration + multi-restart."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _precompute(v, k, t):
    tsubs = list(itertools.combinations(range(v), t))
    nT = len(tsubs)
    tsub_index = {s: i for i, s in enumerate(tsubs)}

    cover = [0] * v
    for i, s in enumerate(tsubs):
        bit = 1 << i
        for x in s:
            cover[x] |= bit

    blocks = list(itertools.combinations(range(v), k))
    block_mask = []
    for b in blocks:
        m = 0
        for s in itertools.combinations(b, t):
            m |= 1 << tsub_index[s]
        block_mask.append(m)

    cover2 = {}
    for a in range(v):
        for b in range(a + 1, v):
            m = 0
            for s in tsubs:
                if a in s and b in s:
                    m |= 1 << tsub_index[s]
            cover2[(a, b)] = m
    return tsubs, tsub_index, blocks, block_mask, cover, cover2


def _greedy(tsubs, blocks, block_mask, cover, cover2, all_mask, v, k, seed_order=None):
    nT = len(tsubs)
    uncovered = all_mask
    used = set()
    out_blocks = []

    def new_mask(block, x):
        L = len(block)
        if L == 0:
            return cover[x]
        if L == 1:
            a = block[0]
            return cover2[(a, x)] if a < x else cover2[(x, a)]
        m = 0
        for i in range(L):
            for j in range(i + 1, L):
                a, b = block[i], block[j]
                m |= cover2[(a, b)] if a < b else cover2[(b, a)]
        return m & cover[x]

    if seed_order is None:
        def next_seed():
            if not uncovered:
                return None
            i = (uncovered & -uncovered).bit_length() - 1
            return tsubs[i]
    else:
        state = {'idx': 0}
        def next_seed():
            while state['idx'] < len(seed_order):
                i = seed_order[state['idx']]
                state['idx'] += 1
                if (uncovered >> i) & 1:
                    return tsubs[i]
            return None

    while True:
        seed = next_seed()
        if seed is None:
            break
        block = list(seed)
        while len(block) < k:
            best_x, best_gain = -1, -1
            for x in range(v):
                if x in block:
                    continue
                g = (uncovered & new_mask(block, x)).bit_count()
                if g > best_gain:
                    best_gain, best_x = g, x
            block.append(best_x)
        block.sort()
        bi = blocks.index(block)
        if bi in used:
            continue
        used.add(bi)
        out_blocks.append(block)
        uncovered &= ~block_mask[bi]
    return out_blocks, uncovered


def construct(v, k, t):
    start = time.time()
    DEADLINE = 145.0
    tsubs, tsub_index, blocks, block_mask, cover, cover2 = _precompute(v, k, t)
    nT = len(tsubs)
    all_mask = (1 << nT) - 1

    blocks1, unc1 = _greedy(tsubs, blocks, block_mask, cover, cover2, all_mask, v, k)
    best = (len(blocks1), blocks1)

    rng = random.Random(12345)
    nT_list = list(range(nT))
    attempts = 0
    while time.time() < DEADLINE - 25.0 and attempts < 2000:
        attempts += 1
        order = nT_list[:]
        rng.shuffle(order)
        b, unc = _greedy(tsubs, blocks, block_mask, cover, cover2, all_mask, v, k, seed_order=order)
        if unc == 0 and len(b) < best[0]:
            best = (len(b), b)
    return best[1]
# EVOLVE-BLOCK-END
