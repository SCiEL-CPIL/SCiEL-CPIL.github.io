"""Covering design C(v,k,t): fast vectorized greedy + randomized restarts + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

DEADLINE = 140.0  # wall-clock safety margin under the 150s kill


def _all_t_subsets(v, t):
    return np.array(list(itertools.combinations(range(v), t)), dtype=np.int32)


def _point_masks(subsets, v):
    """point_mask[x]: uint64 bitmask over rows of `subsets` where point x appears."""
    n_t = subsets.shape[0]
    n_words = (n_t + 63) // 64
    point_mask = np.zeros((v, n_words), dtype=np.uint64)
    for x in range(v):
        hit = np.any(subsets == x, axis=1)
        idx = np.nonzero(hit)[0]
        words = idx // 64
        bits = (np.uint64(1) << (idx % 64))
        for w, b in zip(words, bits):
            point_mask[x, w] |= b
    return point_mask


def _pc(arr):
    total = 0
    for val in arr:
        v = int(val)
        while v:
            total += v & 1
            v >>= 1
    return total


def _greedy_cover(subsets, v, k, t, rng):
    n_t = subsets.shape[0]
    n_words = (n_t + 63) // 64
    point_mask = _point_masks(subsets, v)
    mask = np.zeros(n_words, dtype=np.uint64)
    for w in range(n_words):
        mask[w] = np.uint64(0xFFFFFFFFFFFFFFFF)
    rem = n_t % 64
    if rem:
        mask[-1] = (np.uint64(1) << np.uint64(rem)) - 1

    blocks = []
    while True:
        row = -1
        for w in range(n_words):
            if mask[w]:
                low = int(mask[w] & (-mask[w]))
                bit = low.bit_length() - 1 if low else 0
                row = w * 64 + bit
                break
        if row < 0:
            break
        seed = list(subsets[row])
        block = list(seed)
        block_set = set(seed)
        bm = point_mask[seed[0]].copy()
        for x in seed[1:]:
            bm |= point_mask[x]

        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                newbits = (bm | point_mask[x]) & np.invert(bm) & mask
                g = _pc(newbits)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            block_set.add(best_x)
            bm |= point_mask[best_x]

        block.sort()
        blocks.append(block)
        for x in block:
            mask &= np.invert(point_mask[x])
    return blocks


def _cover_check(blocks, subsets, t):
    n_t = subsets.shape[0]
    covered = np.zeros(n_t, dtype=bool)
    for b in blocks:
        ba = np.array(b, dtype=np.int32)
        covered |= np.all(np.isin(subsets, ba), axis=1)
        if covered.all():
            return True
    return covered.all()


def _local_search(blocks, subsets, v, k, t, rng, deadline):
    n_t = subsets.shape[0]
    best = list(blocks)

    def recompute_cov(blks):
        cov = np.zeros(n_t, dtype=np.int32)
        for b in blks:
            ba = np.array(b, dtype=np.int32)
            cov[np.all(np.isin(subsets, ba), axis=1)] += 1
        return cov

    cov = recompute_cov(best)
    improved = True
    iters = 0
    while improved and time.time() < deadline and iters < 400:
        improved = False
        iters += 1
        order = list(range(len(best)))
        rng.shuffle(order)
        for bi in order:
            if time.time() > deadline:
                break
            if bi >= len(best):
                continue
            b = best[bi]
            ba = np.array(b, dtype=np.int32)
            hit = np.all(np.isin(subsets, ba), axis=1)
            cov2 = cov - hit.astype(np.int32)
            if np.any(cov2 < 0):
                continue
            uniq = hit & (cov2 == 0)
            if not uniq.any():
                best.pop(bi)
                cov = recompute_cov(best)
                improved = True
                break
            uniq_idx = np.nonzero(uniq)[0]
            sub = subsets[uniq_idx]
            n_sub = sub.shape[0]
            nw = (n_sub + 63) // 64
            spoint = _point_masks(sub, v)
            smask = np.zeros(nw, dtype=np.uint64)
            for w in range(nw):
                smask[w] = np.uint64(0xFFFFFFFFFFFFFFFF)
            rem = n_sub % 64
            if rem:
                smask[-1] = (np.uint64(1) << np.uint64(rem)) - 1
            seed = list(sub[0])
            blk = list(seed)
            blk_set = set(seed)
            bm = spoint[seed[0]].copy()
            for x in seed[1:]:
                bm |= spoint[x]
            while len(blk) < k:
                bx = -1
                bg = -1
                for x in range(v):
                    if x in blk_set:
                        continue
                    nb = (bm | spoint[x]) & np.invert(bm) & smask
                    g = _pc(nb)
                    if g > bg:
                        bg = g
                        bx = x
                if bx < 0:
                    break
                blk.append(bx)
                blk_set.add(bx)
                bm |= spoint[bx]
            blk.sort()
            ba2 = np.array(blk, dtype=np.int32)
            if not np.all(np.isin(sub, ba2), axis=1).all():
                continue
            best[bi] = blk
            cov = recompute_cov(best)
            improved = True
            break
    return best


def construct(v, k, t):
    rng = random.Random(12345)
    subsets = _all_t_subsets(v, t)
    n_t = subsets.shape[0]

    best_blocks = None
    best_len = 10**9

    def eval_candidate(b):
        nonlocal best_blocks, best_len
        if not _cover_check(b, subsets, t):
            return False
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = list(b)
        return True

    while time.time() < DEADLINE - 25:
        perm = np.arange(n_t)
        rng.shuffle(perm)
        blocks = _greedy_cover(subsets[perm], v, k, t, rng)
        eval_candidate(blocks)
        if best_blocks is not None and time.time() < DEADLINE - 8:
            ls = _local_search(best_blocks, subsets, v, k, t, rng, DEADLINE - 4)
            eval_candidate(ls)
            if time.time() > DEADLINE - 25:
                break

    if best_blocks is None:
        best_blocks = _greedy_cover(subsets, v, k, t, rng)
        eval_candidate(best_blocks)

    if best_blocks is not None and time.time() < DEADLINE - 3:
        ls = _local_search(best_blocks, subsets, v, k, t, rng, DEADLINE - 2)
        eval_candidate(ls)

    if best_blocks is None:
        best_blocks = []

    seen = set()
    out = []
    for b in best_blocks:
        b = sorted(set(b))
        if len(b) != k:
            continue
        key = tuple(b)
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out
# EVOLVE-BLOCK-END
