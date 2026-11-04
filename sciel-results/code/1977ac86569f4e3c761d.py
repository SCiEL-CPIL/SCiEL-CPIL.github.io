import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 146.0  # safety margin before 150s kill

    # ---- Precompute all t-subsets and index them ----
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # ---- Precompute all k-subsets and their coverage masks (bit i = covers t-sub i) ----
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # Masks as Python ints (arbitrary precision -> handles n_t > 64)
    block_masks = [0] * n_k
    for i, k_sub in enumerate(all_k_subs):
        m = 0
        for t_sub in itertools.combinations(k_sub, t):
            m |= (1 << t_sub_to_idx[t_sub])
        block_masks[i] = m

    # Reverse map: for each t-subset, which blocks contain it
    t_to_blocks = [[] for _ in range(n_t)]
    for i, m in enumerate(block_masks):
        # iterate set bits
        mm = m
        while mm:
            low = mm & (-mm)
            bit = low.bit_length() - 1
            t_to_blocks[bit].append(i)
            mm ^= low

    # numpy array of masks for vectorized popcount (split into two 64-bit halves)
    lo = np.array([m & ((1 << 64) - 1) for m in block_masks], dtype=np.uint64)
    hi = np.array([m >> 64 for m in block_masks], dtype=np.uint64)

    def time_ok():
        return time.time() < deadline - 6.0

    def greedy_cover(seed):
        rng = random.Random(seed)
        # uncovered mask: all bits set
        full_mask = (1 << n_t) - 1
        unc = full_mask
        chosen = []
        used = set()
        while unc:
            if seed > 0:
                # pick a random uncovered t-subset
                mm = unc
                r = rng.randrange(mm.bit_length())
                # find r-th set bit
                cnt = -1
                mm2 = unc
                while True:
                    low = mm2 & (-mm2)
                    mm2 ^= low
                    cnt += 1
                    if cnt == r:
                        s = low.bit_length() - 1
                        break
            else:
                s = (unc & (-unc)).bit_length() - 1  # lowest set bit

            cands = t_to_blocks[s]
            best_idx = -1
            best_gain = -1
            for b in cands:
                if b in used:
                    continue
                gain = bin(block_masks[b] & unc).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b
                    if best_gain == 1:
                        break
            if best_idx == -1:
                break
            chosen.append(best_idx)
            used.add(best_idx)
            unc &= ~block_masks[best_idx]
        return chosen

    def reduce_blocks(chosen):
        if not chosen:
            return chosen
        # cover count per t-subset
        cover = np.zeros(n_t, dtype=np.int32)
        for b in chosen:
            m = block_masks[b]
            mm = m
            while mm:
                low = mm & (-mm)
                cover[low.bit_length() - 1] += 1
                mm ^= low
        changed = True
        while changed:
            changed = False
            newc = []
            for b in chosen:
                m = block_masks[b]
                essential = False
                mm = m
                while mm:
                    low = mm & (-mm)
                    if cover[low.bit_length() - 1] == 1:
                        essential = True
                        break
                    mm ^= low
                if essential:
                    newc.append(b)
                else:
                    changed = True
                    mm = m
                    while mm:
                        low = mm & (-mm)
                        cover[low.bit_length() - 1] -= 1
                        mm ^= low
            chosen = newc
        return chosen

    def local_search(chosen, max_iter=300):
        cur = list(chosen)
        if not cur:
            return cur
        cover = np.zeros(n_t, dtype=np.int32)
        for b in cur:
            m = block_masks[b]
            mm = m
            while mm:
                low = mm & (-mm)
                cover[low.bit_length() - 1] += 1
                mm ^= low
        improved = True
        it = 0
        while improved and it < max_iter and time_ok():
            improved = False
            it += 1
            for i, b in enumerate(cur):
                m = block_masks[b]
                can = True
                mm = m
                while mm:
                    low = mm & (-mm)
                    if cover[low.bit_length() - 1] <= 1:
                        can = False
                        break
                    mm ^= low
                if can:
                    mm = m
                    while mm:
                        low = mm & (-mm)
                        cover[low.bit_length() - 1] -= 1
                        mm ^= low
                    cur.pop(i)
                    improved = True
                    break
        return cur

    # ---- Deterministic greedy first ----
    best = greedy_cover(0)
    best = reduce_blocks(best)
    best_len = len(best)

    # ---- Randomized restarts (now cheap per-iteration thanks to fast masks) ----
    seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
             71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
             151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
             233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313]
    for seed in seeds:
        if not time_ok():
            break
        cand = greedy_cover(seed)
        cand = reduce_blocks(cand)
        if cand and len(cand) < best_len:
            best = cand
            best_len = len(cand)

    # ---- Polish best solution ----
    if best and time_ok():
        pol = local_search(best, max_iter=400)
        if len(pol) < best_len:
            best = pol
            best_len = len(pol)
    if best and time.time() < deadline - 3.0:
        best = reduce_blocks(best)

    return [list(all_k_subs[b]) for b in best]
