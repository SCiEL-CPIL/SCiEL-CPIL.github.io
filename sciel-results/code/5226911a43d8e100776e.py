import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under the 150s kill

    # ---- Precompute t-subsets and bitset masks for every k-block ----
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    n_words = (n_t + 63) // 64

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    # block_masks[i]: np array of n_words uint64; block_t_idx[i]: list of t-subset indices
    block_masks = [None] * n_blocks
    block_t_idx = [None] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = np.zeros(n_words, dtype=np.uint64)
        idxs = []
        for sub in itertools.combinations(b, t):
            j = t_sub_index[sub]
            idxs.append(j)
            mask[j >> 6] |= (1 << (j & 63))
        block_masks[i] = mask
        block_t_idx[i] = idxs

    def elapsed():
        return time.time() - start_time

    # ---- Greedy construction (fast, valid) ----
    def greedy(seed):
        random.seed(seed)
        counts = [0] * n_t
        blocks = []
        uncovered = set(range(n_t))
        limit = deadline - 25
        while uncovered and time.time() < limit:
            idx = random.choice(tuple(uncovered))
            block = list(all_t_subs[idx])
            while len(block) < k:
                best_p, best_gain = -1, -1
                inb = set(block)
                for p in range(v):
                    if p in inb:
                        continue
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (p,)))
                        j = t_sub_index.get(sub, -1)
                        if j in uncovered:
                            g += 1
                    if g > best_gain:
                        best_gain, best_p = g, p
                    elif g == best_gain and best_p != -1 and random.random() < 0.5:
                        best_p = p
                if best_p == -1:
                    break
                block.append(best_p)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                j = t_sub_index[sub]
                counts[j] += 1
                if counts[j] == 1:
                    uncovered.discard(j)
        return blocks

    # ---- VND local search ----
    def vnd(initial_blocks):
        blocks = [list(b) for b in initial_blocks]
        counts = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for j in block_t_idx[block_to_idx[tuple(b)]]:
                counts[j] += 1
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)

        def try_remove(i):
            nonlocal best_blocks, best_len
            b = blocks[i]
            for j in block_t_idx[block_to_idx[tuple(b)]]:
                if counts[j] <= 1:
                    return False
            for j in block_t_idx[block_to_idx[tuple(b)]]:
                counts[j] -= 1
            blocks.pop(i)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [list(x) for x in blocks]
            return True

        def try_swap(i, x, y):
            nonlocal best_blocks, best_len
            b = blocks[i]
            idx_i = block_to_idx[tuple(b)]
            # subsets containing x must be redundant
            for j in block_t_idx[idx_i]:
                if x in all_t_subs[j] and counts[j] <= 1:
                    return False
            # apply
            for j in block_t_idx[idx_i]:
                if x in all_t_subs[j]:
                    counts[j] -= 1
            nb = b[:]
            nb.remove(x)
            nb.append(y)
            nb.sort()
            idx_n = block_to_idx[tuple(nb)]
            for j in block_t_idx[idx_n]:
                if y in all_t_subs[j]:
                    counts[j] += 1
            blocks[i] = nb
            # did this make block i removable?
            removable = True
            for j in block_t_idx[idx_n]:
                if counts[j] <= 1:
                    removable = False
                    break
            if removable:
                for j in block_t_idx[idx_n]:
                    counts[j] -= 1
                blocks.pop(i)
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = [list(x) for x in blocks]
            return True

        def try_2swap(i, j, x, y):
            nonlocal best_blocks, best_len
            b_i = blocks[i]; b_j = blocks[j]
            idx_i = block_to_idx[tuple(b_i)]; idx_j = block_to_idx[tuple(b_j)]
            # subsets of b_i containing x must be covered by b_j too
            for jj in block_t_idx[idx_i]:
                if x in all_t_subs[jj] and not (y in all_t_subs[jj] and counts[jj] >= 1):
                    # need coverage from b_j after swap; approximate: require current count>1
                    if counts[jj] <= 1:
                        return False
            # apply both
            for jj in block_t_idx[idx_i]:
                if x in all_t_subs[jj]:
                    counts[jj] -= 1
            for jj in block_t_idx[idx_j]:
                if y in all_t_subs[jj]:
                    counts[jj] -= 1
            nb_i = b_i[:]; nb_i.remove(x); nb_i.append(y); nb_i.sort()
            nb_j = b_j[:]; nb_j.remove(y); nb_j.append(x); nb_j.sort()
            idx_ni = block_to_idx[tuple(nb_i)]; idx_nj = block_to_idx[tuple(nb_j)]
            for jj in block_t_idx[idx_ni]:
                if y in all_t_subs[jj]:
                    counts[jj] += 1
            for jj in block_t_idx[idx_nj]:
                if x in all_t_subs[jj]:
                    counts[jj] += 1
            blocks[i] = nb_i; blocks[j] = nb_j
            # remove any of the two that became redundant
            for idx in (i, j):
                if idx >= len(blocks):
                    continue
                b = blocks[idx]
                ii = block_to_idx[tuple(b)]
                rem = True
                for jj in block_t_idx[ii]:
                    if counts[jj] <= 1:
                        rem = False
                        break
                if rem:
                    for jj in block_t_idx[ii]:
                        counts[jj] -= 1
                    blocks.pop(idx)
                    if idx < j and idx != i:
                        j -= 1
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [list(x) for x in blocks]
            return True

        while time.time() < deadline - 8:
            progress = False
            # 1) remove redundant
            for i in range(len(blocks)):
                if try_remove(i):
                    progress = True
                    break
            if progress:
                continue
            # 2) 1-exchange
            for i in range(len(blocks)):
                b = blocks[i]
                inb = set(b)
                for x in b:
                    for y in range(v):
                        if y in inb:
                            continue
                        if try_swap(i, x, y):
                            progress = True
                            break
                    if progress:
                        break
                if progress:
                    break
            if progress:
                continue
            # 3) 2-for-2 swap (sampled pairs to bound cost)
            if len(blocks) >= 2:
                done = False
                for i in range(len(blocks)):
                    if done:
                        break
                    for j in range(i + 1, len(blocks)):
                        if done:
                            break
                        if time.time() > deadline - 8:
                            done = True
                            break
                        b_i = blocks[i]; b_j = blocks[j]
                        in_i = set(b_i); in_j = set(b_j)
                        for x in b_i:
                            if x in in_j:
                                continue
                            for y in b_j:
                                if y in in_i:
                                    continue
                                if try_2swap(i, j, x, y):
                                    done = True
                                    break
                            if done:
                                break
                if done:
                    continue
            # no progress: perturb by a random valid swap, then continue
            if len(blocks) >= 1:
                i = random.randrange(len(blocks))
                b = blocks[i]
                x = random.choice(b)
                y = random.choice([p for p in range(v) if p not in set(b)])
                try_swap(i, x, y)
        return best_blocks

    # ---- Driver ----
    best = None
    best_len = float('inf')
    for seed in range(6):
        if elapsed() > deadline - 40:
            break
        g = greedy(seed)
        cand = vnd(g)
        if len(cand) < best_len:
            best_len = len(cand)
            best = cand
        if elapsed() > deadline - 12:
            break
    if best is None:
        best = vnd(greedy(0))
    return best
