import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0

    space = q ** n

    def w2i(w):
        i = 0
        for v in w:
            i = i * q + v
        return i

    def i2w(i):
        w = [0] * n
        for j in range(n - 1, -1, -1):
            w[j] = i % q
            i //= q
        return w

    # Precompute neighbor index lists for every word in the space (distance <= radius).
    # For radius 1: the word itself plus all single-position changes.
    if radius >= 1:
        nbrs = []
        for i in range(space):
            w = i2w(i)
            lst = [i]
            for p in range(n):
                ov = w[p]
                for nv in range(q):
                    if nv != ov:
                        w[p] = nv
                        lst.append(w2i(w))
                w[p] = ov
            nbrs.append(lst)
    else:
        nbrs = [[i] for i in range(space)]

    # Product construction start: tetracode (length 4, size 9) x full space (length n-4).
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    base = [h + list(t) for h in tetracode for t in tails]
    base_idx = [w2i(c) for c in base]

    def check_valid(idx_set):
        cnt = np.zeros(space, dtype=np.int32)
        for i in idx_set:
            cnt[nbrs[i]] += 1
        return bool((cnt > 0).all())

    best = None
    best_len = 10 ** 9
    if check_valid(base_idx):
        best = list(base_idx)
        best_len = len(base_idx)

    # Randomized greedy deletion with tie-breaking by coverage slack.
    def greedy(idx_set, order, rng):
        cur = list(idx_set)
        pos = {x: k for k, x in enumerate(cur)}
        cnt = np.zeros(space, dtype=np.int32)
        for i in cur:
            cnt[nbrs[i]] += 1
        for c in order:
            if c not in pos:
                continue
            nb = nbrs[c]
            if (cnt[nb] <= 1).any():
                continue  # essential: some word only this codeword covers
            k = pos[c]
            del cur[k]
            del pos[c]
            cnt[nb] -= 1
        return cur

    # Local swap: replace one codeword with another to enable further deletions.
    def swap_search(cur, rng, max_iters=3000):
        cnt = np.zeros(space, dtype=np.int32)
        for i in cur:
            cnt[nbrs[i]] += 1
        cur_set = set(cur)
        all_idx = list(range(space))
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            if time.time() > deadline - 2.0:
                break
            for c in list(cur):
                if time.time() > deadline - 2.0:
                    break
                nb = nbrs[c]
                if (cnt[nb] <= 1).any():
                    continue  # c essential, cannot remove
                # try to move c to a better location
                cand = [x for x in all_idx if x not in cur_set]
                rng.shuffle(cand)
                for x in cand[:40]:
                    # simulate remove c, add x
                    cnt[nb] -= 1
                    cnt[nbrs[x]] += 1
                    if not (cnt > 0).all():
                        cnt[nb] += 1
                        cnt[nbrs[x]] -= 1
                        continue
                    # now try to delete more
                    new_cur = [y for y in cur if y != c] + [x]
                    # quick deletion pass
                    for y in list(new_cur):
                        nby = nbrs[y]
                        if (cnt[nby] <= 1).any():
                            continue
                        cnt[nby] -= 1
                        new_cur.remove(y)
                    if len(new_cur) < len(cur):
                        # commit
                        cnt = np.zeros(space, dtype=np.int32)
                        for y in new_cur:
                            cnt[nbrs[y]] += 1
                        cur = new_cur
                        cur_set = set(cur)
                        improved = True
                        break
                    else:
                        # revert
                        cnt[nb] += 1
                        cnt[nbrs[x]] -= 1
                        # restore deleted ones
                        # recompute is safer
                        cnt = np.zeros(space, dtype=np.int32)
                        for y in cur:
                            cnt[nbrs[y]] += 1
                if improved:
                    break
        return cur

    # Multiple randomized restarts.
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for s in seeds:
        if time.time() > deadline - 10.0:
            break
        rng = random.Random(s)
        order = base_idx[:]
        rng.shuffle(order)
        res = greedy(base_idx, order, rng)
        if len(res) < best_len:
            best = res
            best_len = len(res)
        # swap polish on the current best
        res2 = swap_search(res, rng, max_iters=1500)
        if check_valid(res2) and len(res2) < best_len:
            best = res2
            best_len = len(res2)

    if best is None:
        return [i2w(i) for i in base_idx]
    return [i2w(i) for i in best]
