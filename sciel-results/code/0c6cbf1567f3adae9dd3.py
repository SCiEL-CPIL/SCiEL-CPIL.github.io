import time
import random
import numpy as np
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0

    n = 16
    combos = list(combinations(range(1, n + 1), 4))
    m = len(combos)
    masks = np.array([sum(1 << (x - 1) for x in c) for c in combos], dtype=np.int64)

    def conflict_with(x_idx, sel_arr):
        t = len(sel_arr)
        if t < 2:
            return False
        sx = masks[x_idx]
        sm = masks[sel_arr]
        I = sx & sm
        P = sm[:, None] & sm[None, :]
        cond = (P == I[:, None]) & (P == I[None, :])
        np.fill_diagonal(cond, False)
        return bool(cond.any())

    def local_add(sel, sel_set, sel_arr):
        while time.time() < deadline:
            candidates = [i for i in range(m) if i not in sel_set]
            if not candidates:
                break
            rng.shuffle(candidates)
            added = False
            for x in candidates:
                if not conflict_with(x, sel_arr):
                    sel.append(x)
                    sel_set.add(x)
                    sel_arr = np.append(sel_arr, x)
                    added = True
            if not added:
                break
        return sel, sel_set, sel_arr

    best = None
    best_size = 0
    rng = random.Random(20240517)

    # initial deterministic greedy for a safe baseline
    sel = []
    sel_set = set()
    sel_arr = np.array([], dtype=np.int64)
    for x in range(m):
        if not conflict_with(x, sel_arr):
            sel.append(x)
            sel_set.add(x)
            sel_arr = np.append(sel_arr, x)
    if len(sel) > best_size:
        best_size = len(sel)
        best = list(sel)

    it = 0
    while time.time() < deadline:
        it += 1
        order = list(range(m))
        rng.shuffle(order)
        sel = []
        sel_set = set()
        sel_arr = np.array([], dtype=np.int64)
        for x in order:
            if time.time() > deadline:
                break
            if not conflict_with(x, sel_arr):
                sel.append(x)
                sel_set.add(x)
                sel_arr = np.append(sel_arr, x)
        sel, sel_set, sel_arr = local_add(sel, sel_set, sel_arr)
        # kicks to escape local optima
        for _ in range(4):
            if time.time() > deadline:
                break
            if len(sel) < 3:
                break
            k = min(len(sel), rng.randint(2, 6))
            victims = rng.sample(sel, k)
            for v in victims:
                sel.remove(v)
                sel_set.discard(v)
            sel_arr = np.array([i for i in sel], dtype=np.int64)
            sel, sel_set, sel_arr = local_add(sel, sel_set, sel_arr)
        if len(sel) > best_size:
            best_size = len(sel)
            best = list(sel)

    if best is None:
        best = list(range(min(m, 1)))
        best_size = len(best)

    return [sorted(combos[i]) for i in best]
