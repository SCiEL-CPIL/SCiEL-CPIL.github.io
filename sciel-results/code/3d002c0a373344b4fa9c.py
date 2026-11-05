import time
import random
import numpy as np

_D = 7
_N = 2 ** _D

def _build():
    edges = []
    for x in range(_N):
        for i in range(_D):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    n = len(edges)
    eidx = {(x, y): k for k, (x, y) in enumerate(edges)}
    faces = np.zeros((n, _D - 1, 3), dtype=np.int32)
    for k, (x, y) in enumerate(edges):
        i = (x ^ y).bit_length() - 1
        row = 0
        for j in range(_D):
            if j == i:
                continue
            c = x ^ (1 << j)
            e2 = y ^ (1 << j)
            faces[k, row, 0] = eidx[(min(x, c), max(x, c))]
            faces[k, row, 1] = eidx[(min(y, e2), max(y, e2))]
            faces[k, row, 2] = eidx[(min(c, e2), max(c, e2))]
            row += 1
    return edges, faces

def construct():
    t0 = time.time()
    deadline = t0 + 138.0
    edges, faces = _build()
    n = len(edges)
    base = np.arange(n)

    def greedy(order):
        chosen = np.zeros(n, dtype=bool)
        cnt = 0
        for k in order:
            rows = chosen[faces[k]]
            if not rows.all(axis=1).any():
                chosen[k] = True
                cnt += 1
        return chosen, cnt

    chosen, cnt = greedy(base)
    best_chosen, best_count = chosen, cnt

    rng = random.Random(2024)
    # diverse orderings: random shuffles + structured (by bit, by endpoint)
    while time.time() < deadline - 4.0:
        mode = rng.random()
        if mode < 0.7:
            order = base.copy()
            rng.shuffle(order.tolist())
            order = np.array(order.tolist())
        elif mode < 0.85:
            # sort by first endpoint
            order = np.argsort([edges[k][0] for k in base])
        else:
            # sort by bit index then endpoint
            order = np.argsort([(edges[k][0] ^ edges[k][1]).bit_length() - 1, edges[k][0]] for k in base)
        chosen, cnt = greedy(order)
        if cnt > best_count:
            best_count = cnt
            best_chosen = chosen

    out = [list(edges[k]) for k in range(n) if best_chosen[k]]
    return out
