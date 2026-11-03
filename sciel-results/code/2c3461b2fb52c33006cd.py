import time
import numpy as np


def _build_neighbors(q, n):
    N = q ** n
    m = 1 + n * (q - 1)
    base = np.empty((N, m), dtype=np.int64)
    base[:, 0] = np.arange(N)
    pos = 0
    for i in range(n):
        pw = q ** i
        for d in range(q - 1):
            base[:, pos + 1] = base[:, 0] + pw * (d + 1)
            pos += 1
    base[:, pos + 1] = base[:, 0] - pw
    return base


def construct(q, n, radius):
    start = time.time()
    deadline = start + 120.0  # safety margin before the 150s kill

    if radius <= 0:
        return [list(range(q)) * 0 for _ in range(q ** n)]

    if n == 0:
        return [[]]

    N = q ** n
    m = 1 + n * (q - 1)
    nbr = _build_neighbors(q, n)

    cov = np.zeros(N, dtype=np.int32)
    chosen = []
    for c in range(N):
        cov[nbr[c]] += 1
        chosen.append(c)

    # Greedy reduction: drop codewords while every word stays covered.
    while True:
        if time.time() > deadline - 1.0:
            break
        improved = False
        for idx in range(len(chosen) - 1, -1, -1):
            c = chosen[idx]
            cov[nbr[c]] -= 1
            if cov.min() < 0:
                cov[nbr[c]] += 1
            else:
                chosen.pop(idx)
                improved = True
                break
        if not improved:
            break

    # Decode
    pw = [q ** i for i in range(n)]
    out = []
    for c in chosen:
        w = []
        t = c
        for i in range(n):
            w.append(t % q)
            t //= q
        out.append(w)
    return out
