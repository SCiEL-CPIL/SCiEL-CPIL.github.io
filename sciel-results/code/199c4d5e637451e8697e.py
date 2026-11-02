"""Minimize a ternary covering code K_3(8,1) from the 486-word product construction."""
import time
import numpy as np


def _build_baseline(q, n, radius):
    # Product of the ternary tetracode (perfect radius-1 code on 4 coords) with the
    # full space on the remaining n-4 coords: 9 * 3**(n-4) words (486 for n=8).
    tetracode = np.array(
        [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    )
    tails = np.array(
        [list(t) for t in __import__("itertools").product(range(3), repeat=n - 4)]
    )
    return tetracode[:, None, :] + tails[None, :, :]


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under the 150 s wall-clock kill
    nq = q ** n

    code = _build_baseline(q, n, radius)
    m = len(code)
    if m == 0:
        return [list(map(int, w)) for w in code]

    # Per-codeword mask: mask[i, k, c] = 1 iff code[i][k] == c
    mask = np.zeros((m, n, q), dtype=np.int16)
    mask[np.arange(m)[:, None], code, np.arange(q)[None, :]] = 1

    # covered[k, c, tail] = number of codewords with value c at coord k and the given tail
    covered = mask.sum(axis=0)

    def neighbors_of(code_arr):
        # list of (coord, value) describing the 1 + n*(q-1) neighbours of each word
        coords = []
        vals = []
        for i in range(len(code_arr)):
            w = code_arr[i]
            for k in range(n):
                for v in range(q):
                    if v != w[k]:
                        coords.append(k)
                        vals.append(v)
        return np.array(coords, dtype=np.int64), np.array(vals, dtype=np.int64)

    coords, vals = neighbors_of(code)
    n_nb = len(coords)

    def removable_indices():
        # a word is removable iff all its neighbours are covered by >= 1 OTHER word
        sel = covered[coords, vals, np.arange(n_nb) % nq]  # (n_nb,) counts at each neighbour
        # group by word: word i owns neighbours [i*n_nb, (i+1)*n_nb)
        sel = sel.reshape(m, n_nb)
        return np.where(sel > 1)[0]

    while True:
        if time.time() > deadline:
            break
        rem = removable_indices()
        if len(rem) == 0:
            break
        keep = np.ones(m, dtype=bool)
        keep[rem] = False
        code = code[keep]
        mask = mask[keep]
        covered = mask.sum(axis=0)
        m = len(code)
        if m == 0:
            break

    return [list(map(int, w)) for w in code]
