"""Mutually orthogonal Latin squares of order 10.

We need up to 9 MOLS of order 10, each a 10x10 list of Python ints in 0..9.
The known maximum number of MOLS of order 10 is 9.  We construct them from the
group Z_5 x Z_2 by choosing a family of linear maps and then greedily selecting
a large mutually orthogonal set.

Each symbol n in 0..9 corresponds to (a, b) in Z_5 x Z_2 via n = 2a + b.
A square is defined by (t, s) in Z_5 x Z_2 as
    L_{t,s}(i, j) = (t * i1 + j1 mod 5, s * i2 + j2 mod 2)
where (i1, i2) and (j1, j2) are the decompositions of i and j.

We generate all 10 such squares (5 values of t, 2 values of s), keep the distinct
valid Latin squares, and then repeatedly try different orderings to greedily build
a pairwise orthogonal set, keeping the largest one found.
"""

import time
import random

N = 10
SYMBOLS = list(range(N))


def _decompose(n: int):
    return (n // 2, n % 2)


def _compose(a: int, b: int) -> int:
    return 2 * a + b


def _build_square(t: int, s: int):
    sq = [[0] * N for _ in range(N)]
    for i in range(N):
        i1, i2 = _decompose(i)
        for j in range(N):
            j1, j2 = _decompose(j)
            a = (t * i1 + j1) % 5
            b = (s * i2 + j2) % 2
            sq[i][j] = _compose(a, b)
    return sq


def _valid_lsq(sq):
    for i in range(N):
        if sorted(sq[i]) != SYMBOLS:
            return False
    for j in range(N):
        col = [sq[i][j] for i in range(N)]
        if sorted(col) != SYMBOLS:
            return False
    return True


def _orthogonal(a, b):
    seen = set()
    for i in range(N):
        ra = a[i]
        rb = b[i]
        for j in range(N):
            p = (ra[j], rb[j])
            if p in seen:
                return False
            seen.add(p)
    return True


def _group_table():
    return [[(i + j) % N for j in range(N)] for i in range(N)]


def construct():
    """
    Return a list of up to 9 mutually orthogonal Latin squares of order 10.
    """
    start = time.time()
    deadline = start + 80.0  # safety margin under the 90 s kill

    # Generate all candidate squares from the Z_5 x Z_2 construction
    candidates = []
    for t in range(5):
        for s in range(2):
            sq = _build_square(t, s)
            if _valid_lsq(sq):
                candidates.append(sq)

    # Add the group table as an additional candidate
    candidates.append(_group_table())

    # Deduplicate
    unique_candidates = []
    seen = set()
    for sq in candidates:
        key = tuple(tuple(row) for row in sq)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(sq)

    # Greedy selection with multiple random orderings
    best_set = []
    best_count = 0
    rng = random.Random(20240517)

    for _ in range(50):
        order = list(range(len(unique_candidates)))
        rng.shuffle(order)

        current = []
        for idx in order:
            sq = unique_candidates[idx]
            ok = True
            for other in current:
                if not _orthogonal(sq, other):
                    ok = False
                    break
            if ok:
                current.append(sq)

        if len(current) > best_count:
            best_count = len(current)
            best_set = current[:]

        if best_count == 9:
            break

        if time.time() > deadline:
            break

    # Final validation
    out = []
    for sq in best_set:
        clean = [[int(x) for x in row] for row in sq]
        if not _valid_lsq(clean):
            continue
        ok = True
        for o in out:
            if not _orthogonal(o, clean):
                ok = False
                break
        if ok:
            out.append(clean)

    if not out:
        out = [_group_table()]

    return out
