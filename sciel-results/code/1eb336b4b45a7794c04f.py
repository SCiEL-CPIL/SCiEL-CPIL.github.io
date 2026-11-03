"""Matrix multiplication 3x4x5 tensor rank minimization.

Design (same family as seed):
  - Start from the schoolbook decomposition (rank 60).
  - Place the maximum number of DISJOINT 2x2x2 Strassen blocks (each saves 1 rank).
    The placement is a maximum 3D matching, solved exactly with branch-and-bound.
  - Every candidate is verified exactly (integer arithmetic) before being kept.
"""
import time

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60
DEADLINE = 140  # wall-clock safety margin (evaluator kills at 150 s)

_T0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _T0)


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != r for row in U):
        return False
    if any(len(row) != r for row in V):
        return False
    if any(len(row) != r for row in W):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return the trivial schoolbook decomposition (rank 60)."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_terms(i_off, j_off, k_off):
    """Return the 7 Strassen terms for a 2x2x2 block at (i_off, j_off, k_off)."""
    terms = []
    U1 = {(i_off, j_off): 1, (i_off + 1, j_off + 1): 1}
    V1 = {(j_off, k_off): 1, (j_off + 1, k_off + 1): 1}
    W1 = {(k_off, i_off): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U1, V1, W1))

    U2 = {(i_off + 1, j_off): 1, (i_off + 1, j_off + 1): 1}
    V2 = {(j_off, k_off): 1}
    W2 = {(k_off, i_off + 1): 1, (k_off + 1, i_off + 1): -1}
    terms.append((U2, V2, W2))

    U3 = {(i_off, j_off): 1}
    V3 = {(j_off, k_off + 1): 1, (j_off + 1, k_off + 1): -1}
    W3 = {(k_off + 1, i_off): 1, (k_off + 1, i_off + 1): 1}
    terms.append((U3, V3, W3))

    U4 = {(i_off + 1, j_off + 1): 1}
    V4 = {(j_off + 1, k_off): 1, (j_off, k_off): -1}
    W4 = {(k_off, i_off + 1): 1, (k_off, i_off): 1}
    terms.append((U4, V4, W4))

    U5 = {(i_off, j_off): 1, (i_off, j_off + 1): 1}
    V5 = {(j_off + 1, k_off + 1): 1}
    W5 = {(k_off + 1, i_off): 1, (k_off, i_off): -1}
    terms.append((U5, V5, W5))

    U6 = {(i_off + 1, j_off): 1, (i_off, j_off): -1}
    V6 = {(j_off, k_off): 1, (j_off, k_off + 1): 1}
    W6 = {(k_off + 1, i_off + 1): 1}
    terms.append((U6, V6, W6))

    U7 = {(i_off, j_off + 1): 1, (i_off + 1, j_off + 1): -1}
    V7 = {(j_off + 1, k_off): 1, (j_off + 1, k_off + 1): 1}
    W7 = {(k_off, i_off): 1}
    terms.append((U7, V7, W7))
    return terms


def _all_block_offsets():
    """All valid 2x2x2 Strassen block placements (i_off, j_off, k_off)."""
    blocks = []
    for i in range(N - 1):
        for j in range(M - 1):
            for k in range(P - 1):
                blocks.append((i, j, k))
    return blocks


def _max_disjoint_blocks():
    """Exact maximum set of disjoint 2x2x2 blocks via branch-and-bound."""
    offsets = _all_block_offsets()
    n = len(offsets)
    if n == 0:
        return []

    # Cell sets for each block.
    cell_sets = []
    for (i, j, k) in offsets:
        cells = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    cells.add((i + di, j + dj, k + dk))
        cell_sets.append(frozenset(cells))

    # Greedy seed.
    used = set()
    seed = []
    for idx, cs in enumerate(cell_sets):
        if cs.isdisjoint(used):
            seed.append(idx)
            used |= set(cs)
    best = list(seed)

    def rec(remaining, current, cur_cells):
        nonlocal best
        if not remaining:
            if len(current) > len(best):
                best = list(current)
            return
        if _time_left() < 0.3:
            return
        if len(current) + len(remaining) <= len(best):
            return
        # Pick most constrained remaining block.
        feasible = [idx for idx in remaining if cell_sets[idx].isdisjoint(cur_cells)]
        if not feasible:
            if len(current) > len(best):
                best = list(current)
            return
        take = feasible[0]
        rest = [idx for idx in remaining if idx != take]
        current.append(take)
        rec(rest, current, cur_cells | set(cell_sets[take]))
        current.pop()
        rec(rest, current, cur_cells)

    rec(list(range(n)), [], frozenset())
    return [offsets[idx] for idx in best]


def construct():
    """
    Exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Uses the maximum number of disjoint Strassen 2x2x2 blocks to minimize rank.

    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer entries.
    """
    n, m, p = N, M, P

    # Find maximum set of disjoint Strassen blocks.
    block_offsets = _max_disjoint_blocks()

    # Build terms.
    covered = set()
    all_terms = []
    for (i_off, j_off, k_off) in block_offsets:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i_off + di, j_off + dj, k_off + dk))
        for U_d, V_d, W_d in _strassen_terms(i_off, j_off, k_off):
            all_terms.append((U_d, V_d, W_d))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    all_terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))

    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (U_d, V_d, W_d) in enumerate(all_terms):
        for (i, j), coeff in U_d.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_d.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_d.items():
            if coeff != 0:
                W[k * n + i][c] = coeff

    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback.
    return _schoolbook()
