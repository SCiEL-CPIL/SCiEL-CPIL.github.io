import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a greedy set-cover approach over valid Strassen blocks to maximize covered entries,
    then fills remaining entries with schoolbook terms.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def strassen_block_entries(i1, i2, j1, j2, k1, k2):
        """Return the 7 Strassen terms for a 2x2x2 block."""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        terms = []
        # M1 = (a11+a22)*(b11+b22)
        terms.append(({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}))
        # M2 = (a21+a22)*b11
        terms.append(({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}))
        # M3 = a11*(b12-b22)
        terms.append(({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}))
        # M4 = a22*(b21-b11)
        terms.append(({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}))
        # M5 = (a11+a12)*b22
        terms.append(({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}))
        # M6 = (a21-a11)*(b11+b12)
        terms.append(({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}))
        # M7 = (a12-a22)*(b21+b22)
        terms.append(({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}))
        return terms

    def block_covers(i1, i2, j1, j2, k1, k2):
        """Return the set of (i,j,k) triples covered by this block."""
        return set((i, j, k) for i in (i1, i2) for j in (j1, j2) for k in (k1, k2))

    # Generate all possible Strassen blocks
    all_blocks = []
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p):
                        for k2 in range(k1 + 1, p):
                            covers = block_covers(i1, i2, j1, j2, k1, k2)
                            all_blocks.append((covers, (i1, i2, j1, j2, k1, k2)))

    # Greedy set cover: repeatedly pick the block that covers the most uncovered entries
    all_entries = set((i, j, k) for i in range(n) for j in range(m) for k in range(p))
    uncovered = set(all_entries)
    selected_blocks = []

    while uncovered:
        best_block = None
        best_count = 0
        for covers, params in all_blocks:
            new_covered = len(covers & uncovered)
            if new_covered > best_count:
                best_count = new_covered
                best_block = (covers, params)
            elif new_covered == best_count and best_block is not None:
                # Tie-break: prefer blocks that cover more "hard" entries
                pass

        if best_block is None or best_count == 0:
            break

        covers, params = best_block
        uncovered -= covers
        selected_blocks.append(params)

        if time.time() > deadline - 5.0:
            break

    # Add the selected Strassen blocks
    for params in selected_blocks:
        i1, i2, j1, j2, k1, k2 = params
        for u_e, v_e, w_e in strassen_block_entries(i1, i2, j1, j2, k1, k2):
            add_term(u_e, v_e, w_e)

    # Add schoolbook for remaining uncovered entries
    for i, j, k in sorted(uncovered):
        if time.time() > deadline - 2.0:
            # Add all remaining schoolbook terms quickly
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        if (i2, j2, k2) in uncovered:
                            add_term({i2 * m + j2: 1}, {j2 * p + k2: 1}, {k2 * n + i2: 1})
            uncovered = set()
            break
        add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
