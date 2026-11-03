import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 3, 5, 7

    # Generate all possible 2x2x2 Strassen blocks
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, i0 + 1, j0, j0 + 1, k0, k0 + 1))

    # Each block covers 8 (i,j,k) triples
    def block_triples(b):
        i0, i1, j0, j1, k0, k1 = b
        return [(i, j, k) for i in (i0, i1) for j in (j0, j1) for k in (k0, k1)]

    # Total tensor terms
    total_terms = n * m * p  # 105

    # Greedy: pick blocks that cover the most uncovered terms
    uncovered = set(itertools.product(range(n), range(m), range(p)))
    selected_blocks = []
    block_set = set(all_blocks)

    while uncovered and block_set:
        best_block = None
        best_count = 0
        for b in block_set:
            count = len(set(block_triples(b)) & uncovered)
            if count > best_count:
                best_count = count
                best_block = b
        if best_block is None or best_count == 0:
            break
        selected_blocks.append(best_block)
        uncovered -= set(block_triples(best_block))
        block_set.discard(best_block)

    # Build Strassen columns for selected blocks
    U_cols = []
    V_cols = []
    W_cols = []

    for i0, i1, j0, j1, k0, k1 in selected_blocks:
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1

        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1

        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1

        # 7 Strassen multiplications per 2x2x2 block
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})

        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})

        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})

        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})

        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})

        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})

        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})

    # Add schoolbook for remaining uncovered terms
    for i, j, k in uncovered:
        U_cols.append({i * m + j: 1})
        V_cols.append({j * p + k: 1})
        W_cols.append({k * n + i: 1})

    # Convert to dense matrices
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val

    # Verify decomposition
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_comp = np.einsum('ar,br,cr->abc', U, V, W)

    if not np.allclose(T, T_comp, atol=1e-10):
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})
        rank = len(U_cols)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            for idx, val in U_cols[c].items():
                U[idx, c] = val
            for idx, val in V_cols[c].items():
                V[idx, c] = val
            for idx, val in W_cols[c].items():
                W[idx, c] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
