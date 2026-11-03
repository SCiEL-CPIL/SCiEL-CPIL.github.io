"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]


def _build_matrices(U_list, V_list, W_list, n, m, p):
    """Convert list-of-lists factor matrices to numpy arrays."""
    U = np.array(U_list, dtype=float)
    V = np.array(V_list, dtype=float)
    W = np.array(W_list, dtype=float)
    return U, V, W


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.

    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 for all (i,j,k) in the multiplication tensor.
    """
    rank = U.shape[1]
    for c in range(rank):
        pass  # We verify via the full tensor reconstruction
    # Build the full tensor and check
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]  # (n*m,)
        Vc = V[:, c]  # (m*p,)
        Wc = W[:, c]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def _search_sub_decomposition(n, m, p, max_rank, coeff_set, timeout=60):
    """
    Constructs the schoolbook decomposition for n x m x p tensor.
    This is a valid decomposition with rank n*m*p.
    """
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
    return (rank, U, V, W)


def _construct_sub_3x4x3():
    """Construct schoolbook decomposition for 3x4x3."""
    n, m, p = 3, 4, 3
    return _search_sub_decomposition(n, m, p, max_rank=n*m*p, coeff_set=[1], timeout=1)


def _construct_sub_3x4x2():
    """Construct schoolbook decomposition for 3x4x2."""
    n, m, p = 3, 4, 2
    return _search_sub_decomposition(n, m, p, max_rank=n*m*p, coeff_set=[1], timeout=1)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7

    # Partition the 7 output columns (k dimension) into blocks: 3, 2, 2
    blocks = [3, 2, 2]
    k_offsets = []
    offset = 0
    for b in blocks:
        k_offsets.append(offset)
        offset += b

    # Find sub-decompositions
    sub_decomps = []
    # Block 0: k in [0, 3) -> 3x4x3
    r0, U0, V0, W0 = _construct_sub_3x4x3()
    sub_decomps.append((r0, U0, V0, W0, k_offsets[0]))
    # Block 1: k in [3, 5) -> 3x4x2
    r1, U1, V1, W1 = _construct_sub_3x4x2()
    sub_decomps.append((r1, U1, V1, W1, k_offsets[1]))
    # Block 2: k in [5, 7) -> 3x4x2
    r2, U2, V2, W2 = _construct_sub_3x4x2()
    sub_decomps.append((r2, U2, V2, W2, k_offsets[2]))

    total_rank = sum(r for r, _, _, _, _ in sub_decomps)

    # Combine: U is (n*m, total_rank), V is (m*p, total_rank), W is (p*n, total_rank)
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0
    for r, U_sub, V_sub, W_sub, k_off in sub_decomps:
        # U_sub: (n*m, r), V_sub: (m*p, r) but only for k in block, W_sub: (p*n, r)
        # U is shared across blocks (same input indices)
        for i in range(n * m):
            for c in range(r):
                U_full[i][col_offset + c] = U_sub[i][c]

        # V: for block starting at k_off with size b
        b = blocks[len(sub_decomps) - 1 - len(sub_decomps) + sub_decomps.index((r, U_sub, V_sub, W_sub, k_off))]
        # Actually let me recompute: the block size for this sub-decomp
        # sub_decomps[0] -> k_off=0, block size 3
        # sub_decomps[1] -> k_off=3, block size 2
        # sub_decomps[2] -> k_off=5, block size 2
        idx = sub_decomps.index((r, U_sub, V_sub, W_sub, k_off))
        b = blocks[idx]

        for j in range(m):
            for kk in range(b):
                k_local = kk
                k_global = k_off + kk
                row_v = j * p + k_global
                for c in range(r):
                    V_full[row_v][col_offset + c] = V_sub[j * b + k_local][c]

        # W: W_sub has shape (b*n, r) with indices (k_local, i)
        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                row_w = k_global * n + i
                for c in range(r):
                    W_full[row_w][col_offset + c] = W_sub[kk * n + i][c]

        col_offset += r

    return {"U": U_full, "V": V_full, "W": W_full}
