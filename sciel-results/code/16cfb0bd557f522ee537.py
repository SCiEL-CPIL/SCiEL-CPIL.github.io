"""Matrix multiplication 6x6x6 tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using block-structured
    Strassen: 9 blocks of 2x2, each multiplied via Strassen's 7-product algorithm.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin for 150s limit

    n, m, p = DIMENSIONS  # all 6

    # The rank is 7 (Strassen) * 27 (number of 2x2 blocks in 6x6) = 189
    rank = 189
    # Initialize matrices with zeros
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Iterate over the 3x3 grid of 2x2 blocks
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                # Define row index helpers for U, V, W
                # U row index: (row_in_block * m) + col_in_block
                # Block row: i_block, Block col: k_block
                def u_row(a, b):
                    return (i_block * 2 + a) * m + (k_block * 2 + b)

                # V row index: (row_in_block * p) + col_in_block
                # Block row: k_block, Block col: j_block
                def v_row(c, d):
                    return (k_block * 2 + c) * p + (j_block * 2 + d)

                # W row index: (row_in_block * n) + col_in_block
                # Block row: j_block, Block col: i_block
                def w_row(e, f):
                    return (j_block * 2 + f) * n + (i_block * 2 + e)

                # Strassen's 7 multiplications for 2x2 blocks:
                # M1 = (A11 + A22) * (B11 + B22)
                U[u_row(0, 0)][col] = 1
                U[u_row(1, 1)][col] = 1
                V[v_row(0, 0)][col] = 1
                V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = 1
                W[w_row(1, 1)][col] = 1
                col += 1

                # M2 = (A21 + A22) * B11
                U[u_row(1, 0)][col] = 1
                U[u_row(1, 1)][col] = 1
                V[v_row(0, 0)][col] = 1
                W[w_row(1, 0)][col] = 1
                W[w_row(1, 1)][col] = -1
                col += 1

                # M3 = A11 * (B12 - B22)
                U[u_row(0, 0)][col] = 1
                V[v_row(0, 1)][col] = 1
                V[v_row(1, 1)][col] = -1
                W[w_row(0, 1)][col] = 1
                W[w_row(1, 1)][col] = 1
                col += 1

                # M4 = A22 * (B21 - B11)
                U[u_row(1, 1)][col] = 1
                V[v_row(1, 0)][col] = 1
                V[v_row(0, 0)][col] = -1
                W[w_row(0, 0)][col] = 1
                W[w_row(1, 0)][col] = 1
                col += 1

                # M5 = (A11 + A12) * B22
                U[u_row(0, 0)][col] = 1
                U[u_row(0, 1)][col] = 1
                V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = -1
                W[w_row(0, 1)][col] = 1
                col += 1

                # M6 = (A21 - A11) * (B11 + B12)
                U[u_row(1, 0)][col] = 1
                U[u_row(0, 0)][col] = -1
                V[v_row(0, 0)][col] = 1
                V[v_row(0, 1)][col] = 1
                W[w_row(1, 1)][col] = 1
                col += 1

                # M7 = (A12 - A22) * (B21 + B22)
                U[u_row(0, 1)][col] = 1
                U[u_row(1, 1)][col] = -1
                V[v_row(1, 0)][col] = 1
                V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = 1
                col += 1

    # Verify the decomposition on a few random points to ensure correctness
    # This is a sanity check. The construction is deterministic and known to be correct.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    random.seed(42)
    valid = True
    # Check a subset of indices to save time
    for _ in range(100):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        # Compute T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        val = 0.0
        for r in range(rank):
            val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
        # The value should be 1 if i==k (for the diagonal in the tensor sense? No, the tensor is 1 if the index matches the multiplication pattern)
        # Actually, the tensor T_{ijk} is 1 if the indices correspond to the multiplication.
        # In the standard basis, T_{(i,j),(j,k),(k,i)} = 1?
        # The convention is W[k*n+i, r].
        # The tensor element is U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # This should be 1 if the triplet (i,j,k) is part of the multiplication.
        # For matrix multiplication C = A B, C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor is T_{ijk} = delta_{i, i'} delta_{j, j'} delta_{k, k'} ?
        # No, the tensor is the structure tensor of matrix multiplication.
        # T_{(i,j), (j,k), (k,i)} = 1.
        # Here the indices are flattened.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # The value is 1 if this triplet is valid, which it always is in the sum?
        # No, the sum is over r. The value is 1 if the specific (i,j,k) is "connected".
        # Actually, for the matrix multiplication tensor, T_{ijk} = 1 for all i,j,k? No.
        # The tensor is sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}.
        # So T_{(i,j), (j,k), (k,i)} = 1.
        # In our flattened indices:
        # u_idx = i*m + j
        # v_idx = j*p + k
        # w_idx = k*n + i
        # The value should be 1 for ALL i, j, k?
        # Yes, because for any i, j, k, the term A_{ij} B_{jk} contributes to C_{ik}.
        # So the tensor component is 1 for all i, j, k.
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
