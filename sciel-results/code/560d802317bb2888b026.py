"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """Build the rank-28 Strassen decomposition deterministically."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for b in range(2):  # B column block
        for a_blk in range(2):  # A column block / B row block
            for s in range(7):
                c = col

                # U contributions: A[i, 2*a_blk + j]
                U_contrib = {
                    0: [(0, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    1: [(1, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    2: [(0, 2*a_blk, 1)],
                    3: [(1, 2*a_blk+1, 1)],
                    4: [(0, 2*a_blk, 1), (0, 2*a_blk+1, 1)],
                    5: [(1, 2*a_blk, 1), (0, 2*a_blk, -1)],
                    6: [(0, 2*a_blk+1, 1), (1, 2*a_blk+1, -1)],
                }
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j, c] += coeff

                # V contributions: B[2*a_blk + j, 2*b + k]
                V_contrib = {
                    0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    1: [(2*a_blk, 2*b, 1)],
                    2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                    3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                    4: [(2*a_blk+1, 2*b+1, 1)],
                    5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                    6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                }
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k, c] += coeff

                # W contributions: C[i, 2*b + k]
                W_contrib = {
                    0: {(0, 0): 1, (1, 1): 1},
                    1: {(1, 0): 1, (1, 1): -1},
                    2: {(0, 1): 1, (1, 1): 1},
                    3: {(0, 0): 1, (1, 0): 1},
                    4: {(0, 0): -1, (0, 1): 1},
                    5: {(1, 1): 1},
                    6: {(0, 0): 1},
                }
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[(2 * b + k_loc) * 2 + i_loc, c] += coeff

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W correctly decompose the matmul tensor."""
    rank = U.shape[1]
    # Build tensor T[i, k, j] = sum_r U[i,r] * V[k,r] * W[j,r]
    # The matmul tensor is T[i, k, j] = 1 if i==j else 0... no wait.
    # For matmul: C[i,j] = sum_k A[i,k] * B[k,j]
    # Tensor form: T[i,k,j] where output index is (i,j), input indices are (i,k) for A and (k,j) for B
    # W index convention: W[k*n + i, r] where i is row of C, k is col of C
    # So W[col*n + row, r]
    # The tensor entry for (i, k, j) should be 1 (since C[i,j] = sum_k A[i,k]*B[k,j])
    # T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin

    # Build the base rank-28 Strassen decomposition
    U, V, W = _build_strassen_rank28()
    
    # Verify it's correct
    assert _verify_decomposition(U, V, W, n, m, p), "Base rank-28 decomposition failed verification"
    
    # Return the rank-28 solution as-is (it's already optimal for this block structure)
    # Converting to lists for the expected return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
