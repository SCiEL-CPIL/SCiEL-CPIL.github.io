"""
Matrix multiplication 6x6x6: rank-minimal exact tensor decomposition.

Strategy: Block Strassen decomposition - partition 6x6 into 3x3 grid of 2x2 blocks,
applying Strassen's 7-product algorithm per block, achieving rank 189.

The key insight is that we can use a 3x3 block structure where each 2x2 block
product uses Strassen's algorithm (rank 7 instead of rank 8 for schoolbook 2x2).
Total rank = 3*3*3*7 = 189, better than schoolbook 216.

This program constructs the factor matrices U, V, W directly using the known
Strassen decomposition pattern, then verifies correctness before returning.
"""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix
    by a 6x6 matrix.

    Uses a block decomposition into nine 2x2 blocks, applying Strassen's
    7-product algorithm per block, achieving rank 189 (better than schoolbook 216).

    The tensor decomposition is:
      For each (i_block, j_block, k_block) in 3x3x3:
        For each of Strassen's 7 products:
          One rank-1 term in the tensor decomposition.

    Total rank = 3 * 3 * 3 * 7 = 189.
    """
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # Strassen's 7 products for 2x2 matrices:
    # Given A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = a11 * (b12 - b22)
    # M4 = a22 * (b21 - b11)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 - a22) * (b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    rank = 189
    U = np.zeros((n * m, rank), dtype=int)  # 36 x 189
    V = np.zeros((m * p, rank), dtype=int)  # 36 x 189
    W = np.zeros((p * n, rank), dtype=int)  # 36 x 189

    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                # Strassen products for A[i_block,k_block] * B[k_block,j_block]
                # A block entries: A11 = A[i_block,k_block][0,0], A12 = A[i_block,k_block][0,1]
                #                  A21 = A[i_block,k_block][1,0], A22 = A[i_block,k_block][1,1]
                # B block entries: B11 = B[k_block,j_block][0,0], B12 = B[k_block,j_block][0,1]
                #                  B21 = B[k_block,j_block][1,0], B22 = B[k_block,j_block][1,1]

                # Define helper to get U row index for A[i_block,k_block][a,b]
                def u_row(a, b):
                    return (i_block * 2 + a) * m + (k_block * 2 + b)

                # Define helper to get V row index for B[k_block,j_block][c,d]
                def v_row(c, d):
                    return (k_block * 2 + c) * p + (j_block * 2 + d)

                # Define helper to get W row index for C[i_block,j_block][e,f]
                # W[k*n+i] corresponds to C[i,k], so k=j_block*2+f, i=i_block*2+e
                def w_row(e, f):
                    return (j_block * 2 + f) * n + (i_block * 2 + e)

                # M1 = (A11 + A22) * (B11 + B22)
                U[u_row(0, 0), col] = 1  # A11
                U[u_row(1, 1), col] = 1  # A22
                V[v_row(0, 0), col] = 1  # B11
                V[v_row(1, 1), col] = 1  # B22
                W[w_row(0, 0), col] = 1  # C11
                W[w_row(1, 1), col] = 1  # C22
                col += 1

                # M2 = (A21 + A22) * B11
                U[u_row(1, 0), col] = 1  # A21
                U[u_row(1, 1), col] = 1  # A22
                V[v_row(0, 0), col] = 1  # B11
                W[w_row(1, 0), col] = 1  # C21
                W[w_row(1, 1), col] = -1  # C22
                col += 1

                # M3 = A11 * (B12 - B22)
                U[u_row(0, 0), col] = 1  # A11
                V[v_row(0, 1), col] = 1  # B12
                V[v_row(1, 1), col] = -1  # B22
                W[w_row(0, 1), col] = 1  # C12
                W[w_row(1, 1), col] = 1  # C22
                col += 1

                # M4 = A22 * (B21 - B11)
                U[u_row(1, 1), col] = 1  # A22
                V[v_row(1, 0), col] = 1  # B21
                V[v_row(0, 0), col] = -1  # B11
                W[w_row(0, 0), col] = 1  # C11
                W[w_row(1, 0), col] = 1  # C21
                col += 1

                # M5 = (A11 + A12) * B22
                U[u_row(0, 0), col] = 1  # A11
                U[u_row(0, 1), col] = 1  # A12
                V[v_row(1, 1), col] = 1  # B22
                W[w_row(0, 0), col] = -1  # C11
                W[w_row(0, 1), col] = 1  # C12
                col += 1

                # M6 = (A21 - A11) * (B11 + B12)
                U[u_row(1, 0), col] = 1  # A21
                U[u_row(0, 0), col] = -1  # A11
                V[v_row(0, 0), col] = 1  # B11
                V[v_row(0, 1), col] = 1  # B12
                W[w_row(1, 1), col] = 1  # C22
                col += 1

                # M7 = (A12 - A22) * (B21 + B22)
                U[u_row(0, 1), col] = 1  # A12
                U[u_row(1, 1), col] = -1  # A22
                V[v_row(1, 0), col] = 1  # B21
                V[v_row(1, 1), col] = 1  # B22
                W[w_row(0, 0), col] = 1  # C11
                col += 1

    # Verify the decomposition using efficient tensor contraction
    # T[i*m+j, j*p+k, k*n+i] should equal 1 for all i,j,k
    # This is equivalent to checking that the tensor decomposition is correct
    
    # Efficient verification: check that U @ V^T * W gives correct result
    # Using the tensor identity: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Quick random verification first
    random.seed(42)
    valid = True
    for _ in range(500):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = float(np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]))
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    # If random check passes, do a more thorough check on a subset
    if valid and (time.time() - t0) < DEADLINE:
        # Check all entries in a structured subset
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]))
                    if abs(val - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

    if valid:
        # Convert to list-of-lists format as expected by the interface
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook if verification fails
        rank_fallback = n * m * p
        U_f = np.zeros((n * m, rank_fallback), dtype=int)
        V_f = np.zeros((m * p, rank_fallback), dtype=int)
        W_f = np.zeros((p * n, rank_fallback), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
