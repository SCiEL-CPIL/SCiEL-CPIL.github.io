import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Build a rank-28 decomposition using Strassen on 2x2 blocks + schoolbook for remaining.
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # 7 Strassen terms for 2x2x2 block
        # M1 = (A11+A22)(B11+B22)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A21+A22)B11
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A11(B21-B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A22(B22-B11)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A11+A12)B22
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def schoolbook_block(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block decomposition of 2x3x5:
    # Rows 0-1, Cols 0-1: 2x2x2 -> 7 Strassen terms (k=0,1)
    strassen_block(0, 1, 0, 1, 0, 1)
    # Rows 0-1, Cols 0-1, k=2: 2x2x1 -> 4 schoolbook terms
    schoolbook_block([0, 1], [0, 1], [2])
    # Rows 0-1, Col 2, k=0-1: 2x1x2 -> 4 schoolbook terms
    schoolbook_block([0, 1], [2], [0, 1])
    # Rows 0-1, Col 2, k=2: 2x1x1 -> 2 schoolbook terms
    schoolbook_block([0, 1], [2], [2])
    # Rows 0-1, Cols 0-1, k=3-4: 2x2x2 -> 7 Strassen terms
    strassen_block(0, 1, 0, 1, 3, 4)
    # Rows 0-1, Col 2, k=3-4: 2x1x2 -> 4 schoolbook terms
    schoolbook_block([0, 1], [2], [3, 4])

    # Total: 7 + 4 + 4 + 2 + 7 + 4 = 28
    assert col == 28, f"Expected 28 terms, got {col}"

    # Verify the decomposition
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]

    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback to rank-30 schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank from 28 to lower values
    # Strategy: Use the 28 terms as a basis and search for linear dependencies
    # that allow us to express the decomposition with fewer terms.
    
    # The 28 rank-1 tensors T_c = U_c ⊗ V_c ⊗ W_c span a subspace of R^30.
    # We want to find if there's a subset of fewer than 28 such tensors that still
    # can produce the all-ones tensor, OR if we can find a different decomposition.
    
    # Approach: Try to find a rank-r decomposition for r = 27, 26, ... down to some limit.
    # Use the structure: the problem has symmetries we can exploit.
    
    # For now, let's try a targeted search:
    # The 2x3x5 multiplication can be seen as 5 dot products of length 3.
    # Each dot product of length 3 has rank at most 3 (schoolbook) or 2 (Strassen-like for 1x1x3? No, that's just 3 terms).
    # Actually, 1x1x3 dot product requires rank 3.
    
    # Alternative: Think of it as A (2x3) times B (3x5).
    # C = A * B.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # ...
    
    # Let me try a different structural approach.
    # Split B into B_left (3x2) and B_right (3x3).
    # C_left = A * B_left (2x2), C_right = A * B_right (2x3).
    # Rank(2x2x2) <= 7, Rank(2x3x2) <= ?
    # For 2x3x2: this is two dot products of length 3, but they share the same A.
    # Actually 2x3x2 means A is 2x3, B is 3x2.
    # C[0,0] = sum_k A[0,k]B[k,0]
    # C[0,1] = sum_k A[0,k]B[k,1]
    # C[1,0] = sum_k A[1,k]B[k,0]
    # C[1,1] = sum_k A[1,k]B[k,1]
    
    # This is a 2x3x2 tensor. What's its rank?
    # It's like two parallel 1x3x1 dot products, but coupled through A.
    # Actually, we can write:
    # C = [[A0·B0, A0·B1], [A1·B0, A1·B1]]
    # where A0, A1 are the rows of A, B0, B1 are columns of B.
    
    # Using Strassen-like: for the 2x2 output, we need 7 terms if we treat it as 2x2x2 with "effective" 2x2 inputs.
    # But here the "inner dimension" is 3, not 2.
    
    # Let me just try to verify if rank 27 or lower is achievable by checking linear dependencies.
    
    # Construct the 30x28 matrix M where each column is one rank-1 tensor.
    M = np.zeros((n*m*p, rank))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i*m*p + j*p + k, c] = uc[i, j] * vc[j, k] * wc[k, i]
    
    # Check rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    
    # If rank_M < 28, there are linear dependencies among the 28 tensors.
    # But having dependencies doesn't directly give us a lower-rank decomposition of the SAME tensor.
    # We need to find a different set of rank-1 tensors.
    
    # For now, return the valid rank-28 solution.
    # A more sophisticated search for rank < 28 would require numerical optimization
    # or a combinatorial search, which is complex.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
