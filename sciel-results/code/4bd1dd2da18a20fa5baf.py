"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Lader's 23-rank algorithm for 3x3x3 matmul
    # Each term r has U_r[i,j], V_r[j,k], W_r[k,i]
    # We need sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1 for all i,j,k in {0,1,2}
    
    # Lader's 23 terms (from literature):
    # Let me use a known rank-23 decomposition.
    # 
    # Actually, let me use the following approach:
    # 3x3x3 = 2x2x2 (Strassen, rank 7) + remaining
    # 
    # Split i={0,1}, j={0,1}, k={0,1}: Strassen rank 7
    # Remaining:
    #   i={0,1}, j=2, k={0,1,2}: rank 6
    #   i=2, j={0,1}, k={0,1,2}: rank 6
    #   i=2, j=2, k={0,1,2}: rank 3
    #   i={0,1}, j={0,1}, k=2: rank 4
    #   i=2, j={0,1}, k=2: rank 2
    #   i={0,1}, j=2, k=2: rank 2
    #   i=2, j=2, k=2: rank 1
    # Total: 7+6+6+3+4+2+2+1 = 31. Not better than 23.
    
    # Let me use a different known rank-23 algorithm.
    # 
    # From Lader (1976), the 23-rank algorithm for 3x3x3:
    # I'll use a formulation where we define 23 scalar multiplications.
    
    # Alternative: use the fact that 3x3x3 has rank 23 and use a known list.
    # 
    # Let me try a constructive approach using block decomposition with
    # overlapping Strassen-like tricks.
    
    # For now, let's use a safe 3x3x3 decomposition with rank 27 (schoolbook)
    # and see if we can improve.
    
    # Actually, let me just use the 3x3x4 = 3x3x3 + 3x3x1 approach
    # with 3x3x3 at rank 27 (schoolbook) for now, giving 27+9=36.
    # That's worse than 34.
    
    # Let me try: 3x3x4 as two 3x3x2 blocks.
    # Each 3x3x2: split into 2x2x2 (Strassen 7) + remaining
    #   i={0,1}, j={0,1}, k={0,1}: Strassen 7
    #   i={0,1}, j=2, k={0,1}: rank 4
    #   i=2, j={0,1}, k={0,1}: rank 4
    #   i=2, j=2, k={0,1}: rank 2
    #   i={0,1}, j={0,1}, k=1: wait, k is already in {0,1}
    # 
    # 3x3x2:
    #   i={0,1}, j={0,1}, k={0,1}: Strassen 7
    #   i={0,1}, j=2, k={0,1}: rank 4
    #   i=2, j={0,1}, k={0,1}: rank 4
    #   i=2, j=2, k={0,1}: rank 2
    # Total: 17
    # Two 3x3x2 blocks: 34. Same as before.
    
    # Let me try a 2x3x3 block with better decomposition.
    # 2x3x3: i={0,1}, j={0,1,2}, k={0,1,2}
    # 
    # Split j={0,1} and j=2:
    #   i={0,1}, j={0,1}, k={0,1,2}: 2x2x3
    #     Split k={0,1} and k=2:
    #       i={0,1}, j={0,1}, k={0,1}: Strassen 7
    #       i={0,1}, j={0,1}, k=2: rank 4
    #     Subtotal: 11
    #   i={0,1}, j=2, k={0,1,2}: rank 6
    # Total for 2x3x3: 17
    # 
    # 3x3x4 = 2x3x4 + 1x3x4
    # 2x3x4 = 2x3x3 + 2x3x1 = 17 + 6 = 23
    # 1x3x4 = 1x3x3 + 1x3x1 = 9 + 3 = 12
    # Total: 23 + 12 = 35. Worse.
    
    # Let me try yet another split:
    # 3x3x4 = 3x2x4 + 3x1x4
    # 3x2x4: i={0,1,2}, j={0,1}, k={0,1,2,3}
    #   Split i={0,1} and i=2:
    #     i={0,1}, j={0,1}, k={0,1,2,3}: 2x2x4
    #       Split k={0,1} and k={2,3}:
    #         i={0,1}, j={0,1}, k={0,1}: Strassen 7
    #         i={0,1}, j={0,1}, k={2,3}: Strassen 7
    #       Subtotal: 14
    #     i=2, j={0,1}, k={0,1,2,3}: rank 8
    #   Total for 3x2x4: 22
    # 3x1x4: i={0,1,2}, j=2, k={0,1,2,3}: rank 12
    # Total: 22 + 12 = 34. Same.
    
    # OK, let me try to find a better 2x2x4 decomposition.
    # 2x2x4: i={0,1}, j={0,1}, k={0,1,2,3}
    # 
    # Can we do better than 14?
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # 
    # Consider: M1 = (A00+A11)(B00+B11+B22+B33) -- but B is 2x4, so B[j,k] for j={0,1}, k={0,1,2,3}
    # 
    # Let me define:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00+B10)
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)(B11+B01)
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # These give the 2x2x2 part for k={0,1}.
    # 
    # For k={2,3}, we need another 7 terms. Total 14.
    # 
    # Can we share some terms? If we use:
    # M1' = (A00+A11)(B00+B11+B22+B33)
    # This would contribute to both k-blocks.
    # 
    # Let me try:
    # M1 = (A00+A11)(B00+B11+B22+B33)
    # M2 = (A10+A11)(B00+B10+B22+B32)
    # M3 = A00(B10-B11+B22-B23)
    # M4 = A11(B01-B00+B23-B22)
    # M5 = (A00+A01)(B11+B01+B33+B23)
    # M6 = (A10-A00)(B00+B10+B22+B32)
    # M7 = (A01-A11)(B01+B11+B23+B33)
    # 
    # This is getting complicated. Let me just verify the current 34-rank solution
    # and try to find a 33-rank solution by optimization.
    
    # For now, return the known 34-rank solution.
    # 
    # Strassen 2x2x2 terms
    strassen_terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,1],[1,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        ([[1,0],[0,0]], [[0,0],[1,-1]], [[0,1],[0,0]]),
        ([[0,0],[0,1]], [[-1,1],[0,0]], [[1,-1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[1,0],[0,0]]),
        ([[ -1,0],[1,0]], [[1,0],[1,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,1],[0,1]], [[1,0],[1,0]]),
    ]
    
    # Verify Strassen
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for (Ur, Vr, Wr) in strassen_terms:
                    s += Ur[i][j] * Vr[j][k] * Wr[k][i]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build 3x3x4 with rank 34
    total_rank = 34
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        for (Ur, Vr, Wr) in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    if Ur[ii][jj] != 0:
                        U[(i_off+ii)*m + (j_off+jj)][col] += Ur[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if Vr[jj][kk] != 0:
                        V[(j_off+jj)*p + (k_off+kk)][col] += Vr[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if Wr[kk][ii] != 0:
                        W[(k_off+kk)*n + (i_off+ii)][col] += Wr[kk][ii]
            col += 1
    
    def add_schoolbook_block(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_schoolbook_block([0,1], [2], [0,1])
    add_schoolbook_block([0,1], [2], [2,3])
    add_schoolbook_block([2], [0,1], [0,1])
    add_schoolbook_block([2], [0,1], [2,3])
    add_schoolbook_block([2], [2], [0,1])
    add_schoolbook_block([2], [2], [2,3])
    
    if time.time() - t0 < DEADLINE:
        if not _validate(U, V, W, n, m, p):
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
