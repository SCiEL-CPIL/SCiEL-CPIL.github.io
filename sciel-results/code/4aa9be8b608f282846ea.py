"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Standard Strassen decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) where indices are:
        U: [i*2+j, c], V: [j*2+k, c], W: [k*2+i, c]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # C00 += M1, C11 += M1

        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # C10 += M2, C11 -= M2

        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # C01 += M3, C11 += M3

        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        # C00 += M4, C10 += M4

        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # C00 -= M5, C01 += M5

        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # C11 += M6

        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        # C00 += M7

        return U, V, W

    def verify_strassen_block():
        """Verify that the Strassen block correctly decomposes 2x2x2."""
        U_s, V_s, W_s = strassen_2x2x2()
        U_r = U_s.reshape(2, 2, 7)
        V_r = V_s.reshape(2, 2, 7)
        W_r = W_s.reshape(2, 2, 7)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((2, 2, 2)), atol=1e-9, rtol=0)

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    def build_decomposition(i_blocks, j_blocks, k_blocks):
        """Build decomposition from block structure."""
        U_cols = []
        V_cols = []
        W_cols = []
        
        for i_blk in i_blocks:
            for j_blk in j_blocks:
                for k_blk in k_blocks:
                    ni = len(i_blk)
                    nj = len(j_blk)
                    nk = len(k_blk)
                    
                    if ni == 1 and nj == 1 and nk == 1:
                        U_col = [0.0] * (n * m)
                        V_col = [0.0] * (m * p)
                        W_col = [0.0] * (p * n)
                        U_col[i_blk[0]*m + j_blk[0]] = 1.0
                        V_col[j_blk[0]*p + k_blk[0]] = 1.0
                        W_col[k_blk[0]*n + i_blk[0]] = 1.0
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                    elif ni == 2 and nj == 2 and nk == 2:
                        U_s, V_s, W_s = strassen_2x2x2()
                        for c in range(7):
                            U_col = [0.0] * (n * m)
                            V_col = [0.0] * (m * p)
                            W_col = [0.0] * (p * n)
                            for ii in range(2):
                                for jj in range(2):
                                    U_col[i_blk[ii]*m + j_blk[jj]] = U_s[ii*2+jj][c]
                            for jj in range(2):
                                for kk in range(2):
                                    V_col[j_blk[jj]*p + k_blk[kk]] = V_s[jj*2+kk][c]
                            for kk in range(2):
                                for ii in range(2):
                                    W_col[k_blk[kk]*n + i_blk[ii]] = W_s[kk*2+ii][c]
                            U_cols.append(U_col)
                            V_cols.append(V_col)
                            W_cols.append(W_col)
                    elif ni == 2 and nj == 2 and nk == 1:
                        U_col = [0.0] * (n * m)
                        V_col = [0.0] * (m * p)
                        W_col = [0.0] * (p * n)
                        for ii in range(2):
                            for jj in range(2):
                                U_col[i_blk[ii]*m + j_blk[jj]] = 1.0
                        for jj in range(2):
                            V_col[j_blk[jj]*p + k_blk[0]] = 1.0
                        for ii in range(2):
                            W_col[k_blk[0]*n + i_blk[ii]] = 1.0
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                    elif ni == 1 and nj == 2 and nk == 1:
                        U_col = [0.0] * (n * m)
                        V_col = [0.0] * (m * p)
                        W_col = [0.0] * (p * n)
                        U_col[i_blk[0]*m + j_blk[0]] = 1.0
                        U_col[i_blk[0]*m + j_blk[1]] = 1.0
                        V_col[j_blk[0]*p + k_blk[0]] = 1.0
                        V_col[j_blk[1]*p + k_blk[0]] = 1.0
                        W_col[k_blk[0]*n + i_blk[0]] = 1.0
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                    elif ni == 2 and nj == 1 and nk == 1:
                        U_col = [0.0] * (n * m)
                        V_col = [0.0] * (m * p)
                        W_col = [0.0] * (p * n)
                        U_col[i_blk[0]*m + j_blk[0]] = 1.0
                        U_col[i_blk[1]*m + j_blk[0]] = 1.0
                        V_col[j_blk[0]*p + k_blk[0]] = 1.0
                        W_col[k_blk[0]*n + i_blk[0]] = 1.0
                        W_col[k_blk[0]*n + i_blk[1]] = 1.0
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                    else:
                        for ii in range(ni):
                            for jj in range(nj):
                                for kk in range(nk):
                                    U_col = [0.0] * (n * m)
                                    V_col = [0.0] * (m * p)
                                    W_col = [0.0] * (p * n)
                                    U_col[i_blk[ii]*m + j_blk[jj]] = 1.0
                                    V_col[j_blk[jj]*p + k_blk[kk]] = 1.0
                                    W_col[k_blk[kk]*n + i_blk[ii]] = 1.0
                                    U_cols.append(U_col)
                                    V_cols.append(V_col)
                                    W_cols.append(W_col)
        
        if not U_cols:
            return None, 0
        
        rank = len(U_cols)
        U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
        V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
        W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
        
        return {"U": U_mat, "V": V_mat, "W": W_mat}, rank

    def build_hierarchical():
        """Build decomposition using hierarchical Strassen approach.
        Split 3x4x5 into:
        - 2x4x5 part (rows 0,1): split m=4 into 2+2, each 2x2x5
          - 2x2x5: split p=5 into 2+2+1, each 2x2x2 (Strassen) or 2x2x1 (schoolbook)
          - 2x2x2: Strassen rank 7
          - 2x2x1: schoolbook rank 4
          - So 2x2x5 = 7+7+4 = 18
          - 2x4x5 = 18+18 = 36
        - 1x4x5 part (row 2): split m=4 into 2+2, each 1x2x5
          - 1x2x5: split p=5 into 2+2+1
          - 1x2x2: schoolbook rank 4
          - 1x2x1: schoolbook rank 2
          - So 1x2x5 = 4+4+2 = 10
          - 1x4x5 = 10+10 = 20
        Total: 36+20 = 56
        """
        U_cols = []
        V_cols = []
        W_cols = []
        
        U_s, V_s, W_s = strassen_2x2x2()
        
        # Part 1: 2x4x5 (i=0,1; j=0,1,2,3; k=0,1,2,3,4)
        # Split j into [0,1] and [2,3]
        for j_off in [0, 2]:
            # 2x2x5 block at j_off
            # Split k into [0,1], [2,3], [4]
            for k_off, nk in [(0, 2), (2, 2), (4, 1)]:
                for i in [0, 1]:
                    for j in [j_off, j_off+1]:
                        for k in range(k_off, k_off+nk):
                            if nk == 2:
                                # Use Strassen for 2x2x2
                                pass  # Handled below
                            else:
                                # 2x2x1: schoolbook
                                U_col = [0.0] * (n * m)
                                V_col = [0.0] * (m * p)
                                W_col = [0.0] * (p * n)
                                U_col[i*m + j] = 1.0
                                V_col[j*p + k] = 1.0
                                W_col[k*n + i] = 1.0
                                U_cols.append(U_col)
                                V_cols.append(V_col)
                                W_cols.append(W_col)
        
        # Rebuild more carefully
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Part 1: 2x4x5 (i=0,1)
        for j_off in [0, 2]:
            # 2x2x5 block
            # k=0,1: Strassen 2x2x2
            for c in range(7):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                for ii in range(2):
                    for jj in range(2):
                        U_col[ii*m + (j_off+jj)] = U_s[ii*2+jj][c]
                for jj in range(2):
                    for kk in range(2):
                        V_col[(j_off+jj)*p + kk] = V_s[jj*2+kk][c]
                for kk in range(2):
                    for ii in range(2):
                        W_col[kk*n + ii] = W_s[kk*2+ii][c]
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)
            
            # k=2,3: Strassen 2x2x2
            for c in range(7):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                for ii in range(2):
                    for jj in range(2):
                        U_col[ii*m + (j_off+jj)] = U_s[ii*2+jj][c]
                for jj in range(2):
                    for kk in range(2):
                        V_col[(j_off+jj)*p + (2+kk)] = V_s[jj*2+kk][c]
                for kk in range(2):
                    for ii in range(2):
                        W_col[(2+kk)*n + ii] = W_s[kk*2+ii][c]
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)
            
            # k=4: 2x2x1 schoolbook (rank 4)
            for ii in range(2):
                for jj in range(2):
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[ii*m + (j_off+jj)] = 1.0
                    V_col[(j_off+jj)*p + 4] = 1.0
                    W_col[4*n + ii] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
        
        # Part 2: 1x4x5 (i=2)
        for j_off in [0, 2]:
            # 1x2x5 block
            # k=0,1: 1x2x2 schoolbook (rank 4)
            for jj in range(2):
                for kk in range(2):
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[2*m + (j_off+jj)] = 1.0
                    V_col[(j_off+jj)*p + kk] = 1.0
                    W_col[kk*n + 2] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
            
            # k=2,3: 1x2x2 schoolbook (rank 4)
            for jj in range(2):
                for kk in range(2):
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[2*m + (j_off+jj)] = 1.0
                    V_col[(j_off+jj)*p + (2+kk)] = 1.0
                    W_col[(2+kk)*n + 2] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
            
            # k=4: 1x2x1 schoolbook (rank 2)
            for jj in range(2):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[2*m + (j_off+jj)] = 1.0
                V_col[(j_off+jj)*p + 4] = 1.0
                W_col[4*n + 2] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)
        
        rank = len(U_cols)
        U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
        V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
        W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
        
        return {"U": U_mat, "V": V_mat, "W": W_mat}, rank

    best_result = schoolbook()
    best_rank = n * m * p

    # First, verify our Strassen block is correct
    if not verify_strassen_block():
        return best_result

    # Try hierarchical approach (should give rank 56)
    if time.time() < deadline:
        result, rank = build_hierarchical()
        if result and verify_fast(result["U"], result["V"], result["W"], n, m, p):
            if rank < best_rank:
                best_result = result
                best_rank = rank

    # Try block structure approaches
    if time.time() < deadline:
        i_blocks_1 = [[0,1], [2]]
        j_blocks_1 = [[0,1], [2,3]]
        k_blocks_1 = [[0,1], [2,3], [4]]
        
        result, rank = build_decomposition(i_blocks_1, j_blocks_1, k_blocks_1)
        if result and verify_fast(result["U"], result["V"], result["W"], n, m, p):
            if rank < best_rank:
                best_result = result
                best_rank = rank

    if time.time() < deadline:
        i_blocks_2 = [[0,1], [2]]
        j_blocks_2 = [[0,1], [2,3]]
        k_blocks_2 = [[0,1,2], [3,4]]
        
        result, rank = build_decomposition(i_blocks_2, j_blocks_2, k_blocks_2)
        if result and verify_fast(result["U"], result["V"], result["W"], n, m, p):
            if rank < best_rank:
                best_result = result
                best_rank = rank

    if time.time() < deadline:
        i_blocks_3 = [[0], [1,2]]
        j_blocks_3 = [[0,1], [2,3]]
        k_blocks_3 = [[0,1], [2,3], [4]]
        
        result, rank = build_decomposition(i_blocks_3, j_blocks_3, k_blocks_3)
        if result and verify_fast(result["U"], result["V"], result["W"], n, m, p):
            if rank < best_rank:
                best_result = result
                best_rank = rank

    if time.time() < deadline:
        i_blocks_4 = [[0,1], [2]]
        j_blocks_4 = [[0], [1,2], [3]]
        k_blocks_4 = [[0,1], [2,3], [4]]
        
        result, rank = build_decomposition(i_blocks_4, j_blocks_4, k_blocks_4)
        if result and verify_fast(result["U"], result["V"], result["W"], n, m, p):
            if rank < best_rank:
                best_result = result
                best_rank = rank

    return best_result
# EVOLVE-BLOCK-END
