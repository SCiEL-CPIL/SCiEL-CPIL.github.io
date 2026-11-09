import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # For a 2x2x2 block, the 7 rank-1 terms are:
    # Each term: U_vec (4 entries for A), V_vec (4 entries for B), W_vec (4 entries for C)
    # A indices: [a11, a12, a21, a22] -> row indices in U
    # B indices: [b11, b12, b21, b22] -> row indices in V  
    # C indices: [c11, c12, c21, c22] -> row indices in W
    
    # Strassen terms (U_coeff, V_coeff, W_coeff)
    # M1 = (a11+a22)(b11+b22), contributes to c11(+1), c22(-1)
    # M2 = (a21+a22)b11, contributes to c21(+1), c22(-1)
    # M3 = a11(b12-b22), contributes to c12(+1), c22(+1)
    # M4 = a22(b21-b11), contributes to c11(+1), c21(+1)
    # M5 = (a11+a12)b22, contributes to c11(-1), c12(+1)
    # M6 = (a21-a11)(b11+b12), contributes to c22(+1)
    # M7 = (a11+a12)(b21+b22) -- wait, let me recheck
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (for A: [a11, a12, a21, a22]):
    U_strassen = [
        [1, 0, 0, 1],  # M1: a11+a22
        [0, 0, 1, 1],  # M2: a21+a22
        [1, 0, 0, 0],  # M3: a11
        [0, 0, 0, 1],  # M4: a22
        [1, 1, 0, 0],  # M5: a11+a12
        [-1, 0, 1, 0], # M6: a21-a11
        [0, 1, 0, -1], # M7: a12-a22
    ]
    
    # V vectors (for B: [b11, b12, b21, b22]):
    V_strassen = [
        [1, 0, 0, 1],  # M1: b11+b22
        [1, 0, 0, 0],  # M2: b11
        [0, 1, 0, -1], # M3: b12-b22
        [0, 0, 1, -1], # M4: b21-b11
        [0, 0, 0, 1],  # M5: b22
        [1, 1, 0, 0],  # M6: b11+b12
        [0, 0, 1, 1],  # M7: b21+b22
    ]
    
    # W vectors (for C: [c11, c12, c21, c22]):
    W_strassen = [
        [1, 0, 0, -1],  # M1: c11+1, c22-1
        [0, 0, 1, -1],  # M2: c21+1, c22-1
        [0, 1, 0, 1],   # M3: c12+1, c22+1
        [1, 0, 1, 0],   # M4: c11+1, c21+1
        [-1, 1, 0, 0],  # M5: c11-1, c12+1
        [0, 0, 0, 1],   # M6: c22+1
        [1, 0, 0, 0],   # M7: c11+1
    ]
    
    # Build the full decomposition
    # Tile: i in {0,2}, j in {0,2,4}, k in {0,2,4}
    # For j=4, it's a 2x1x2 block (not 2x2x2)
    
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # Collect all rank-1 terms
    terms_U = []
    terms_V = []
    terms_W = []
    
    # Add 2x2x2 Strassen blocks
    for i_off in [0, 2]:
        for j_off in [0, 2]:  # Only full 2x2 in j
            for k_off in [0, 2, 4]:
                for s in range(7):
                    # U: 4 rows for A submatrix
                    u_vec = [0.0] * U_rows
                    # A[i_off + di, j_off + dj] -> row (i_off+di)*m + (j_off+dj)
                    for di in range(2):
                        for dj in range(2):
                            u_vec[(i_off + di) * m + (j_off + dj)] = U_strassen[s][di * 2 + dj]
                    terms_U.append(u_vec)
                    
                    # V: 4 rows for B submatrix
                    v_vec = [0.0] * V_rows
                    # B[j_off + dj, k_off + dk] -> row (j_off+dj)*p + (k_off+dk)
                    for dj in range(2):
                        for dk in range(2):
                            v_vec[(j_off + dj) * p + (k_off + dk)] = V_strassen[s][dj * 2 + dk]
                    terms_V.append(v_vec)
                    
                    # W: 4 rows for C submatrix
                    w_vec = [0.0] * W_rows
                    # C[k_off + dk, i_off + di] -> row (k_off+dk)*n + (i_off+di)
                    for dk in range(2):
                        for di in range(2):
                            w_vec[(k_off + dk) * n + (i_off + di)] = W_strassen[s][dk * 2 + di]
                    terms_W.append(w_vec)
    
    # Add 2x1x2 blocks for j=4 (single j index)
    # For j=4, we have A[i,4] * B[4,k] contributing to C[i,k]
    # This is just schoolbook: rank = 2*1*2 = 4 per i-block
    for i_off in [0, 2]:
        for k_off in [0, 2, 4]:
            for di in range(2):
                for dk in range(2):
                    u_vec = [0.0] * U_rows
                    u_vec[(i_off + di) * m + 4] = 1.0
                    terms_U.append(u_vec)
                    
                    v_vec = [0.0] * V_rows
                    v_vec[4 * p + (k_off + dk)] = 1.0
                    terms_V.append(v_vec)
                    
                    w_vec = [0.0] * W_rows
                    w_vec[(k_off + dk) * n + (i_off + di)] = 1.0
                    terms_W.append(w_vec)
    
    rank = len(terms_U)
    
    # Convert to matrices
    U = [terms_U[r] for r in range(rank)]
    V = [terms_V[r] for r in range(rank)]
    W = [terms_W[r] for r in range(rank)]
    
    # Wait, the format should be: U is (n*m, rank), so U[row][col]
    # Let me restructure
    U_mat = [[0.0] * rank for _ in range(U_rows)]
    V_mat = [[0.0] * rank for _ in range(V_rows)]
    W_mat = [[0.0] * rank for _ in range(W_rows)]
    
    for r in range(rank):
        for i in range(U_rows):
            if terms_U[r][i] != 0.0:
                U_mat[i][r] = terms_U[r][i]
        for i in range(V_rows):
            if terms_V[r][i] != 0.0:
                V_mat[i][r] = terms_V[r][i]
        for i in range(W_rows):
            if terms_W[r][i] != 0.0:
                W_mat[i][r] = terms_W[r][i]
    
    # Verify
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False, (i, j, k, s)
        return True, None
    
    valid, info = verify(U_mat, V_mat, W_mat, n, m, p, rank)
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U_mat = [[0.0] * rank for _ in range(U_rows)]
        V_mat = [[0.0] * rank for _ in range(V_rows)]
        W_mat = [[0.0] * rank for _ in range(W_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_mat[i*m+j][col] = 1.0
                    V_mat[j*p+k][col] = 1.0
                    W_mat[k*n+i][col] = 1.0
                    col += 1
        rank = n * m * p
    
    if time.time() > deadline:
        pass
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
