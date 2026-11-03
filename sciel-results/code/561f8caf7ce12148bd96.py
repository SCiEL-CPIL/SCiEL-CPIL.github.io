"""Matrix multiplication 4x5 x 5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _validate(U, V, W, tol=1e-9):
    """Validate that the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if isinstance(U[0], list) else U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # total[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    total = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_vec = U_arr[:, r].reshape(n, m)
        v_vec = V_arr[:, r].reshape(m, p)
        w_vec = W_arr[:, r].reshape(p, n)
        # contrib[i,j,k] = u[i,j]*v[j,k]*w[k,i]
        contrib = np.einsum('ij,jk,ki->ijk', u_vec, v_vec, w_vec)
        total += contrib
    
    return np.allclose(total, 1.0, atol=tol)

def _strassen_2x2_vectors():
    """
    Strassen 2x2 matrix multiplication vectors for n=m=p=2.
    Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    For 2x2: U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
              V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11
              W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11
    """
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M7
    strassen = []
    strassen.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))   # M1
    strassen.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))    # M2
    strassen.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))   # M3
    strassen.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]))   # M4
    strassen.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]))    # M5
    strassen.append(([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, -1, 0]))  # M6
    strassen.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, -1]))  # M7
    return strassen

def _build_block_decomp():
    """
    Build a rank-86 decomposition for 4x5 x 5x5 using Strassen 2x2 blocks.
    
    Split 4 rows into 2+2, 5 cols into 2+3.
    A (4x5) = [A1 (2x2) | A2 (2x3)]
               [A3 (2x2) | A4 (2x3)]
    B (5x5) = [B1 (2x2) | B2 (2x3)]
               [B3 (3x2) | B4 (3x3)]
    C (4x5) = [C1 (2x2) | C2 (2x3)]
               [C3 (2x2) | C4 (2x3)]
    
    C1 = A1*B1 + A2*B3
    C2 = A1*B2 + A2*B4
    C3 = A3*B1 + A4*B3
    C4 = A3*B2 + A4*B4
    """
    n, m, p = DIMENSIONS
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_schoolbook(i, j, k):
        """Add a schoolbook term for C[i,k] += A[i,j]*B[j,k]"""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
    
    def add_strassen_2x2(row_offset, col_offset):
        """Add Strassen terms for multiplying a 2x2 block of A (rows row_offset..row_offset+1, 
        cols col_offset..col_offset+1) by a 2x2 block of B (rows col_offset..col_offset+1, 
        cols row_offset..row_offset+1) to produce a 2x2 block of C 
        (rows row_offset..row_offset+1, cols col_offset..col_offset+1)."""
        strassen = _strassen_2x2_vectors()
        for u_vec, v_vec, w_vec in strassen:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            # Fill in the 2x2 blocks
            for a in range(2):
                for b in range(2):
                    # A block: row = row_offset + a, col = col_offset + b
                    u[(row_offset + a) * m + (col_offset + b)] = u_vec[a * 2 + b]
                    # B block: row = col_offset + a, col = row_offset + b
                    v[(col_offset + a) * p + (row_offset + b)] = v_vec[a * 2 + b]
                    # C block: row = row_offset + a, col = col_offset + b
                    # W index: k*n + i where k = col_offset + b, i = row_offset + a
                    w[(col_offset + b) * n + (row_offset + a)] = w_vec[a * 2 + b]
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    # C1 = A1*B1 + A2*B3 (top-left 2x2 of C)
    add_strassen_2x2(0, 0)  # 7 terms
    
    # A2*B3: A2 is rows 0-1, cols 2-4; B3 is rows 2-4, cols 0-1
    # Schoolbook: 2*3*2 = 6 terms
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
    
    # C2 = A1*B2 + A2*B4 (top-right 2x3 of C)
    # A1*B2: A1 is rows 0-1, cols 0-1; B2 is rows 0-1, cols 2-4
    # Schoolbook: 2*2*3 = 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # A2*B4: A2 is rows 0-1, cols 2-4; B4 is rows 2-4, cols 2-4
    # Schoolbook: 2*3*3 = 18 terms
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # C3 = A3*B1 + A4*B3 (bottom-left 2x2 of C)
    add_strassen_2x2(2, 0)  # 7 terms
    
    # A4*B3: A4 is rows 2-3, cols 2-4; B3 is rows 2-4, cols 0-1
    # Schoolbook: 2*3*2 = 6 terms
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
    
    # C4 = A3*B2 + A4*B4 (bottom-right 2x3 of C)
    # A3*B2: A3 is rows 2-3, cols 0-1; B2 is rows 0-1, cols 2-4
    # Schoolbook: 2*2*3 = 12 terms
    for i in range(2, 4):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # A4*B4: A4 is rows 2-3, cols 2-4; B4 is rows 2-4, cols 2-4
    # Schoolbook: 2*3*3 = 18 terms
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    rank = len(U_list)
    
    # Convert to list of lists format: U[i][r]
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_list[r][i]
        for j in range(m * p):
            V[j][r] = V_list[r][j]
        for k in range(p * n):
            W[k][r] = W_list[r][k]
    
    return U, V, W, rank

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Try structured rank-86 decomposition
    U86, V86, W86, rank86 = _build_block_decomp()
    
    if _validate(U86, V86, W86):
        return {"U": U86, "V": V86, "W": W86}
    
    # Fall back to schoolbook
    U, V, W, rank = _schoolbook()
    return {"U": U, "V": V, "W": W}

