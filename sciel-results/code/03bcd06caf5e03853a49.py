"""Matrix multiplication 3x5x5 - block decomposition with Strassen"""
DIMENSIONS = [3, 5, 5]

import time
_start = time.time()
_DEADLINE = 120.0

def _get_strassen_222():
    """Return U,V,W for 2x2x2 Strassen (rank 7).
    U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    Index: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For n=m=p=2: U[2i+j, c], V[2j+k, c], W[2k+i, c]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11=(0,0), A22=(1,1)
    U[0][0] = 1; U[3][0] = 1
    # V: B11=(0,0), B22=(1,1)
    V[0][0] = 1; V[3][0] = 1
    # W: C11=(k=0,i=0), C12=(k=0,i=1), C21=(k=1,i=0), C22=(k=1,i=1)
    # C = M1 + M4 - M5 + M7 for C11
    # C = M3 + M5 for C12
    # C = M2 + M4 for C21
    # C = M1 - M2 + M3 + M6 for C22
    
    # Let me define the 7 Strassen products properly:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Reset and fill properly
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # M1 contributes to C11(+1) and C22(+1)
    W[0][0] = 1   # C11: k=0,i=0
    W[3][0] = 1   # C22: k=1,i=1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # M2 contributes to C21(+1) and C22(-1)
    W[2][1] = 1   # C21: k=1,i=0
    W[3][1] = -1  # C22: k=1,i=1
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # M3 contributes to C12(+1) and C22(+1)
    W[1][2] = 1   # C12: k=0,i=1
    W[3][2] = 1   # C22: k=1,i=1
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # M4 contributes to C11(+1) and C21(+1)
    W[0][3] = 1   # C11: k=0,i=0
    W[2][3] = 1   # C21: k=1,i=0
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # M5 contributes to C11(-1) and C12(+1)
    W[0][4] = -1  # C11: k=0,i=0
    W[1][4] = 1   # C12: k=0,i=1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # M6 contributes to C22(+1)
    W[3][5] = 1   # C22: k=1,i=1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # M7 contributes to C11(+1)
    W[0][6] = 1   # C11: k=0,i=0
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for n x m x p."""
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
    return U, V, W

def _build_block(n, m, p):
    """Build a decomposition for n x m x p matrix multiplication."""
    if n == 2 and m == 2 and p == 2:
        return _get_strassen_222()
    else:
        return _schoolbook_nm_p(n, m, p)

def _assemble(n_rows, n_cols, blocks, block_results):
    """Assemble block results into full U, V, W matrices.
    n_rows: number of rows in output (n)
    n_cols: number of cols in output (p)
    blocks: list of (i_start, j_start, k_start, n_b, m_b, p_b)
    block_results: list of (U_b, V_b, W_b) for each block
    """
    total_rank = sum(len(Wb) for _, _, Wb in block_results)
    U = [[0]*total_rank for _ in range(n_rows * n_cols)]  # Wait, this is wrong
    
    # Actually for matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    # 
    # But we're decomposing the tensor for C = A*B where A is n_rows x n_mid, B is n_mid x n_cols
    # So the full dimensions are: n = n_rows, m = n_mid, p = n_cols
    # U: (n*m, rank) = (n_rows * n_mid, rank)
    # V: (m*p, rank) = (n_mid * n_cols, rank)
    # W: (p*n, rank) = (n_cols * n_rows, rank)
    
    # I need to know n_mid too. Let me restructure.
    return None  # Placeholder

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Partition: split into blocks
    # n = 3 = 2 + 1
    # m = 5 = 2 + 2 + 1
    # p = 5 = 2 + 2 + 1
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Build all block decompositions
    block_decomps = []
    for n_b in n_parts:
        for m_b in m_parts:
            for p_b in p_parts:
                U_b, V_b, W_b = _build_block(n_b, m_b, p_b)
                block_decomps.append((n_b, m_b, p_b, U_b, V_b, W_b))
    
    total_rank = sum(len(W_b) for _, _, _, _, _, W_b in block_decomps)
    
    # Allocate full matrices
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Fill in blocks
    col_offset = 0
    i_off = 0
    for ni, n_b in enumerate(n_parts):
        j_off = 0
        for mj, m_b in enumerate(m_parts):
            k_off = 0
            for pk, p_b in enumerate(p_parts):
                # Find the block decom
                idx = ni * len(m_parts) * len(p_parts) + mj * len(p_parts) + pk
                _, _, _, U_b, V_b, W_b = block_decomps[idx]
                r_b = len(W_b)
                
                # Fill U: rows i_off..i_off+n_b-1, cols j_off..j_off+m_b-1
                for ii in range(n_b):
                    for jj in range(m_b):
                        row = i_off + ii
                        col_local = jj  # Within block, U index is ii*m_b + jj
                        for c in range(r_b):
                            U[row * m + j_off + jj][col_offset + c] = U_b[ii * m_b + jj][c]
                
                # Fill V: rows j_off..j_off+m_b-1, cols k_off..k_off+p_b-1
                for jj in range(m_b):
                    for kk in range(p_b):
                        row = j_off + jj
                        col_local = kk
                        for c in range(r_b):
                            V[row * p + k_off + kk][col_offset + c] = V_b[jj * p_b + kk][c]
                
                # Fill W: W index is k*n + i, so row = k_off+kk mapped to (k_off+kk)*n + i_off+ii
                for kk in range(p_b):
                    for ii in range(n_b):
                        for c in range(r_b):
                            W[(k_off + kk) * n + i_off + ii][col_offset + c] = W_b[kk * n_b + ii][c]
                
                col_offset += r_b
                k_off += p_b
            j_off += m_b
        i_off += n_b
    
    # Verify time
    elapsed = time.time() - _start
    if elapsed > _DEADLINE - 5:
        # Safety: return schoolbook if we're running out of time
        return _schoolbook_nm_p(n, m, p)
    
    return {"U": U, "V": V, "W": W}
