import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Strassen at 4x4 level (blocks are 2x2)
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # bigM1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # bigM2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    # bigM3 = A11(B12-B22) -> C12 += M3, C22 += M3
    # bigM4 = A22(B21-B11) -> C11 += M4, C21 += M4
    # bigM5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    # bigM6 = (A21-A11)(B11+B12) -> C22 += M6
    # bigM7 = (A12-A22)(B21+B22) -> C11 += M7
    
    big_mults = [
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((0,1), 1), ((1,1), -1)]
        },
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((1,0), 1), ((1,1), 1)]
        },
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((0,1), 1)]
        },
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((1,0), 1)]
        },
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    for big_idx, bm in enumerate(big_mults):
        for r2 in range(7):
            col = big_idx * 7 + r2
            
            # Update U
            for (bi, bj), sign_a in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        u_val = U2[i2*2+k2, r2]
                        if u_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + k2
                            U[I*4+J, col] += sign_a * u_val
            
            # Update V
            for (bk, bj), sign_b in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        v_val = V2[k2*2+j2, r2]
                        if v_val != 0:
                            J = 2*bk + k2
                            K = 2*bj + j2
                            V[J*4+K, col] += sign_b * v_val
            
            # Update W
            for (bi, bj), sign_c in bm['C']:
                for i2 in range(2):
                    for j2 in range(2):
                        w_val = W2[i2*2+j2, r2]
                        if w_val != 0:
                            I = 2*bi + i2
                            J = 2*bj + j2
                            W[I*4+J, col] += sign_c * w_val
                            
    return U, V, W

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def _validate_numpy(U, V, W, n, m, p):
    """Validate using NumPy for speed."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if val != 1:
                    return False
    return True

def _build_4x4x9_structured():
    """
    Build 4x4x9 decomposition exploiting the module structure.
    Uses one Strassen 4x4x4 block for columns 0-3, and a shared-structure
    approach for columns 4-8 that leverages the fact that U depends only on A.
    
    Key insight: For columns 4-8, we can use a decomposition where the U factor
    is derived from the same algebraic structure as the first block, potentially
    allowing for rank reduction through shared invariants.
    """
    U444, V444, W444 = _build_4x4x4_strassen()
    
    # For columns 4-8, use a hybrid approach:
    # - Columns 4-7: Use the same Strassen pattern as columns 0-3 (shared U)
    # - Column 8: Use a minimal decomposition
    
    # Total rank: 49 (cols 0-3) + 49 (cols 4-7, shared U) + 16 (col 8)
    # But we can potentially reduce by noticing that if U is shared,
    # we might be able to use a lower-rank decomposition for the combined 8 columns.
    
    # For now, use the straightforward approach but with optimized construction
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: columns 0-3
    U[:, 0:49] = U444
    for J in range(4):
        for K in range(4):
            V[J*9+K, 0:49] = V444[J*4+K, :]
    for K in range(4):
        for I in range(4):
            W[K*4+I, 0:49] = W444[K*4+I, :]
    
    # Block 2: columns 4-7 (same U pattern)
    U[:, 49:98] = U444
    for J in range(4):
        for K in range(4):
            V[J*9+(K+4), 49:98] = V444[J*4+K, :]
    for K in range(4):
        for I in range(4):
            W[(K+4)*4+I, 49:98] = W444[K*4+I, :]
    
    # Block 3: column 8 (schoolbook)
    for idx in range(16):
        I = idx // 4
        J = idx % 4
        U[I*4+J, 98+idx] = 1
        V[J*9+8, 98+idx] = 1
        W[8*4+I, 98+idx] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build 4x4x9 decomposition
    U, V, W = _build_4x4x9_structured()
    
    # Convert to lists for the final output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Validate
    if not _validate_numpy(U_list, V_list, W_list, 4, 4, 9):
        return _schoolbook_fallback()
    
    return {"U": U_list, "V": V_list, "W": W_list}
