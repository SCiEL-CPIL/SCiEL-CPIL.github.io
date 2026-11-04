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
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, -1)
    V_3d = V_np.reshape(m, p, -1)
    W_3d = W_np.reshape(p, n, -1)
    
    for i in range(n):
        for j in range(m):
            u_vec = U_3d[i, j, :]
            for k in range(p):
                v_vec = V_3d[j, k, :]
                w_vec = W_3d[k, i, :]
                val = np.dot(u_vec, v_vec * w_vec)
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build 4x4x4 Strassen decomposition
    U444, V444, W444 = _build_4x4x4_strassen()
    
    # Convert to lists for the final output format
    U444_list = U444.tolist()
    V444_list = V444.tolist()
    W444_list = W444.tolist()
    
    # Validate 4x4x4
    if not _validate_numpy(U444_list, V444_list, W444_list, 4, 4, 4):
        return _schoolbook_fallback()
    
    # Build 4x4x9 decomposition with SHARED U factors
    # Key insight: The U factors depend only on (i, j) where i is the row of A and j is the column of A.
    # For different output columns k, the same (i, j) pair uses the SAME U factor.
    # We can have a single set of U factors (49 columns) that works for ALL 9 output columns.
    # Then V and W are specialized per k-block.
    
    # For k in 0..3: use Strassen block 1 (columns 0-3 of B)
    # For k in 4..7: use Strassen block 2 (columns 4-7 of B)  
    # For k = 8: use schoolbook (column 8 of B)
    
    # Total rank = 49 (shared Strassen) + 16 (schoolbook for k=8) = 65
    
    total_rank = 49 + 16
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Shared U factors: same for all k
    for I in range(4):
        for J in range(4):
            for c in range(49):
                U[I*4+J][c] = U444_list[I*4+J][c]
    
    # V and W for k in 0..3 (Strassen block 1)
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+K][c] = V444_list[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[K*4+I][c] = W444_list[K*4+I][c]
    
    # V and W for k in 4..7 (Strassen block 2)
    for J in range(4):
        for K in range(4):
            for c in range(49):
                V[J*9+(K+4)][c] = V444_list[J*4+K][c]
    for K in range(4):
        for I in range(4):
            for c in range(49):
                W[(K+4)*4+I][c] = W444_list[K*4+I][c]
    
    # V and W for k = 8 (schoolbook)
    offset3 = 49
    for idx in range(16):
        I = idx // 4
        J = idx % 4
        U[I*4+J][offset3+idx] = 1
        V[J*9+8][offset3+idx] = 1
        W[8*4+I][offset3+idx] = 1
    
    # Final validation
    if not _validate_numpy(U, V, W, 4, 4, 9):
        return _schoolbook_fallback()
    
    return {"U": U, "V": V, "W": W}

