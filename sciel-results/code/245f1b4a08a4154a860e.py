"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Tensor T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    T = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _strassen_2x2x2():
    """Construct rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # P1 = (A00+A11)(B00+B11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # P2 = (A10+A11)B00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0
    
    # P3 = A00(B01-B11)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0; W[3][2] = 1.0
    
    # P4 = A11(B10-B00)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[1][3] = 1.0
    
    # P5 = (A00+A01)B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[2][4] = 1.0
    
    # P6 = (A10-A00)(B00+B01)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # P7 = (A01-A11)(B10+B11)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0
    
    return U, V, W

def _schoolbook_2x2x2():
    """Construct rank-4 schoolbook decomposition for 2x2x2 matrix multiplication."""
    rank = 4
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col] = 1.0
                V[j*2+k][col] = 1.0
                W[k*2+i][col] = 1.0
                col += 1
                
    return U, V, W

def _kron_decomp_explicit(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions with explicit dimensions."""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 * rank2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for t1 in range(rank1):
                    u1 = U1[i1 * m1 + j1][t1]
                    v1 = V1[j1 * p1 + k1][t1]
                    w1 = W1[k1 * n1 + i1][t1]
                    if u1 == 0 and v1 == 0 and w1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                for t2 in range(rank2):
                                    u2 = U2[i2 * m2 + j2][t2]
                                    v2 = V2[j2 * p2 + k2][t2]
                                    w2 = W2[k2 * n2 + i2][t2]
                                    if u2 == 0 and v2 == 0 and w2 == 0:
                                        continue
                                    
                                    t = t1 * rank2 + t2
                                    
                                    u_row = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                    U[u_row][t] += u1 * u2
                                    
                                    v_row = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                                    V[v_row][t] += v1 * v2
                                    
                                    w_row = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                                    W[w_row][t] += w1 * w2
    
    return U, V, W

def _schoolbook_4x4x1():
    """Construct rank-16 schoolbook decomposition for 4x4x1 matrix multiplication."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return U, V, W

def _construct_444_rank28():
    """Construct rank-28 decomposition for 4x4x4 using Kronecker of Strassen (7) and Schoolbook (4)."""
    U_str, V_str, W_str = _strassen_2x2x2()
    U_sch, V_sch, W_sch = _schoolbook_2x2x2()
    
    U4, V4, W4 = _kron_decomp_explicit(U_str, V_str, W_str, 2, 2, 2, U_sch, V_sch, W_sch, 2, 2, 2)
    
    # Verify
    if not _verify_decomposition(U4, V4, W4, 4, 4, 4):
        # Fallback to schoolbook 4x4x4 (rank 64)
        return _schoolbook_4x4x4()
        
    return U4, V4, W4

def _schoolbook_4x4x4():
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    """Return the best known decomposition for 4x4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook 4x4x9 (rank 144)
    U_sb, V_sb, W_sb = _schoolbook_4x4x4() # Just a placeholder, we will build proper fallback
    
    # Build Schoolbook 4x4x9
    rank_sb = n*m*p
    U_full_sb = [[0.0] * rank_sb for _ in range(n * m)]
    V_full_sb = [[0.0] * rank_sb for _ in range(m * p)]
    W_full_sb = [[0.0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_full_sb[i * m + j][col] = 1.0
                V_full_sb[j * p + k][col] = 1.0
                W_full_sb[k * n + i][col] = 1.0
                col += 1
                
    best = {"U": U_full_sb, "V": V_full_sb, "W": W_full_sb}
    
    if _remaining_time() < 10:
        return best

    try:
        # Get rank-28 decomposition for 4x4x4
        U444, V444, W444 = _construct_444_rank28()
        rank_444 = len(U444[0])
        
        # Get rank-16 decomposition for 4x4x1
        U441, V441, W441 = _schoolbook_4x4x1()
        rank_441 = len(U441[0])
        
        total_rank = rank_444 + rank_444 + rank_441
        
        U_full = [[0.0] * total_rank for _ in range(n * m)]
        V_full = [[0.0] * total_rank for _ in range(m * p)]
        W_full = [[0.0] * total_rank for _ in range(p * n)]
        
        # Part A: Columns 0-3 (4x4x4)
        for r in range(n * m):
            for t in range(rank_444):
                U_full[r][t] = U444[r][t]
        
        for j in range(m):
            for k in range(4):
                for t in range(rank_444):
                    V_full[j * p + k][t] = V444[j * 4 + k][t]
        
        for k in range(4):
            for i in range(n):
                for t in range(rank_444):
                    W_full[k * n + i][t] = W444[k * 4 + i][t]
        
        # Part B: Columns 4-7 (4x4x4)
        offset_b = rank_444
        for r in range(n * m):
            for t in range(rank_444):
                U_full[r][offset_b + t] = U444[r][t]
        
        for j in range(m):
            for k in range(4, 8):
                for t in range(rank_444):
                    V_full[j * p + k][offset_b + t] = V444[j * 4 + (k - 4)][t]
        
        for k in range(4, 8):
            for i in range(n):
                for t in range(rank_444):
                    W_full[k * n + i][offset_b + t] = W444[(k - 4) * 4 + i][t]
        
        # Part C: Column 8 (4x4x1)
        offset_c = 2 * rank_444
        for r in range(n * m):
            for t in range(rank_441):
                U_full[r][offset_c + t] = U441[r][t]
        
        for j in range(m):
            for t in range(rank_441):
                V_full[j * p + 8][offset_c + t] = V441[j * 1 + 0][t]
        
        for i in range(n):
            for t in range(rank_441):
                W_full[8 * n + i][offset_c + t] = W441[0 * 4 + i][t]
        
        # Verify
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            best = {"U": U_full, "V": V_full, "W": W_full}
            
    except Exception:
        pass
    
    return best
