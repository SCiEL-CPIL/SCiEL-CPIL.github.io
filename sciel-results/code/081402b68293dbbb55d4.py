import time
import numpy as np

def _winograd_2x2():
    """
    Returns U, V, W for 2x2 by 2x2 matrix multiplication using Winograd F(2).
    Rank 7.
    U shape: (4, 7) - indexed by i*2+j
    V shape: (4, 7) - indexed by j*2+k
    W shape: (4, 7) - indexed by k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (a+b)(e+g): c00 += 1, c01 -= 1
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[1][0] = -1
    
    # M2 = (a-b)(e-g): c00 += 1, c01 += 1
    U[0][1] = 1; U[1][1] = -1
    V[0][1] = 1; V[2][1] = -1
    W[0][1] = 1; W[1][1] = 1
    
    # M3 = (c+d)(g+h): c10 += 1, c11 -= 1
    U[2][2] = 1; U[3][2] = 1
    V[2][2] = 1; V[3][2] = 1
    W[2][2] = 1; W[3][2] = -1
    
    # M4 = (c-d)(g-h): c10 += 1, c11 += 1
    U[2][3] = 1; U[3][3] = -1
    V[2][3] = 1; V[3][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = ah: c10 += 1
    U[0][4] = 1
    V[3][4] = 1
    W[2][4] = 1
    
    # M6 = bf: c01 += 1
    U[1][5] = 1
    V[1][5] = 1
    W[1][5] = 1
    
    # M7 = de: c00 += 1, c11 += 1
    U[3][6] = 1
    V[0][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m by m x p matrix multiplication.
    Rank n*m*p.
    U shape: (n*m, rank) - indexed by i*m+j
    V shape: (m*p, rank) - indexed by j*p+k
    W shape: (p*n, rank) - indexed by k*n+i
    """
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

def _hybrid_2x2_x_2x3():
    """
    Returns U, V, W for (2x2) * (2x3) matrix multiplication.
    This multiplies a 2x2 matrix by a 2x3 matrix, producing a 2x3 result.
    Rank 9.
    U shape: (4, 9) - indexed by i*2+j for A (2x2)
    V shape: (6, 9) - indexed by j*3+k for B (2x3)
    W shape: (6, 9) - indexed by k*2+i for C (2x3)
    """
    # Split into: (2x2)*(2x1) for k=0, and (2x2)*(2x1) for k=1, and (2x2)*(2x1) for k=2
    # Each (2x2)*(2x1) has rank... let's use schoolbook for simplicity: rank 4 each
    # Total rank 12, but we can do better.
    # Actually, let's just use schoolbook for now: rank 2*2*3 = 12
    # But we can be smarter: for each output column k, we compute A @ B[:,k]
    # A @ B[:,k] is (2x2)*(2x1) = 2x1, which has rank 4 by schoolbook
    # Total: 4*3 = 12
    # Let's try a better approach: Winograd-like
    # For 2x2 times 2x1, rank is 4 (schoolbook) or can it be lower?
    # Actually (2x2)*(2x1) is just A times a vector, which is rank 4 by schoolbook
    # Minimum rank for (n,m,p) where m*p and n*m... 
    # Let's just use schoolbook: rank 12
    rank = 12
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return U, V, W

def _hybrid_2x3_x_2x2():
    """
    Returns U, V, W for (2x3) * (3x2) matrix multiplication.
    Rank 12 by schoolbook.
    U shape: (6, rank) - indexed by i*3+j for A (2x3)
    V shape: (6, rank) - indexed by j*2+k for B (3x2)
    W shape: (4, rank) - indexed by k*2+i for C (2x2)
    """
    n, m, p = 2, 3, 2
    rank = n*m*p
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

def _schoolbook_2x3_3x3():
    """
    Returns U, V, W for (2x3) * (3x3) matrix multiplication.
    Rank 18 by schoolbook.
    U shape: (6, rank) - indexed by i*3+j for A (2x3)
    V shape: (9, rank) - indexed by j*3+k for B (3x3)
    W shape: (6, rank) - indexed by k*2+i for C (2x3)
    """
    n, m, p = 2, 3, 3
    rank = n*m*p
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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin
    
    n, m, p = 2, 5, 5
    
    # Global shapes:
    # U: (n*m, rank) = (10, rank), indexed by i*5+j
    # V: (m*p, rank) = (25, rank), indexed by j*5+k
    # W: (p*n, rank) = (10, rank), indexed by k*2+i
    
    # We'll accumulate columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_loc, V_loc, W_loc, i_off, j_off, k_off):
        """
        U_loc indexed by local_i * m_loc + local_j
        V_loc indexed by local_j * p_loc + local_k
        W_loc indexed by local_k * n_loc + local_i
        
        Global:
        U global row = i_off + local_i, but within row: j offset
        Actually U[i*m+j], so local U row = li*m_loc + lj
        Global U row = (i_off+li)*m + (j_off + lj)
        
        V[j*p+k], local V row = lj*p_loc + lk
        Global V row = (j_off+lj)*p + (k_off + lk)
        
        W[k*n+i], local W row = lk*n_loc + li
        Global W row = (k_off+lk)*n + (i_off + li)
        """
        m_loc = V_loc[0].__len__() if False else len(V_loc[0])  # This doesn't work
        # We need to know local dimensions
        # Let's infer: U_loc has rows = n_loc*m_loc, V_loc has rows = m_loc*p_loc, W_loc has rows = p_loc*n_loc
        n_loc = len(W_loc[0]) if False else None
        # Better: pass dimensions or infer
        # U_loc rows = n_loc * m_loc
        # V_loc rows = m_loc * p_loc  
        # W_loc rows = p_loc * n_loc
        # From U_loc rows and V_loc rows:
        # Let's just require consistent indexing
        
        rank_loc = len(U_loc[0]) if U_loc and len(U_loc) > 0 else 0
        if rank_loc == 0:
            return
            
        # Determine local dims
        # For simplicity, we'll infer from the structure
        # U_loc has n_loc*m_loc rows
        # We need n_loc and m_loc
        # W_loc has p_loc*n_loc rows
        # Let's assume n_loc=2 for our blocks
        
        U_loc_rows = len(U_loc)
        V_loc_rows = len(V_loc)
        W_loc_rows = len(W_loc)
        
        # n_loc * m_loc = U_loc_rows
        # m_loc * p_loc = V_loc_rows
        # p_loc * n_loc = W_loc_rows
        
        # For our specific blocks, we know the dimensions
        # Block 1: 2x2 x 2x2 -> n=2,m=2,p=2
        # Block 2: 2x2 x 2x3 -> n=2,m=2,p=3
        # Block 3: 2x3 x 3x2 -> n=2,m=3,p=2
        # Block 4: 2x3 x 3x3 -> n=2,m=3,p=3
        
        # We'll determine n_loc from W_loc_rows / p_loc or U_loc_rows / m_loc
        # This is getting complex. Let me just hardcode the mappings.
        
        # Actually, let me restructure: pass the local dimensions
        # For now, let me just manually add each block
        
    # Let me redo this more carefully by directly constructing
    
    total_rank = 0
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # i in [0,1], j in [0,1], k in [0,1]
    U1, V1, W1 = _winograd_2x2()
    rank1 = len(U1[0])
    for r in range(rank1):
        u_col = [0]*10
        v_col = [0]*25
        w_col = [0]*10
        for li in range(2):
            for lj in range(2):
                u_col[(li)*5 + lj] = U1[li*2+lj][r]
        for lj in range(2):
            for lk in range(2):
                v_col[(lj)*5 + lk] = V1[lj*2+lk][r]
        for lk in range(2):
            for li in range(2):
                w_col[(lk)*2 + li] = W1[lk*2+li][r]
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    total_rank += rank1
    
    # Block 2: A[0:2,0:2] * B[0:2,2:5] -> C[2:5,0:2]
    # i in [0,1], j in [0,1], k in [2,4]
    U2, V2, W2 = _hybrid_2x2_x_2x3()
    rank2 = len(U2[0])
    for r in range(rank2):
        u_col = [0]*10
        v_col = [0]*25
        w_col = [0]*10
        for li in range(2):
            for lj in range(2):
                u_col[(li)*5 + lj] = U2[li*2+lj][r]
        for lj in range(2):
            for lk in range(3):
                v_col[(lj)*5 + (lk+2)] = V2[lj*3+lk][r]
        for lk in range(3):
            for li in range(2):
                w_col[((lk+2))*2 + li] = W2[lk*2+li][r]
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    total_rank += rank2
    
    # Block 3: A[0:2,2:5] * B[2:5,0:2] -> C[0:2,2:5]
    # i in [0,1], j in [2,4], k in [0,1]
    U3, V3, W3 = _hybrid_2x3_x_2x2()
    rank3 = len(U3[0])
    for r in range(rank3):
        u_col = [0]*10
        v_col = [0]*25
        w_col = [0]*10
        for li in range(2):
            for lj in range(3):
                u_col[(li)*5 + (lj+2)] = U3[li*3+lj][r]
        for lj in range(3):
            for lk in range(2):
                v_col[(lj+2)*5 + lk] = V3[lj*2+lk][r]
        for lk in range(2):
            for li in range(2):
                w_col[(lk)*2 + li] = W3[lk*2+li][r]
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    total_rank += rank3
    
    # Block 4: A[0:2,2:5] * B[2:5,2:5] -> C[2:5,2:5]
    # i in [0,1], j in [2,4], k in [2,4]
    U4, V4, W4 = _schoolbook_2x3_3x3()
    rank4 = len(U4[0])
    for r in range(rank4):
        u_col = [0]*10
        v_col = [0]*25
        w_col = [0]*10
        for li in range(2):
            for lj in range(3):
                u_col[(li)*5 + (lj+2)] = U4[li*3+lj][r]
        for lj in range(3):
            for lk in range(3):
                v_col[(lj+2)*5 + (lk+2)] = V4[lj*3+lk][r]
        for lk in range(3):
            for li in range(2):
                w_col[((lk+2))*2 + li] = W4[lk*2+li][r]
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    total_rank += rank4
    
    # Build final matrices
    U = [[0]*total_rank for _ in range(10)]
    V = [[0]*total_rank for _ in range(25)]
    W = [[0]*total_rank for _ in range(10)]
    
    for r in range(total_rank):
        for idx in range(10):
            U[idx][r] = U_cols[r][idx]
        for idx in range(25):
            V[idx][r] = V_cols[r][idx]
        for idx in range(10):
            W[idx][r] = W_cols[r][idx]
    
    return {"U": U, "V": V, "W": W}
