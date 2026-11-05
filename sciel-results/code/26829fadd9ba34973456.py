import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    DEADLINE = 140.0
    
    def laderman_333():
        """
        Laderman's 3x3x3 matrix multiplication algorithm (rank 23).
        Returns U, V, W where:
          U: (9, 23) - rows indexed by i*3+j (A entries)
          V: (9, 23) - rows indexed by j*3+k (B entries)
          W: (9, 23) - rows indexed by k*3+i (C entries)
        """
        rank = 23
        U = np.zeros((9, rank), dtype=np.float64)
        V = np.zeros((9, rank), dtype=np.float64)
        W = np.zeros((9, rank), dtype=np.float64)
        
        # Laderman's 23 products for C = A * B (3x3)
        # Term 0: c11 += a11*b11
        U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
        # Term 1: c22 += a22*b22
        U[4,1] = 1; V[4,1] = 1; W[4,1] = 1
        # Term 2: c33 += a33*b33
        U[8,2] = 1; V[8,2] = 1; W[8,2] = 1
        
        # Terms 3-6: for the (1,2) block
        # (a11+a12)(b21+b22)
        U[0,3] = 1; U[1,3] = 1
        V[3,3] = 1; V[4,3] = 1
        W[0,3] = 1; W[3,3] = 1
        # (a11-a12)(b21-b22)
        U[0,4] = 1; U[1,4] = -1
        V[3,4] = 1; V[4,4] = -1
        W[0,4] = 1; W[3,4] = 1
        # (a11+a12)(b21-b22)
        U[0,5] = 1; U[1,5] = 1
        V[3,5] = 1; V[4,5] = -1
        W[0,5] = 1; W[3,5] = -1
        # (a11-a12)(b21+b22)
        U[0,6] = 1; U[1,6] = -1
        V[3,6] = 1; V[4,6] = 1
        W[0,6] = 1; W[3,6] = -1
        
        # Terms 7-10: for the (2,3) block
        # (a22+a23)(b32+b33)
        U[4,7] = 1; U[5,7] = 1
        V[6,7] = 1; V[7,7] = 1
        W[4,7] = 1; W[6,7] = 1
        # (a22-a23)(b32-b33)
        U[4,8] = 1; U[5,8] = -1
        V[6,8] = 1; V[7,8] = -1
        W[4,8] = 1; W[6,8] = 1
        # (a22+a23)(b32-b33)
        U[4,9] = 1; U[5,9] = 1
        V[6,9] = 1; V[7,9] = -1
        W[4,9] = 1; W[6,9] = -1
        # (a22-a23)(b32+b33)
        U[4,10] = 1; U[5,10] = -1
        V[6,10] = 1; V[7,10] = 1
        W[4,10] = 1; W[6,10] = -1
        
        # Terms 11-14: for the (3,1) block
        # (a33+a31)(b13+b11)
        U[8,11] = 1; U[6,11] = 1
        V[2,11] = 1; V[0,11] = 1
        W[8,11] = 1; W[2,11] = 1
        # (a33-a31)(b13-b11)
        U[8,12] = 1; U[6,12] = -1
        V[2,12] = 1; V[0,12] = -1
        W[8,12] = 1; W[2,12] = 1
        # (a33+a31)(b13-b11)
        U[8,13] = 1; U[6,13] = 1
        V[2,13] = 1; V[0,13] = -1
        W[8,13] = 1; W[2,13] = -1
        # (a33-a31)(b13+b11)
        U[8,14] = 1; U[6,14] = -1
        V[2,14] = 1; V[0,14] = 1
        W[8,14] = 1; W[2,14] = -1
        
        # Terms 15-18: for the (1,3) and (3,1) cross terms
        # (a13+a31)(b11+b33)
        U[2,15] = 1; U[6,15] = 1
        V[0,15] = 1; V[8,15] = 1
        W[0,15] = 1; W[8,15] = 1
        # (a13-a31)(b11-b33)
        U[2,16] = 1; U[6,16] = -1
        V[0,16] = 1; V[8,16] = -1
        W[0,16] = 1; W[8,16] = 1
        # (a13+a31)(b11-b33)
        U[2,17] = 1; U[6,17] = 1
        V[0,17] = 1; V[8,17] = -1
        W[0,17] = 1; W[8,17] = -1
        # (a13-a31)(b11+b33)
        U[2,18] = 1; U[6,18] = -1
        V[0,18] = 1; V[8,18] = 1
        W[0,18] = 1; W[8,18] = -1
        
        # Terms 19-22: for the (1,2) and (2,1) cross terms
        # (a12+a21)(b12+b21)
        U[1,19] = 1; U[3,19] = 1
        V[1,19] = 1; V[3,19] = 1
        W[3,19] = 1; W[1,19] = 1
        # (a12-a21)(b12-b21)
        U[1,20] = 1; U[3,20] = -1
        V[1,20] = 1; V[3,20] = -1
        W[3,20] = 1; W[1,20] = 1
        # (a12+a21)(b12-b21)
        U[1,21] = 1; U[3,21] = 1
        V[1,21] = 1; V[3,21] = -1
        W[3,21] = 1; W[1,21] = -1
        # (a12-a21)(b12+b21)
        U[1,22] = 1; U[3,22] = -1
        V[1,22] = 1; V[3,22] = 1
        W[3,22] = 1; W[1,22] = -1
        
        return U, V, W
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k"""
        rank = U.shape[1]
        # Check all required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Build the 3x3x6 decomposition
    U_l, V_l, W_l = laderman_333()
    
    # Verify Laderman's 3x3x3
    laderman_ok = verify_decomp(U_l, V_l, W_l, 3, 3, 3)
    
    if not laderman_ok:
        # Fallback to schoolbook rank 54
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Construct rank-46 solution: two copies of Laderman
    rank = 46
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: k = 0,1,2
    for r in range(23):
        U[:, r] = U_l[:, r]
        for j in range(3):
            for k in range(3):
                V[j*3+k, r] = V_l[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W[k*3+i, r] = W_l[k*3+i, r]
    
    # Block 2: k = 3,4,5
    for r in range(23):
        r2 = r + 23
        U[:, r2] = U_l[:, r]
        for j in range(3):
            for k in range(3):
                V[j*3 + (k+3), r2] = V_l[j*3+k, r]
        for k in range(3):
            for i in range(3):
                W[(k+3)*3 + i, r2] = W_l[k*3+i, r]
    
    # Verify the full decomposition
    if verify_decomp(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
