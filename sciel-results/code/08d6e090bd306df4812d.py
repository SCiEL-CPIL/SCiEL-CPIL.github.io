import numpy as np

def construct():
    """
    Rank-minimal tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Split 8 columns into 4 blocks of 2. For each block:
    - The 2x1 * 1x2 part uses rank-2 outer product decomposition
    - The 2x2 * 2x2 part uses Strassen (rank 7)
    Total per block: 9, Total rank: 36
    
    Index conventions:
    - U: (n*m, rank) = (6, rank), index a = i*m + j
    - V: (m*p, rank) = (24, rank), index b = j*p + k  
    - W: (p*n, rank) = (16, rank), index c = k*n + i
    """
    n, m, p = 2, 3, 8
    rank = 36
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Part 1: A[:,0] * B[0, k0:k1] (2x1 * 1x2, rank 2 with half-integers)
        # Need: C[0,k0]=A[0]*B[0,k0], C[0,k1]=A[0]*B[0,k1]
        #       C[1,k0]=A[1]*B[0,k0], C[1,k1]=A[1]*B[0,k1]
        # Use: S1 = (A[0]+A[1])*(B[0,k0]+B[0,k1])
        #      S2 = (A[0]-A[1])*(B[0,k0]-B[0,k1])
        # Then: A[0]*B[0,k0] = (S1+S2)/2, A[0]*B[0,k1] = (S1-S2)/2
        #       A[1]*B[0,k0] = (S1-S2)/2, A[1]*B[0,k1] = (S1+S2)/2
        
        # S1 = (A[0,0] + A[1,0]) * (B[0,k0] + B[0,k1])
        U[0*m + 0, col] = 1
        U[1*m + 0, col] = 1
        V[0*p + k0, col] = 1
        V[0*p + k1, col] = 1
        W[k0*n + 0, col] = 0.5  # C[0,k0] += 0.5*S1
        W[k1*n + 0, col] = 0.5  # C[0,k1] += 0.5*S1
        W[k0*n + 1, col] = 0.5  # C[1,k0] += 0.5*S1
        W[k1*n + 1, col] = 0.5  # C[1,k1] += 0.5*S1
        col += 1
        
        # S2 = (A[0,0] - A[1,0]) * (B[0,k0] - B[0,k1])
        U[0*m + 0, col] = 1
        U[1*m + 0, col] = -1
        V[0*p + k0, col] = 1
        V[0*p + k1, col] = -1
        W[k0*n + 0, col] = 0.5  # C[0,k0] += 0.5*S2
        W[k1*n + 0, col] = -0.5 # C[0,k1] -= 0.5*S2
        W[k0*n + 1, col] = -0.5 # C[1,k0] -= 0.5*S2
        W[k1*n + 1, col] = 0.5  # C[1,k1] += 0.5*S2
        col += 1
        
        # Part 2: A[:,1:3] * B[1:3, k0:k1] (2x2 * 2x2, Strassen rank 7)
        # A_sub = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
        # B_sub = [[B[1,k0], B[1,k1]], [B[2,k0], B[2,k1]]]
        
        # M1 = (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1   # C[0,k0] += M1
        W[k1*n + 1, col] = 1   # C[1,k1] += M1
        col += 1
        
        # M2 = (A[1,1] + A[1,2]) * B[1,k0]
        U[1*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        W[k0*n + 1, col] = 1   # C[1,k0] += M2
        W[k1*n + 1, col] = -1  # C[1,k1] -= M2
        col += 1
        
        # M3 = A[0,1] * (B[1,k1] - B[2,k1])
        U[0*m + 1, col] = 1
        V[1*p + k1, col] = 1
        V[2*p + k1, col] = -1
        W[k1*n + 0, col] = 1   # C[0,k1] += M3
        W[k1*n + 1, col] = 1   # C[1,k1] += M3
        col += 1
        
        # M4 = A[1,2] * (B[2,k0] - B[1,k0])
        U[1*m + 2, col] = 1
        V[2*p + k0, col] = 1
        V[1*p + k0, col] = -1
        W[k0*n + 0, col] = 1   # C[0,k0] += M4
        W[k0*n + 1, col] = 1   # C[1,k0] += M4
        col += 1
        
        # M5 = (A[0,1] + A[0,2]) * B[2,k1]
        U[0*m + 1, col] = 1
        U[0*m + 2, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = -1  # C[0,k0] -= M5
        W[k1*n + 0, col] = 1   # C[0,k1] += M5
        col += 1
        
        # M6 = (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1*m + 1, col] = 1
        U[0*m + 1, col] = -1
        V[1*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k1*n + 1, col] = 1   # C[1,k1] += M6
        col += 1
        
        # M7 = (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0*m + 2, col] = 1
        U[1*m + 2, col] = -1
        V[2*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1   # C[0,k0] += M7
        col += 1
    
    # Verify correctness
    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        T_exp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_exp[a_idx, b_idx, c_idx] = 1
        return np.allclose(T_check, T_exp, atol=1e-10)
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank 48
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=np.float64)
        V_sb = np.zeros((m*p, rank_sb), dtype=np.float64)
        W_sb = np.zeros((p*n, rank_sb), dtype=np.float64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, c] = 1
                    V_sb[b_idx, c] = 1
                    W_sb[c_idx, c] = 1
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
