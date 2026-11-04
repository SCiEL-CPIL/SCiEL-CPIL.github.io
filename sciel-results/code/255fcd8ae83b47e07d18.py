"""Program for 4x4x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for 4x4x4 (rank 49) and Schoolbook for 4x4x1 (rank 16).
    # Total rank 65.
    
    # Strassen 2x2x2 terms
    # Indices: 0,1
    s_terms = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1})
    ]
    
    # Build 4x4x4 terms by composing outer and inner 2x2x2
    terms_4x4 = []
    for oA, oB, oC in s_terms:
        for iA, iB, iC in s_terms:
            A = {}
            B = {}
            C = {}
            for (bI, bJ), oa in oA.items():
                for (li, lj), ia in iA.items():
                    gi, gj = bI*2+li, bJ*2+lj
                    A[(gi,gj)] = A.get((gi,gj),0) + oa*ia
            for (bJ, bK), ob in oB.items():
                for (lj, lk), ib in iB.items():
                    gj, gk = bJ*2+lj, bK*2+lk
                    B[(gj,gk)] = B.get((gj,gk),0) + ob*ib
            for (bK, bI), oc in oC.items():
                for (lk, li), ic in iC.items():
                    gk, gi = bK*2+lk, bI*2+li
                    C[(gk,gi)] = C.get((gk,gi),0) + oc*ic
            terms_4x4.append((A,B,C))
            
    R1 = len(terms_4x4) # 49
    R2 = n * m # 16
    rank = R1 + R2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Fill U, V, W for 4x4x4 part
    for c, (A, B, C) in enumerate(terms_4x4):
        for (i, j), val in A.items():
            U[i*m+j][c] += val
        for (j, k), val in B.items():
            V[j*p+k][c] += val
        for (k, i), val in C.items():
            W[k*n+i][c] += val
            
    # Fill U, V, W for 4x4x1 part (column 4)
    t = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j][R1+t] = 1
            V[j*p+4][R1+t] = 1
            W[4*n+i][R1+t] = 1
            t += 1
            
    # Verify
    import numpy as np
    A_mat = np.random.rand(n, m)
    B_mat = np.random.rand(m, p)
    C_mat = A_mat @ B_mat
    
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    
    C_dec = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                s += np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
            C_dec[i,k] = s
            
    if not np.allclose(C_dec, C_mat, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0]*rank_sb for _ in range(n*m)]
        V_sb = [[0]*rank_sb for _ in range(m*p)]
        W_sb = [[0]*rank_sb for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
