import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We'll use a block decomposition approach.
    # Split m=3 into m1=2 and m2=1.
    # Split p=8 into p1=4 and p2=4.
    #
    # The multiplication tensor T_{i,j,k} = delta for C[i,k] = sum_j A[i,j] B[j,k]
    # We decompose this into 4 sub-problems:
    # 1. (i in 0..1, j in 0..1, k in 0..3): 2x2 * 2x4
    # 2. (i in 0..1, j in 0..1, k in 4..7): 2x2 * 2x4
    # 3. (i in 0..1, j in 2..2, k in 0..3): 2x1 * 1x4
    # 4. (i in 0..1, j in 2..2, k in 4..7): 2x1 * 1x4
    #
    # For 2x2 * 2x4: We can view this as two independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 has rank 7 (Strassen). So 2x2 * 2x4 has rank 7+7 = 14.
    #
    # For 2x1 * 1x4: This is just scaling, rank = 2*4 = 8.
    #
    # Total rank = 2 * 14 + 2 * 8 = 28 + 16 = 44.
    
    # Strassen's 7 rank-1 terms for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B
    #
    # The 7 intermediate products are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-d)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # In tensor form, the rank-1 terms are (u, v, w) where:
    # u is in R^{2*2} (flattened A), v is in R^{2*2} (flattened B), w is in R^{2*2} (flattened C)
    #
    # Flattening convention: A[i,j] -> index i*2+j (row-major)
    # B[j,k] -> index j*2+k (row-major)
    # C[i,k] -> index i*2+k (row-major)
    #
    # M1 = (a+d)(e+h):
    # u1 = [1,0,1,0] (a=1, b=0, c=1, d=0)
    # v1 = [1,1,0,0] (e=1, f=0, g=1, h=0)
    # w1 = [1,0,0,1] (C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=1)
    #
    # M2 = (c+d)e:
    # u2 = [0,0,1,1] (a=0, b=0, c=1, d=1)
    # v2 = [1,0,0,0] (e=1, f=0, g=0, h=0)
    # w2 = [0,0,1,0] (C[0,0]=0, C[0,1]=0, C[1,0]=1, C[1,1]=0)
    #
    # M3 = a(f-h):
    # u3 = [1,0,0,0] (a=1, b=0, c=0, d=0)
    # v3 = [0,1,0,-1] (e=0, f=1, g=0, h=-1)
    # w3 = [0,1,0,0] (C[0,0]=0, C[0,1]=1, C[1,0]=0, C[1,1]=0)
    #
    # M4 = d(g-f):
    # u4 = [0,0,0,1] (a=0, b=0, c=0, d=1)
    # v4 = [0,-1,1,0] (e=0, f=-1, g=1, h=0)
    # w4 = [1,0,0,0] (C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=0)
    #
    # M5 = (a+b)h:
    # u5 = [1,1,0,0] (a=1, b=1, c=0, d=0)
    # v5 = [0,0,0,1] (e=0, f=0, g=0, h=1)
    # w5 = [-1,1,0,0] (C[0,0]=-1, C[0,1]=1, C[1,0]=0, C[1,1]=0)
    #
    # M6 = (c-d)(e+f):
    # u6 = [0,0,1,-1] (a=0, b=0, c=1, d=-1)
    # v6 = [1,1,0,0] (e=1, f=1, g=0, h=0)
    # w6 = [0,0,0,1] (C[0,0]=0, C[0,1]=0, C[1,0]=0, C[1,1]=1)
    #
    # M7 = (b-d)(g+h):
    # u7 = [0,1,0,-1] (a=0, b=1, c=0, d=-1)
    # v7 = [0,0,1,1] (e=0, f=0, g=1, h=1)
    # w7 = [1,0,0,-1] (C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=-1)
    
    strassen_terms = [
        ([1,0,1,0], [1,1,0,0], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [0,-1,1,0], [1,0,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,-1]),
    ]
    
    # Now construct the full decomposition.
    # U has shape (n*m, rank) = (6, 44)
    # V has shape (m*p, rank) = (24, 44)
    # W has shape (p*n, rank) = (16, 44)
    
    rank = 44
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Sub-problem 1: i in 0..1, j in 0..1, k in 0..3 (2x2 * 2x4)
    # This is two 2x2 * 2x2 multiplications: k in 0..1 and k in 2..3
    for k_offset in [0, 2]:
        for (u, v, w) in strassen_terms:
            # u is for A[i,j] with i in 0..1, j in 0..1
            # v is for B[j,k] with j in 0..1, k in k_offset..k_offset+1
            # w is for C[i,k] with i in 0..1, k in k_offset..k_offset+1
            
            # U indices: i*m + j = i*3 + j for i in 0..1, j in 0..1
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                U[i*3 + j, col] = u[idx]
            
            # V indices: j*p + k for j in 0..1, k in k_offset..k_offset+1
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*8 + (k_offset + k), col] = v[idx]
            
            # W indices: k*n + i for k in k_offset..k_offset+1, i in 0..1
            for idx in range(4):
                k = idx // 2
                i = idx % 2
                W[(k_offset + k)*2 + i, col] = w[idx]
            
            col += 1
    
    # Sub-problem 2: i in 0..1, j in 0..1, k in 4..7 (2x2 * 2x4)
    # This is two 2x2 * 2x2 multiplications: k in 4..5 and k in 6..7
    for k_offset in [4, 6]:
        for (u, v, w) in strassen_terms:
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                U[i*3 + j, col] = u[idx]
            
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*8 + (k_offset + k), col] = v[idx]
            
            for idx in range(4):
                k = idx // 2
                i = idx % 2
                W[(k_offset + k)*2 + i, col] = w[idx]
            
            col += 1
    
    # Sub-problem 3: i in 0..1, j in 2..2, k in 0..3 (2x1 * 1x4)
    # Rank 8: for each i in 0..1 and k in 0..3
    for i in range(2):
        for k in range(4):
            U[i*3 + 2, col] = 1
            V[2*8 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
    
    # Sub-problem 4: i in 0..1, j in 2..2, k in 4..7 (2x1 * 1x4)
    for i in range(2):
        for k in range(4, 8):
            U[i*3 + 2, col] = 1
            V[2*8 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
