"""Matrix multiplication 2x2x3 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank 9 decomposition: view 2x2x3 as three 2x2x1 multiplications.
    # A (2x2) * B (2x3) = C (2x3)
    # Split B into three columns, each 2x1.
    # For each column, 2x2x1 matrix-vector mult has rank 3.
    # Total rank = 3 * 3 = 9.
    #
    # For 2x2x1: A = [[a,b],[c,d]], v = [x,y], C = [ax+by, cx+dy]
    # Rank 3 decomposition:
    #   term 0: a*x -> C[0]
    #   term 1: b*y -> C[0]  
    #   term 2: (c+d)*y -> C[1] and adjust... 
    # Actually, rank of 2x2x1 is 3. Let me use a known decomposition.
    #
    # C[0] = a*x + b*y
    # C[1] = c*x + d*y
    #
    # 3 bilinear forms:
    #   m0 = a*x  -> contributes to C[0]
    #   m1 = b*y  -> contributes to C[0]
    #   m2 = (c+d)*x  -> contributes to C[1]... no that doesn't work cleanly.
    #
    # Let me think differently. The 2x2x1 tensor T[i,j,k] where i in {0,1} (A row), j in {0,1} (A col), k in {0,1} (v index).
    # T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1.
    #
    # Rank 3 decomposition:
    #   term 0: A[0,0]*v[0] = a*x -> C[0]
    #   term 1: A[0,1]*v[1] = b*y -> C[0]
    #   term 2: (A[1,0]+A[1,1])*v[1]... no.
    #
    # Better: use the fact that we can write:
    #   C[0] = a*x + b*y
    #   C[1] = c*x + d*y
    #
    # Let's use:
    #   m0 = a*x
    #   m1 = b*y  
    #   m2 = (c+d)*(x) ... no, that's not right either.
    #
    # Actually, a clean rank-3 decomposition:
    #   m0 = a*x
    #   m1 = (b+d)*y
    #   m2 = (c+a)*x
    #   C[0] = m0 + m1 - ... this is getting complicated.
    #
    # Let me just use the trivial rank-4 for each and then try to verify rank 9 works.
    # Actually, let me reconsider. The seed has rank 12 (schoolbook).
    # 
    # For the full 2x2x3 tensor, schoolbook gives rank 12.
    # But we can do better by sharing computation across columns of B.
    #
    # Alternative: treat it as A * B where B is 2x3.
    # C[0,:] = A[0,:]*B[0,:] + A[1,:]*B[1,:]  (row 0 of C)
    # C[1,:] = A[0,:]*B[0,:] + A[1,:]*B[1,:]  (row 1 of C)
    #
    # Each row of C is a dot product of length 2, but we have 3 such dot products per row.
    # Total 6 dot products of length 2.
    #
    # Using Strassen-style sharing: For computing multiple dot products of the same vectors,
    # we can share some multiplications.
    #
    # Let me just go with a careful rank-9 construction.
    #
    # Decomposition per column k (k=0,1,2) of B:
    #   We need to compute A * B[:,k] = C[:,k]
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Rank 3 for each 2x2x1:
    #   Use: m0 = A[0,0]*B[0,k]
    #        m1 = A[0,1]*B[1,k]
    #        m2 = (A[1,0]+A[1,1]) * (B[0,k]+B[1,k])
    #   Then C[0,k] = m0 + m1
    #   C[1,k] = m2 - m1 + ... no.
    #
    # Let me look up: 2x2x1 tensor has rank 3. One decomposition:
    #   T = e0⊗e0⊗e0 + e0⊗e1⊗e1 + (e1+e0)⊗(e0+e1)⊗(e0+e1) - e0⊗e0⊗e0 - e0⊗e1⊗e1
    #   = (e1+e0)⊗(e0+e1)⊗(e0+e1)
    #   That's rank 1, which is wrong. The tensor is not rank 1.
    #
    # Let me just construct rank 9 directly with the understanding that it should work:
    # For each column k, use 3 multiplications:
    #   m_{k,0} = A[0,0] * B[0,k]
    #   m_{k,1} = A[0,1] * B[1,k]  
    #   m_{k,2} = (A[1,0]+A[1,1]) * (B[0,k]+B[1,k])
    #   C[0,k] = m_{k,0} + m_{k,1}
    #   C[1,k] = m_{k,2} - m_{k,1} + ... 
    #
    # C[1,k] should be A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # m_{k,2} = A[1,0]*B[0,k] + A[1,0]*B[1,k] + A[1,1]*B[0,k] + A[1,1]*B[1,k]
    # m_{k,2} - A[1,0]*B[1,k] - A[1,1]*B[0,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] ✓
    # But we don't have A[1,0]*B[1,k] or A[1,1]*B[0,k] as separate terms.
    #
    # This approach doesn't cleanly give rank 3 per column.
    #
    # Let me just return the schoolbook rank 12, which is guaranteed correct.
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
# EVOLVE-BLOCK-END
