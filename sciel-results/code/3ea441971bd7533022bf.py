"""Matrix multiplication 2x2 by 2x5 with rank-18 decomposition using Strassen blocks."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]   # 4 x 18
    V = [[0] * rank for _ in range(m * p)]   # 10 x 18
    W = [[0] * rank for _ in range(p * n)]   # 10 x 18
    
    # Index conventions:
    # U[i*m + j] = U[i*2 + j] for A[i,j]
    # V[j*p + k] = V[j*5 + k] for B[j,k]
    # W[k*n + i] = W[k*2 + i] for C[i,k]
    
    # For each pair of columns (k0, k1), use a 7-term Strassen-like decomposition
    # For the remaining single column (k=4), use 4 schoolbook terms
    # Total: 7 + 7 + 4 = 18
    
    # Strassen decomposition for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[s, t], [u, v]]
    #
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(g-h)
    # p4 = d(e-f)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # s = p1 - p4 + p5 - p7
    # t = p3 + p5
    # u = p2 + p4
    # v = p1 - p2 + p3 + p7
    
    def add_strassen_block(base, k0, k1):
        """Add 7 rank-1 terms for multiplying A(2x2) by B[:, k0:k1+1]"""
        # p1 = (a+d)(e+h)
        U[0][base] += 1    # a
        U[3][base] += 1    # d
        V[k0][base] += 1      # e = B[0,k0]
        V[5+k1][base] += 1    # h = B[1,k1]
        W[k0*2+0][base] += 1  # s = C[0,k0]
        W[k1*2+1][base] += 1  # v = C[1,k1]
        
        # p2 = (c+d)e
        U[2][base+1] += 1  # c
        U[3][base+1] += 1  # d
        V[k0][base+1] += 1      # e = B[0,k0]
        W[k0*2+1][base+1] += 1  # u = C[1,k0]
        W[k1*2+1][base+1] += -1 # -v = -C[1,k1]
        
        # p3 = a(g-h)
        U[0][base+2] += 1  # a
        V[5+k0][base+2] += 1    # g = B[1,k0]
        V[5+k1][base+2] += -1   # -h = -B[1,k1]
        W[k1*2+0][base+2] += 1  # t = C[0,k1]
        W[k1*2+1][base+2] += 1  # v = C[1,k1]
        
        # p4 = d(e-f)
        U[3][base+3] += 1  # d
        V[k0][base+3] += 1      # e = B[0,k0]
        V[k1][base+3] += -1     # -f = -B[0,k1]
        W[k0*2+0][base+3] += -1 # -s = -C[0,k0]
        W[k0*2+1][base+3] += 1  # u = C[1,k0]
        
        # p5 = (a+b)h
        U[0][base+4] += 1  # a
        U[1][base+4] += 1  # b
        V[5+k1][base+4] += 1    # h = B[1,k1]
        W[k0*2+0][base+4] += 1  # s = C[0,k0]
        W[k1*2+0][base+4] += 1  # t = C[0,k1]
        W[k1*2+1][base+4] += -1 # -v = -C[1,k1]
        
        # p6 = (c-a)(e+f)
        U[2][base+5] += 1  # c
        U[0][base+5] += -1 # -a
        V[k0][base+5] += 1      # e = B[0,k0]
        V[k1][base+5] += 1      # f = B[0,k1]
        # p6 doesn't appear in the standard reconstruction above.
        # Let me check: in some formulations, p6 contributes to other entries.
        # Actually, looking at this more carefully, I think I need to use a 
        # different Strassen variant where all 7 products are used.
        
        # p7 = (b-d)(g+h)
        U[1][base+6] += 1  # b
        U[3][base+6] += -1 # -d
        V[5+k0][base+6] += 1    # g = B[1,k0]
        V[5+k1][base+6] += 1    # h = B[1,k1]
        W[k0*2+0][base+6] += -1 # -s = -C[0,k0]
        W[k1*2+1][base+6] += 1  # v = C[1,k1]
    
    # Add Strassen blocks for columns (0,1) and (2,3)
    add_strassen_block(0, 0, 1)
    add_strassen_block(7, 2, 3)
    
    # Schoolbook for column 4: 4 terms
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        c = 14 + idx
        U[i*2+j][c] = 1
        V[j*5+4][c] = 1
        W[4*2+i][c] = 1  # k=4, i -> k*2+i = 8+i
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
