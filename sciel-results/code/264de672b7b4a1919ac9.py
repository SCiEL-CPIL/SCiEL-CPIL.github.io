"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 6 x 44
    V = [[0]*rank for _ in range(m*p)]  # 24 x 44
    W = [[0]*rank for _ in range(p*n)]  # 16 x 44
    
    col = 0
    
    # U indices: A[i,j] -> U[i*m+j]
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[1,0]=U[3], A[1,1]=U[4], A[1,2]=U[5]
    
    # V indices: B[j,k] -> V[j*p+k]
    # B[0,k]=V[k], B[1,k]=V[8+k], B[2,k]=V[16+k]
    
    # W indices: C[i,k] -> W[k*n+i]
    # C[0,k]=W[2k], C[1,k]=W[2k+1]
    
    for pair in range(4):
        c0 = pair * 2
        c1 = pair * 2 + 1
        
        # m1 = (P00+P11)(Q00+Q11)
        U[0][col] = 1  # P00
        U[4][col] = 1  # P11
        V[c0][col] = 1    # Q00
        V[8+c1][col] = 1  # Q11
        W[c0*2+0][col] = 1   # C[0,c0]
        W[c1*2+1][col] = 1   # C[1,c1]
        col += 1
        
        # m2 = (P10+P11)Q00
        U[3][col] = 1  # P10
        U[4][col] = 1  # P11
        V[c0][col] = 1    # Q00
        W[c0*2+1][col] = 1   # C[1,c0]
        col += 1
        
        # m3 = P00(Q10-Q11)
        U[0][col] = 1  # P00
        V[8+c0][col] = 1   # Q10
        V[8+c1][col] = -1  # Q11
        W[c1*2+0][col] = 1   # C[0,c1]
        W[c1*2+1][col] = 1   # C[1,c1]
        col += 1
        
        # m4 = P11(Q01-Q00)
        U[4][col] = 1  # P11
        V[c1][col] = 1    # Q01
        V[c0][col] = -1   # Q00
        W[c0*2+0][col] = 1   # C[0,c0]
        W[c0*2+1][col] = 1   # C[1,c0]
        col += 1
        
        # m5 = (P00+P01)Q11
        U[0][col] = 1  # P00
        U[1][col] = 1  # P01
        V[8+c1][col] = 1  # Q11
        W[c0*2+0][col] = -1  # C[0,c0]
        W[c1*2+0][col] = 1   # C[0,c1]
        col += 1
        
        # m6 = (P10-P00)(Q00+Q01)
        U[3][col] = 1  # P10
        U[0][col] = -1 # P00
        V[c0][col] = 1    # Q00
        V[c1][col] = 1    # Q01
        W[c1*2+1][col] = 1   # C[1,c1]
        col += 1
        
        # m7 = (P01-P11)(Q10+Q11)
        U[1][col] = 1  # P01
        U[4][col] = -1 # P11
        V[8+c0][col] = 1   # Q10
        V[8+c1][col] = 1   # Q11
        W[c0*2+0][col] = 1   # C[0,c0]
        col += 1
    
    # Part 2: A_right (2x1) * B_bottom (1x8), direct
    for k in range(8):
        U[2][col] = 1  # A[0,2]
        V[16+k][col] = 1  # B[2,k]
        W[k*2+0][col] = 1  # C[0,k]
        col += 1
        U[5][col] = 1  # A[1,2]
        V[16+k][col] = 1  # B[2,k]
        W[k*2+1][col] = 1  # C[1,k]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
