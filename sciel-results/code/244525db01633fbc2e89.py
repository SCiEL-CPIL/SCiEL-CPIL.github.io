"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Rank-49 exact decomposition for 4x4 matrix multiplication
    via tensor product of two 2x2 Strassen rank-7 decompositions."""
    
    # 2x2 Strassen rank-7 factors
    # A indices: 0=a,1=b,2=c,3=d for [[a,b],[c,d]]
    # B indices: 0=e,1=f,2=g,3=h for [[e,f],[g,h]]  
    # C indices via W[2*k+i]: 0=C[0,0],1=C[1,0],2=C[0,1],3=C[1,1]
    
    U2 = [[1,0,0,0,1,0,0],    # a
          [0,0,0,0,1,0,1],    # b
          [0,1,0,0,0,1,0],    # c
          [1,1,0,1,0,-1,-1]]  # d
    V2 = [[1,1,0,0,0,1,0],    # e
          [0,0,1,0,0,1,0],    # f
          [0,0,0,1,0,0,1],    # g
          [1,0,-1,-1,1,0,1]]  # h
    W2 = [[1,0,0,-1,0,1,1],   # C[0,0]
          [0,0,1,0,1,0,0],    # C[1,0]
          [0,1,0,1,0,0,0],    # C[0,1]
          [1,0,-1,0,1,1,0]]   # C[1,1]
    
    # Tensor product to get 4x4 factors
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        u1 = U2[i1*2+j1][r1]
                        v1 = V2[j1*2+k1][r1]
                        w1 = W2[k1*2+i1][r1]
                        if u1 == 0 and v1 == 0 and w1 == 0:
                            continue
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u2 = U2[i2*2+j2][r2]
                                    v2 = V2[j2*2+k2][r2]
                                    w2 = W2[k2*2+i2][r2]
                                    if u2 == 0 and v2 == 0 and w2 == 0:
                                        continue
                                    
                                    u_idx = (i1*2+i2)*2 + (j1*2+j2)
                                    v_idx = (j1*2+j2)*2 + (k1*2+k2)
                                    w_idx = (k1*2+k2)*2 + (i1*2+i2)
                                    
                                    U[u_idx][col] += u1 * u2
                                    V[v_idx][col] += v1 * v2
                                    W[w_idx][col] += w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
