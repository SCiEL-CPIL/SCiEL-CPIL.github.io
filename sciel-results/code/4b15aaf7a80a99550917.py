"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decompose M_{3,5,5} as M_{3,1,1} \otimes M_{1,5,5}
    n1, m1, p1 = 3, 1, 1
    n2, m2, p2 = 1, 5, 5
    
    # Construct T1: M_{3,1,1}
    # Rank R1 = 3
    R1 = n1 * m1 * p1
    U1 = [[0] * R1 for _ in range(n1 * m1)]
    V1 = [[0] * R1 for _ in range(m1 * p1)]
    W1 = [[0] * R1 for _ in range(p1 * n1)]
    
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
                
    # Construct T2: M_{1,5,5}
    # Rank R2 = 25
    R2 = n2 * m2 * p2
    U2 = [[0] * R2 for _ in range(n2 * m2)]
    V2 = [[0] * R2 for _ in range(m2 * p2)]
    W2 = [[0] * R2 for _ in range(p2 * n2)]
    
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
                
    # Combine via Kronecker Product
    R = R1 * R2
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            
            # Iterate over non-zero entries in U1, V1, W1
            # Since they are sparse (one 1 per col), we can just find the indices
            # But for simplicity and correctness, we iterate over all i, j, k
            # Optimization: Pre-compute non-zero indices for T1 and T2
            
            # For T1, non-zero in U1 is at (i1, j1) for r1
            # We need to map r1 to i1, j1, k1
            # In T1 construction:
            # col = i1*m1 + j1*p1 + k1? No.
            # col increments 0, 1, 2.
            # i=0, j=0, k=0 -> U[0][0]=1, V[0][0]=1, W[0][0]=1
            # i=1, j=0, k=0 -> U[1][1]=1, V[0][1]=1, W[0][1]=1
            # i=2, j=0, k=0 -> U[2][2]=1, V[0][2]=1, W[0][2]=1
            
            # For T2, non-zero in U2 is at (i2, j2) for r2
            # r2 = i2*m2 + j2*p2 + k2?
            # i=0, j=0, k=0 -> U[0][0]=1
            # i=0, j=0, k=1 -> U[0][1]=1
            # ...
            
            # To avoid nested loops over all indices for combination,
            # we can just iterate over the full sizes since they are small.
            
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1][r1]
                    if u1_val == 0: continue
                    
                    for k1 in range(p1):
                        v1_val = V1[j1 * p1 + k1][r1]
                        if v1_val == 0: continue
                        w1_val = W1[k1 * n1 + i1][r1]
                        if w1_val == 0: continue
                        
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2_val = U2[i2 * m2 + j2][r2]
                                if u2_val == 0: continue
                                
                                for k2 in range(p2):
                                    v2_val = V2[j2 * p2 + k2][r2]
                                    if v2_val == 0: continue
                                    w2_val = W2[k2 * n2 + i2][r2]
                                    if w2_val == 0: continue
                                    
                                    # Combine indices
                                    # U index: (i1, i2) and (j1, j2) ?
                                    # Matrix A is n x m.
                                    # Index in A is (i, j).
                                    # i = i1 * n2 + i2 ?
                                    # j = j1 * m2 + j2 ?
                                    
                                    i_idx = i1 * n2 + i2
                                    j_idx = j1 * m2 + j2
                                    k_idx = k1 * p2 + k2
                                    
                                    u_idx = i_idx * m + j_idx
                                    v_idx = j_idx * p + k_idx
                                    w_idx = k_idx * n + i_idx
                                    
                                    val = u1_val * u2_val * v1_val * v2_val * w1_val * w2_val
                                    # Actually, the tensor entry is sum of products.
                                    # U, V, W matrices store the coefficients.
                                    # U[u_idx][r] += U1[i1,j1][r1] * U2[i2,j2][r2]
                                    # Wait, the Kronecker product of tensors T1 and T2 is:
                                    # T[i1,i2, j1,j2, k1,k2] = T1[i1,j1,k1] * T2[i2,j2,k2]
                                    # The decomposition is:
                                    # T = sum_r U[:,r] V[:,r] W[:,r]
                                    # So U[u_idx][r] = U1[u1_idx][r1] * U2[u2_idx][r2]
                                    
                                    U[u_idx][r] += u1_val * u2_val
                                    V[v_idx][r] += v1_val * v2_val
                                    W[w_idx][r] += w1_val * w2_val
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
