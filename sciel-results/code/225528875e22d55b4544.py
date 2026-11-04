"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Kronecker product structure: M_{n,1,1} \otimes M_{1,m,p}
    # n = n1 * n2, m = m1 * m2, p = p1 * p2
    # Here n1=3, n2=1, m1=1, m2=5, p1=1, p2=5
    n1, n2 = n, 1
    m1, m2 = 1, m
    p1, p2 = 1, p
    
    # Rank of M_{n1, m1, p1} is n1*m1*p1 = 3
    r1 = n1 * m1 * p1
    # Rank of M_{n2, m2, p2} is n2*m2*p2 = 5
    r2 = n2 * m2 * p2
    
    rank = r1 * r2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Generate decomposition for M_{n1, m1, p1}
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p1)]
    W1 = [[0] * r1 for _ in range(p1 * n1)]
    
    col1 = 0
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                U1[i1 * m1 + j1][col1] = 1
                V1[j1 * p1 + k1][col1] = 1
                W1[k1 * n1 + i1][col1] = 1
                col1 += 1
                
    # Generate decomposition for M_{n2, m2, p2}
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p2)]
    W2 = [[0] * r2 for _ in range(p2 * n2)]
    
    col2 = 0
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                U2[i2 * m2 + j2][col2] = 1
                V2[j2 * p2 + k2][col2] = 1
                W2[k2 * n2 + i2][col2] = 1
                col2 += 1
                
    # Combine via Kronecker product
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    # Global indices
                    i = i1 * n2 + i2
                    j = j1 * m2 + j2
                    u_idx = i * m + j
                    
                    for k1 in range(p1):
                        for k2 in range(p2):
                            k = k1 * p2 + k2
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            for r1_idx in range(r1):
                                if U1[i1 * m1 + j1][r1_idx] != 0 and V1[j1 * p1 + k1][r1_idx] != 0 and W1[k1 * n1 + i1][r1_idx] != 0:
                                    for r2_idx in range(r2):
                                        if U2[i2 * m2 + j2][r2_idx] != 0 and V2[j2 * p2 + k2][r2_idx] != 0 and W2[k2 * n2 + i2][r2_idx] != 0:
                                            r = r1_idx * r2 + r2_idx
                                            U[u_idx][r] = 1
                                            V[v_idx][r] = 1
                                            W[w_idx][r] = 1
                                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
