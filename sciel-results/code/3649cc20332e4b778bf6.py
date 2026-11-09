"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use Kronecker product of Strassen (2,2,2) rank 7 and schoolbook (1,2,3) rank 6
    # Total rank = 42
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 1, 2, 3
    r1 = 7
    r2 = 6
    rank = r1 * r2
    
    # Strassen decomposition for (2,2,2)
    U1 = [
        [1, 0, 1, 0, 1, -1, 0],   # a: i1=0,j1=0
        [0, 0, 0, 0, 1, 0, 1],    # b: i1=0,j1=1
        [0, 1, 0, 0, 0, 1, 0],    # c: i1=1,j1=0
        [1, 1, 0, 1, 0, 0, -1],   # d: i1=1,j1=1
    ]
    
    V1 = [
        [1, 1, 0, -1, 0, 1, 0],   # e: j1=0,k1=0
        [0, 0, 1, 0, 0, 1, 0],    # f: j1=0,k1=1
        [0, 0, 0, 1, 0, 0, 1],    # g: j1=1,k1=0
        [1, 0, -1, 0, 1, 0, 1],   # h: j1=1,k1=1
    ]
    
    W1 = [
        [1, 0, 0, -1, -1, 0, 1],   # C11: k1=0,i1=0
        [0, 0, 1, 0, 1, 0, 0],     # C12: k1=0,i1=1
        [0, 1, 0, 1, 0, 0, 0],     # C21: k1=1,i1=0
        [1, -1, -1, 0, 0, 1, 0],   # C22: k1=1,i1=1
    ]
    
    # Schoolbook decomposition for (1,2,3)
    U2 = [
        [1, 1, 1, 0, 0, 0],   # j2=0
        [0, 0, 0, 1, 1, 1],   # j2=1
    ]
    
    V2 = [
        [1, 0, 0, 0, 0, 0],   # j2=0,k2=0
        [0, 1, 0, 0, 0, 0],   # j2=0,k2=1
        [0, 0, 1, 0, 0, 0],   # j2=0,k2=2
        [0, 0, 0, 1, 0, 0],   # j2=1,k2=0
        [0, 0, 0, 0, 1, 0],   # j2=1,k2=1
        [0, 0, 0, 0, 0, 1],   # j2=1,k2=2
    ]
    
    W2 = [
        [1, 0, 0, 1, 0, 0],   # k2=0
        [0, 1, 0, 0, 1, 0],   # k2=1
        [0, 0, 1, 0, 0, 1],   # k2=2
    ]
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for rS in range(r1):
        for rB in range(r2):
            r = rS * r2 + rB
            
            for iS in range(n1):
                for jS in range(m1):
                    for jB in range(m2):
                        i = iS
                        j = jS * m2 + jB
                        row_u = i * m + j
                        U[row_u][r] = U1[iS * m1 + jS][rS] * U2[jB][rB]
            
            for jS in range(m1):
                for kS in range(p1):
                    for jB in range(m2):
                        for kB in range(p2):
                            j = jS * m2 + jB
                            k = kS * p2 + kB
                            row_v = j * p + k
                            V[row_v][r] = V1[jS * p1 + kS][rS] * V2[jB * p2 + kB][rB]
            
            for kS in range(p1):
                for iS in range(n1):
                    for kB in range(p2):
                        k = kS * p2 + kB
                        i = iS
                        row_w = k * n + i
                        W[row_w][r] = W1[kS * n1 + iS][rS] * W2[kB][rB]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
