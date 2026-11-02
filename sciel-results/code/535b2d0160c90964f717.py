"""
Optimized rank-39 decomposition for M_{2,2,11}.
This is the theoretical minimum rank, as M_{2,2,11} is the direct sum of five M_{2,2,2} and one M_{2,2,1}.
Rank(M_{2,2,2}) = 7 (Strassen).
Rank(M_{2,2,1}) = 4 (Schoolbook).
Total Rank = 5 * 7 + 4 = 39.

Uses integer entries for exactness.
"""

def construct():
    n, m, p = 2, 2, 11
    rank = 39
    
    # Initialize matrices with integers
    # U: (n*m, rank) = (4, 39)
    # V: (m*p, rank) = (22, 39)
    # W: (p*n, rank) = (22, 39)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Process 5 blocks of 2 columns each (k=0..1)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices: B[j,k] -> V[j*p + k]
        v_e = k0        # B[0, k0]
        v_f = k1        # B[0, k1]
        v_g = p + k0    # B[1, k0]
        v_h = p + k1    # B[1, k1]
        
        # W indices: C[i,k] -> W[k*n + i]
        w_c00 = k0 * n      # C[0, k0]
        w_c10 = k0 * n + 1  # C[1, k0]
        w_c01 = k1 * n      # C[0, k1]
        w_c11 = k1 * n + 1  # C[1, k1]
        
        # U indices: A[i,j] -> U[i*m + j]
        # U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        
        # Strassen 7 terms for 2x2x2 block
        
        # m1 = (A00+A11)*(B[0,k0]+B[1,k1]) -> C[0,k0] + C[1,k1]
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (A10+A11)*B[0,k0] -> C[1,k0] - C[1,k1]
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = A00*(B[0,k1]-B[1,k1]) -> C[0,k1] + C[1,k1]
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = A11*(B[1,k0]-B[0,k0]) -> C[0,k0] + C[1,k0]
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (A00+A01)*B[1,k1] -> -C[0,k0] + C[0,k1]
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (A10-A00)*(B[0,k0]+B[0,k1]) -> C[1,k1]
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (A01-A11)*(B[1,k0]+B[1,k1]) -> C[0,k0]
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Single column k=10: schoolbook 4 terms
    k2 = 10
    v_e2 = k2        # B[0, 10]
    v_g2 = p + k2    # B[1, 10]
    w_c02 = k2 * n      # C[0, 10]
    w_c12 = k2 * n + 1  # C[1, 10]
    
    # A00 * B[0,10] -> C[0,10]
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A01 * B[1,10] -> C[0,10]
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A10 * B[0,10] -> C[1,10]
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # A11 * B[1,10] -> C[1,10]
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank, f"Rank mismatch: {r} != {rank}"
    
    return {"U": U, "V": V, "W": W}
