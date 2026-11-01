"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 130.0  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    
    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition: for all i,j,jp,k,kp,ip:
        delta(i,ip)*delta(j,jp)*delta(k,kp) = sum_t U[i*m+j,t]*V[jp*p+k,t]*W[kp*n+ip,t]
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1 if (i == ip and j == jp and k == kp) else 0
                                actual = 0
                                for t in range(rank):
                                    actual += U[i * m + j][t] * V[jp * p + k][t] * W[kp * n + ip][t]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    # Rank-28 construction: Strassen for each of 4 column pairs
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(4):  # 4 column pairs: (0,1), (2,3), (4,5), (6,7)
        base = 7 * c
        k0 = 2 * c      # first column in pair
        k1 = 2 * c + 1  # second column in pair
        
        # s=0: M0 = (A00+A11)(B[0,k0] + B[1,k1])
        # Contributes: +1 to C[0,k0], +1 to C[1,k1]
        U[0][base + 0] = 1   # A[0,0]
        U[3][base + 0] = 1   # A[1,1]
        V[0 * p + k0][base + 0] = 1    # B[0,k0]
        V[1 * p + k1][base + 0] = 1    # B[1,k1]
        W[k0 * n + 0][base + 0] = 1    # C[0,k0]
        W[k1 * n + 1][base + 0] = 1    # C[1,k1]
        
        # s=1: M1 = (A10+A11)B[0,k0]
        # Contributes: +1 to C[1,k0]
        U[2][base + 1] = 1   # A[1,0]
        U[3][base + 1] = 1   # A[1,1]
        V[0 * p + k0][base + 1] = 1    # B[0,k0]
        W[k0 * n + 1][base + 1] = 1    # C[1,k0]
        
        # s=2: M2 = A00(B[1,k0] - B[1,k1])
        # Contributes: +1 to C[0,k1]
        U[0][base + 2] = 1   # A[0,0]
        V[1 * p + k0][base + 2] = 1     # B[1,k0]
        V[1 * p + k1][base + 2] = -1    # B[1,k1]
        W[k1 * n + 0][base + 2] = 1     # C[0,k1]
        
        # s=3: M3 = A11(B[0,k1] - B[0,k0])
        # Contributes: +1 to C[0,k0]
        U[3][base + 3] = 1   # A[1,1]
        V[0 * p + k1][base + 3] = 1     # B[0,k1]
        V[0 * p + k0][base + 3] = -1    # B[0,k0]
        W[k0 * n + 0][base + 3] = 1     # C[0,k0]
        
        # s=4: M4 = (A00+A01)B[1,k1]
        # Contributes: -1 to C[0,k0], +1 to C[1,k0]
        U[0][base + 4] = 1   # A[0,0]
        U[1][base + 4] = 1   # A[0,1]
        V[1 * p + k1][base + 4] = 1     # B[1,k1]
        W[k0 * n + 0][base + 4] = -1    # C[0,k0]
        W[k0 * n + 1][base + 4] = 1     # C[1,k0]
        
        # s=5: M5 = (A01-A10)(B[0,k0] + B[0,k1])
        # Contributes: +1 to C[0,k1]
        U[1][base + 5] = 1   # A[0,1]
        U[2][base + 5] = -1  # A[1,0]
        V[0 * p + k0][base + 5] = 1     # B[0,k0]
        V[0 * p + k1][base + 5] = 1     # B[0,k1]
        W[k1 * n + 0][base + 5] = 1     # C[0,k1]
        
        # s=6: M6 = (A10-A00)(B[1,k1] + B[1,k0])
        # Contributes: +1 to C[0,k0]
        U[2][base + 6] = 1   # A[1,0]
        U[0][base + 6] = -1  # A[0,0]
        V[1 * p + k1][base + 6] = 1     # B[1,k1]
        V[1 * p + k0][base + 6] = 1     # B[1,k0]
        W[k0 * n + 0][base + 6] = 1     # C[0,k0]
    
    # Verify the construction
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
