"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: 2x1x10 schoolbook (j=0 only)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: 2x2x10 using 5 Strassen 2x2x2 blocks
    # j indices are 1 and 2 (0-indexed: 1, 2)
    # k indices grouped in pairs: (0,5), (1,6), (2,7), (3,8), (4,9)
    for group in range(5):
        k0 = group
        k1 = group + 5
        # Strassen's 7 products for 2x2 * 2x2
        # A: rows i=0,1; cols j=1,2
        # B: rows j=1,2; cols k=k0,k1
        # C: rows i=0,1; cols k=k0,k1
        
        # P1 = (A[0,1]+A[1,2]) * (B[1,k0]+B[2,k1])
        # Wait, let me be more careful with Strassen
        # Standard Strassen:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A10)B11
        # P6 = (A11-A00)(B00+B10)
        # P7 = (A10-A11)(B01+B11)
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # In our indexing: A[i,j] where i in {0,1}, j in {1,2}
        # So A00 = A[0,1], A01 = A[0,2], A10 = A[1,1], A11 = A[1,2]
        # B[j,k] where j in {1,2}, k in {k0,k1}
        # So B00 = B[1,k0], B01 = B[1,k1], B10 = B[2,k0], B11 = B[2,k1]
        # C[i,k] where i in {0,1}, k in {k0,k1}
        # So C00 = C[0,k0], C01 = C[0,k1], C10 = C[1,k0], C11 = C[1,k1]
        
        # P1 = (A[0,1]+A[1,2]) * (B[1,k0]+B[2,k1])
        # U: A[0,1] -> U[0*3+1] = U[1], A[1,2] -> U[1*3+2] = U[5]
        # V: B[1,k0] -> V[1*10+k0] = V[10+k0], B[2,k1] -> V[2*10+k1] = V[20+k1]
        # W: contributes to C[0,k0] (+1), C[0,k1] (0), C[1,k0] (0), C[1,k1] (+1)
        U[1][col] = 1; U[5][col] = 1
        V[10+k0][col] = 1; V[20+k1][col] = 1
        W[k0*2+0][col] = 1; W[k1*2+1][col] = 1
        col += 1
        
        # P2 = (A[1,1]+A[1,2]) * B[1,k0]
        # U: A[1,1] -> U[1*3+1] = U[4], A[1,2] -> U[5]
        # V: B[1,k0] -> V[10+k0]
        # W: contributes to C[1,k0] (+1)
        U[4][col] = 1; U[5][col] = 1
        V[10+k0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # P3 = A[0,1] * (B[1,k1]-B[2,k1])
        # U: A[0,1] -> U[1]
        # V: B[1,k1] -> V[10+k1], B[2,k1] -> V[20+k1]
        # W: contributes to C[0,k1] (+1)
        U[1][col] = 1
        V[10+k1][col] = 1; V[20+k1][col] = -1
        W[k1*2+0][col] = 1
        col += 1
        
        # P4 = A[1,2] * (B[2,k0]-B[1,k0])
        # U: A[1,2] -> U[5]
        # V: B[2,k0] -> V[20+k0], B[1,k0] -> V[10+k0]
        # W: contributes to C[0,k0] (+1), C[1,k0] (+1)
        U[5][col] = 1
        V[20+k0][col] = 1; V[10+k0][col] = -1
        W[k0*2+0][col] = 1; W[k0*2+1][col] = 1
        col += 1
        
        # P5 = (A[0,1]+A[1,1]) * B[2,k1]
        # U: A[0,1] -> U[1], A[1,1] -> U[4]
        # V: B[2,k1] -> V[20+k1]
        # W: contributes to C[0,k1] (+1), C[0,k0] (-1)
        U[1][col] = 1; U[4][col] = 1
        V[20+k1][col] = 1
        W[k1*2+0][col] = 1; W[k0*2+0][col] = -1
        col += 1
        
        # P6 = (A[1,2]-A[0,1]) * (B[1,k0]+B[2,k0])
        # U: A[1,2] -> U[5], A[0,1] -> U[1]
        # V: B[1,k0] -> V[10+k0], B[2,k0] -> V[20+k0]
        # W: contributes to C[1,k1] (+1)
        U[5][col] = 1; U[1][col] = -1
        V[10+k0][col] = 1; V[20+k0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # P7 = (A[1,1]-A[1,2]) * (B[1,k1]+B[2,k1])
        # U: A[1,1] -> U[4], A[1,2] -> U[5]
        # V: B[1,k1] -> V[10+k1], B[2,k1] -> V[20+k1]
        # W: contributes to C[0,k0] (+1), C[0,k1] (0), C[1,k1] (0)
        # Wait, let me recheck C00 = P1 + P4 - P5 + P7, so P7 contributes +1 to C[0,k0]
        U[4][col] = 1; U[5][col] = -1
        V[10+k1][col] = 1; V[20+k1][col] = 1
        W[k0*2+0][col] = 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
