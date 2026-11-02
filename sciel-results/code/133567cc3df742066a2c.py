DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use rank-19 decomposition for 5x5 matrix multiplication,
    # applied independently for each of the 3 rows.
    # Total rank = 3 * 19 = 57.
    
    def get_5x5_rank19():
        """Return (U, V, W) for 5x5 matmul with rank 19."""
        r = 19
        U = [[0.0]*r for _ in range(25)]
        V = [[0.0]*r for _ in range(25)]
        W = [[0.0]*r for _ in range(25)]
        
        # Terms 0-8: schoolbook for 3x3 (indices 0,1,2)
        for i in range(3):
            for j in range(3):
                t = i*3 + j
                for k in range(3):
                    U[i*5+j][t] += 1.0
                    V[j*5+k][t] += 1.0
                    W[i*5+k][t] += 1.0
        
        # Terms 9-15: Strassen for 2x2 (indices 3,4)
        def g2(l): return l + 3
        # M1 = (A00+A11)(B00+B11)
        t = 9
        U[g2(0)*5+g2(0)][t] = 1.0; U[g2(1)*5+g2(1)][t] = 1.0
        V[g2(0)*5+g2(0)][t] = 1.0; V[g2(1)*5+g2(1)][t] = 1.0
        W[g2(0)*5+g2(0)][t] = 1.0; W[g2(1)*5+g2(1)][t] = 1.0
        # M2 = A11*(B10-B00)
        t = 10
        U[g2(1)*5+g2(1)][t] = 1.0
        V[g2(1)*5+g2(0)][t] = 1.0; V[g2(0)*5+g2(0)][t] = -1.0
        W[g2(1)*5+g2(0)][t] = 1.0; W[g2(1)*5+g2(1)][t] = 1.0
        # M3 = (A00-A10)*(B10+B11)
        t = 11
        U[g2(0)*5+g2(0)][t] = 1.0; U[g2(1)*5+g2(0)][t] = -1.0
        V[g2(1)*5+g2(0)][t] = 1.0; V[g2(1)*5+g2(1)][t] = 1.0
        W[g2(0)*5+g2(0)][t] = 1.0; W[g2(1)*5+g2(0)][t] = 1.0
        # M4 = A00*(B01-B11)
        t = 12
        U[g2(0)*5+g2(0)][t] = 1.0
        V[g2(0)*5+g2(1)][t] = 1.0; V[g2(1)*5+g2(1)][t] = -1.0
        W[g2(0)*5+g2(0)][t] = 1.0; W[g2(0)*5+g2(1)][t] = 1.0
        # M5 = (A01+A10)*(B00+B11)
        t = 13
        U[g2(0)*5+g2(1)][t] = 1.0; U[g2(1)*5+g2(0)][t] = 1.0
        V[g2(0)*5+g2(0)][t] = 1.0; V[g2(1)*5+g2(1)][t] = 1.0
        W[g2(0)*5+g2(1)][t] = 1.0; W[g2(1)*5+g2(0)][t] = 1.0
        # M6 = (A00+A01)*(B11-B10)
        t = 14
        U[g2(0)*5+g2(0)][t] = 1.0; U[g2(0)*5+g2(1)][t] = 1.0
        V[g2(1)*5+g2(1)][t] = 1.0; V[g2(1)*5+g2(0)][t] = -1.0
        W[g2(0)*5+g2(1)][t] = 1.0; W[g2(1)*5+g2(1)][t] = 1.0
        # M7 = (A10+A11)*(B01-B00)
        t = 15
        U[g2(1)*5+g2(0)][t] = 1.0; U[g2(1)*5+g2(1)][t] = 1.0
        V[g2(0)*5+g2(1)][t] = 1.0; V[g2(0)*5+g2(0)][t] = -1.0
        W[g2(0)*5+g2(1)][t] = 1.0; W[g2(1)*5+g2(1)][t] = 1.0
        
        # Terms 16-18: cross terms
        # C[0:3, 3:5] += A[0:3, 3:5] * B[3:5, 3:5] and 
        # C[3:5, 0:3] += A[3:5, 0:3] * B[0:3, 0:3]
        # Use 3 shared terms
        # Term 16: handles part of cross
        t = 16
        U[0*5+3][t] = 1.0; U[1*5+3][t] = 1.0; U[2*5+3][t] = 1.0
        V[3*5+0][t] = 1.0; V[3*5+1][t] = 1.0; V[3*5+2][t] = 1.0
        W[0*5+3][t] = 1.0; W[1*5+3][t] = 1.0; W[2*5+3][t] = 1.0
        W[0*5+4][t] = 1.0; W[1*5+4][t] = 1.0; W[2*5+4][t] = 1.0
        
        # This is getting too complex and likely wrong.
        # Let me just use schoolbook for the remaining parts.
        # Actually, let me abandon this and use a correct but higher rank.
        
        return U, V, W
    
    # Given the difficulty of constructing a correct rank-19 decomposition,
    # use schoolbook rank 25 for 5x5, total rank 75.
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
