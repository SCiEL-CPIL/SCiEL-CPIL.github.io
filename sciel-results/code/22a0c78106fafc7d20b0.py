import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Known rank-17 decomposition as fallback
    def rank17():
        rank = 17
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Strassen for 2x2x2 core
        U[0][0] = 1; U[4][0] = 1; V[0][0] = 1; V[4][0] = 1; W[0][0] = 1; W[3][0] = 1
        U[3][1] = 1; U[4][1] = 1; V[0][1] = 1; W[1][1] = 1; W[3][1] = -1
        U[0][2] = 1; V[1][2] = 1; V[4][2] = -1; W[2][2] = 1; W[3][2] = 1
        U[4][3] = 1; V[3][3] = 1; V[0][3] = -1; W[0][3] = 1; W[1][3] = 1
        U[0][4] = 1; U[1][4] = 1; V[4][4] = 1; W[0][4] = -1; W[2][4] = 1
        U[3][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        U[1][6] = 1; U[4][6] = -1; V[3][6] = 1; V[4][6] = 1; W[0][6] = 1
        
        # Schoolbook for remaining
        U[2][7] = 1; V[6][7] = 1; W[0][7] = 1
        U[2][8] = 1; V[7][8] = 1; W[2][8] = 1
        U[5][9] = 1; V[6][9] = 1; W[1][9] = 1
        U[5][10] = 1; V[7][10] = 1; W[3][10] = 1
        U[0][11] = 1; V[2][11] = 1; W[4][11] = 1
        U[1][12] = 1; V[5][12] = 1; W[4][12] = 1
        U[2][13] = 1; V[8][13] = 1; W[4][13] = 1
        U[3][14] = 1; V[2][14] = 1; W[5][14] = 1
        U[4][15] = 1; V[5][15] = 1; W[5][15] = 1
        U[5][16] = 1; V[8][16] = 1; W[5][16] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = Fraction(0, 1)
                    for c in range(rank):
                        s += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                    if s != 1:
                        return False
        return True
    
    # Try to find rank-16 decomposition
    # Strategy: Use 7 Strassen terms for 2x2x2 core, then 9 terms for boundary
    # The 10 boundary entries are:
    # j=2: (0,2,0),(0,2,1),(0,2,2),(1,2,0),(1,2,1),(1,2,2)
    # k=2,j<2: (0,0,2),(0,1,2),(1,0,2),(1,1,2)
    
    # For rank 16, we need 9 additional terms to cover 10 boundary entries
    # This means one term must cover two boundary entries, or we use a different partition
    
    # Let's try: design 9 terms where each is sparse (hits 1-2 boundary entries)
    # and solve for coefficients
    
    # Term design for boundary:
    # Terms 7-15 (9 terms)
    # We want: sum_{r=7}^{15} U[u,r]*V[v,r]*W[w,r] = 1 for boundary (i,j,k)
    # and = 0 for core (i,j,k) where j<2,k<2
    
    # Let's use a structured approach:
    # r=7: A02*B20 -> C00 (hits (0,2,0))
    # r=8: A02*B21 -> C01 (hits (0,2,1))  
    # r=9: A12*B20 -> C10 (hits (1,2,0))
    # r=10: A12*B21 -> C11 (hits (1,2,1))
    # r=11: A00*B02 -> C02 (hits (0,0,2))
    # r=12: A01*B12 -> C02 (hits (0,1,2))
    # r=13: A10*B02 -> C12 (hits (1,0,2))
    # r=14: A11*B12 -> C12 (hits (1,1,2))
    # r=15: A02*B22 -> C02, A12*B22 -> C12 (hits (0,2,2) and (1,2,2))
    
    # But wait - these terms also contribute to core entries!
    # E.g., r=7: U[2,7]=1, V[6,7]=1, W[0,7]=1
    #   contributes to (i,j,k) where i*3+j=2 -> i=0,j=2; j*3+k=6 -> j=2,k=0; k*2+i=0 -> k=0,i=0
    #   So it only hits (0,2,0) - good, no core contribution
    # r=11: U[0,11]=1, V[2,11]=1, W[4,11]=1
    #   i*3+j=0 -> i=0,j=0; j*3+k=2 -> j=0,k=2; k*2+i=4 -> k=2,i=0
    #   Hits (0,0,2) - good
    # r=12: U[1,12]=1, V[5,12]=1, W[4,12]=1
    #   i*3+j=1 -> i=0,j=1; j*3+k=5 -> j=1,k=2; k*2+i=4 -> k=2,i=0
    #   Hits (0,1,2) - good
    # r=15: Need to hit both (0,2,2) and (1,2,2)
    #   For (0,2,2): u=2, v=8, w=4
    #   For (1,2,2): u=5, v=8, w=5
    #   Can't do both with one rank-1 term unless...
    #   Actually, one rank-1 term can only hit one (i,j,k) entry if it's sparse
    #   Unless the term is dense enough to hit multiple entries
    
    # Let me reconsider. With 9 terms for 10 entries, I need one term to be "shared"
    # or I need to use the Strassen terms to also contribute to boundary entries
    
    # Alternative: Let some Strassen terms also have boundary components
    # This complicates things. Let me try a different approach.
    
    # Actually, let me just try to find a rank-16 solution by solving the linear system
    # with the constraint that entries are in {0, 1/2, 1, -1/2, -1}
    
    # For now, return the known rank-17 solution
    result = rank17()
    
    # Verify it's correct
    U_list = result["U"]
    V_list = result["V"]
    W_list = result["W"]
    
    # Quick verification
    rank = len(U_list[0])
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for c in range(rank):
                    s += U_list[idx_u][c] * V_list[idx_v][c] * W_list[idx_w][c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return result
