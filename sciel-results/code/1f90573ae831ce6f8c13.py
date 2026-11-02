"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen 2x2x2x2 decomposition (rank 7)
    # P[i,j], Q[j,k], X[r,c] = sum_m P[r,m] Q[m,c]
    # U[i*2+j, c], V[j*2+k, c], W[r*2+c, c]
    
    def strassen_factors():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (P00+P11)(Q00+Q11) -> X00 += 1, X11 += 1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (P10+P11)Q00 -> X10 += 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = P00(Q10-Q11) -> X01 += 1
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # M4 = P11(Q01-Q00) -> X00 += 1, X10 += 1
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (P00+P01)Q11 -> X00 -= 1, X01 += 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (P10-P00)(Q00+Q01) -> X11 += 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (P01-P11)(Q10+Q11) -> X00 += 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    sU, sV, sW = strassen_factors()
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 8x28
    V = [[0]*rank for _ in range(m*p)]  # 16x28
    W = [[0]*rank for _ in range(p*n)]  # 8x28
    
    def add_strassen(offset, A_col_off, B_row_off, B_col_off, C_col_off):
        for c in range(7):
            # U: P[r,c] = A[r, A_col_off+c], U index = r*m + (A_col_off+c)
            for r in range(2):
                for c2 in range(2):
                    if sU[r*2+c2][c] != 0:
                        a_idx = r * m + (A_col_off + c2)
                        U[a_idx][offset + c] += sU[r*2+c2][c]
            # V: Q[r,c] = B[B_row_off+r, B_col_off+c], V index = (B_row_off+r)*p + (B_col_off+c)
            for r in range(2):
                for c2 in range(2):
                    if sV[r*2+c2][c] != 0:
                        v_idx = (B_row_off + r) * p + (B_col_off + c2)
                        V[v_idx][offset + c] += sV[r*2+c2][c]
            # W: X[r,c] = C[r, C_col_off+c], W index = (C_col_off+c)*n + r
            for r in range(2):
                for c2 in range(2):
                    if sW[r*2+c2][c] != 0:
                        w_idx = (C_col_off + c2) * n + r
                        W[w_idx][offset + c] += sW[r*2+c2][c]
    
    # Four subproblems
    add_strassen(0, 0, 0, 0, 0)    # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
    add_strassen(7, 0, 0, 2, 2)    # A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
    add_strassen(14, 2, 2, 0, 0)   # A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
    add_strassen(21, 2, 2, 2, 2)   # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
