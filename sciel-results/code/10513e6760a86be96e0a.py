"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    rank = 32
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    def add_strassen(a_r1, a_r2, a_c1, a_c2, 
                     b_r1, b_r2, b_c1, b_c2,
                     c_r1, c_r2, c_c1, c_c2, col):
        # For C = AB, C[i,j] = sum_k A[i,k]*B[k,j]
        # A submatrix: rows a_r1,a_r2; cols a_c1,a_c2
        # B submatrix: rows b_r1,b_r2; cols b_c1,b_c2
        # C submatrix: rows c_r1,c_r2; cols c_c1,c_c2
        
        # Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
        
        # M1
        U[a_r1*m + a_c1][col] += 1
        U[a_r2*m + a_c2][col] += 1
        V[b_r1*p + b_c1][col] += 1
        V[b_r2*p + b_c2][col] += 1
        W[c_c1*n + c_r1][col] += 1
        W[c_c2*n + c_r2][col] += 1
        
        # M2
        U[a_r2*m + a_c1][col+1] += 1
        U[a_r2*m + a_c2][col+1] += 1
        V[b_r1*p + b_c1][col+1] += 1
        W[c_c1*n + c_r2][col+1] += 1
        W[c_c2*n + c_r2][col+1] -= 1
        
        # M3
        U[a_r1*m + a_c1][col+2] += 1
        V[b_r2*p + b_c1][col+2] += 1
        V[b_r2*p + b_c2][col+2] -= 1
        W[c_c2*n + c_r1][col+2] += 1
        W[c_c2*n + c_r2][col+2] += 1
        
        # M4
        U[a_r2*m + a_c2][col+3] += 1
        V[b_r1*p + b_c2][col+3] += 1
        V[b_r1*p + b_c1][col+3] -= 1
        W[c_c1*n + c_r1][col+3] += 1
        W[c_c1*n + c_r2][col+3] += 1
        
        # M5
        U[a_r1*m + a_c1][col+4] += 1
        U[a_r1*m + a_c2][col+4] += 1
        V[b_r2*p + b_c2][col+4] += 1
        W[c_c1*n + c_r1][col+4] -= 1
        W[c_c2*n + c_r1][col+4] += 1
        
        # M6
        U[a_r2*m + a_c1][col+5] += 1
        U[a_r1*m + a_c1][col+5] -= 1
        V[b_r1*p + b_c1][col+5] += 1
        V[b_r2*p + b_c1][col+5] += 1
        W[c_c2*n + c_r2][col+5] += 1
        
        # M7
        U[a_r1*m + a_c2][col+6] += 1
        U[a_r2*m + a_c2][col+6] -= 1
        V[b_r1*p + b_c2][col+6] += 1
        V[b_r2*p + b_c2][col+6] += 1
        W[c_c1*n + c_r1][col+6] += 1
        
        return col + 7
    
    col = 0
    col = add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, col)
    col = add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, col)
    col = add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, col)
    col = add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, col)
    
    for k in range(4):
        for j in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
    
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
