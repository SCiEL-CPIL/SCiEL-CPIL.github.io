"""Search for rank-minimal exact tensor decomposition of 2x3 @ 3x8 matrix multiply."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145  # safety margin under 150s

    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def strassen_decomposition():
        """
        Decompose 2x2 x 2x2 x 2x2 matrix multiplication using Strassen's algorithm.
        Returns U, V, W as lists of lists of Fractions with rank 7.
        
        Strassen's algorithm:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B12 - B22)
        M4 = A22(B21 - B11)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11
        
        # U: (4, 7), V: (4, 7), W: (4, 7)
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1: A00+A11, B00+B11
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: A10+A11, B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3: A00, B01-B11
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4: A11, B10-B00
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5: A00+A01, B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6: A10-A00, B00+B01
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: A01-A11, B10+B11
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        
        return Uf, Vf, Wf

    def construct_2x3x8():
        rank = 44
        U = [[0]*rank for _ in range(n*m)]  # 6 x 44
        V = [[0]*rank for _ in range(m*p)]  # 24 x 44
        W = [[0]*rank for _ in range(p*n)]  # 16 x 44
        
        # Part 1: A_left (2x2) * B_left (2x8), using Strassen for each 2x2 block of B.
        for s in range(4):
            U_s, V_s, W_s = strassen_decomposition()
            
            base = s * 7
            for la in range(4):
                for c in range(7):
                    if U_s[la][c] != 0:
                        U[la][base+c] = U_s[la][c]
            
            b_map = {0: 2*s, 1: 2*s+1, 2: 8+2*s, 3: 8+2*s+1}
            for lb in range(4):
                for c in range(7):
                    if V_s[lb][c] != 0:
                        V[b_map[lb]][base+c] = V_s[lb][c]
            
            c_map = {0: 4*s, 1: 4*s+2, 2: 4*s+1, 3: 4*s+3}
            for lc in range(4):
                for c in range(7):
                    if W_s[lc][c] != 0:
                        W[c_map[lc]][base+c] = W_s[lc][c]
        
        # Part 2: A_right (2x1) * B_right (1x8), schoolbook.
        base = 28
        for i in range(2):
            for k in range(8):
                c = base + i*8 + k
                U[4+i][c] = 1
                V[16+k][c] = 1
                W[k*2+i][c] = 1
        
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        
        return {"U": Uf, "V": Vf, "W": Wf}

    def verify_decomp(decomp):
        U = decomp["U"]
        V = decomp["V"]
        W = decomp["W"]
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    best = construct_2x3x8()
    best_rank = 44

    if not verify_decomp(best):
        best = schoolbook()
        best_rank = 48

    return best
# EVOLVE-BLOCK-END
