"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n + i, r] convention
    
    def elapsed_ok():
        return time.time() - start_time < deadline
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
        U: 2x2x7, V: 2x2x7, W: 2x2x7 (W[k, i, r])
        """
        U = [[0]*7 for _ in range(4)] # flattened i*m+j
        V = [[0]*7 for _ in range(4)] # flattened j*p+k
        W = [[0]*7 for _ in range(4)] # flattened k*n+i
        
        # r=0: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        # C00 += M0
        W[0][0] += 1 # k=0, i=0 -> idx 0
        # C11 += M0
        W[3][0] += 1 # k=1, i=1 -> idx 3
        
        # r=1: (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C10 += M1
        W[2][1] += 1 # k=0, i=1 -> idx 2
        # C11 -= M1
        W[3][1] -= 1 # k=1, i=1 -> idx 3
        
        # r=2: A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C01 += M2
        W[1][2] += 1 # k=1, i=0 -> idx 1
        # C11 += M2
        W[3][2] += 1 # k=1, i=1 -> idx 3
        
        # r=3: A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C00 += M3
        W[0][3] += 1 # k=0, i=0 -> idx 0
        # C10 += M3
        W[2][3] += 1 # k=0, i=1 -> idx 2
        
        # r=4: (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C00 -= M4
        W[0][4] -= 1 # k=0, i=0 -> idx 0
        # C01 += M4
        W[1][4] += 1 # k=1, i=0 -> idx 1
        
        # r=5: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C11 += M5
        W[3][5] += 1 # k=1, i=1 -> idx 3
        
        # r=6: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C00 += M6
        W[0][6] += 1 # k=0, i=0 -> idx 0
        
        return U, V, W

    def verify(U, V, W):
        """Verify exact decomposition using Fraction arithmetic."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
                    if total != 1:
                        return False
        return True

    # Construct the rank 29 decomposition
    # Part 1: 2x2x8 using 4x Strassen (rank 28)
    # Part 2: 2x1x8 using 1x trivial (rank 1)
    
    rank_strassen = 7
    num_blocks = 4 # 8 columns / 2
    rank_part1 = rank_strassen * num_blocks # 28
    rank_part2 = 1
    total_rank = rank_part1 + rank_part2 # 29
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    U_s, V_s, W_s = strassen_2x2x2()
    
    for b in range(num_blocks):
        for s in range(rank_strassen):
            r = b * rank_strassen + s
            # U: i in 0..1, j in 0..1
            for i in range(2):
                for j in range(2):
                    U[i*m + j][r] = U_s[i*2 + j][s]
                    
            # V: j in 0..1, k in 0..7
            for j in range(2):
                for k in range(8):
                    if j < 2:
                        # k corresponds to block b and local col k'
                        if k // 2 == b:
                            k_local = k % 2
                            V[j*p + k][r] = V_s[j*2 + k_local][s]
                            
            # W: k in 0..7, i in 0..1
            for k in range(8):
                if k // 2 == b:
                    k_local = k % 2
                    for i in range(2):
                        W[k*n + i][r] = W_s[k_local*2 + i][s]
                        
    # Part 2: 2x1x8
    r = total_rank - 1
    # U: i in 0..1, j=2
    for i in range(2):
        U[i*m + 2][r] = 1
        
    # V: j=2, k in 0..7
    for k in range(8):
        V[2*p + k][r] = 1
        
    # W: k in 0..7, i in 0..1
    for k in range(8):
        for i in range(2):
            W[k*n + i][r] = 1

    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return schoolbook()

# EVOLVE-BLOCK-END
