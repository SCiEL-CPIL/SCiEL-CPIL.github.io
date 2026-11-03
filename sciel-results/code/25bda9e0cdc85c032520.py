"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    n, m, p = 3, 4, 7
    deadline = time.time() + 140  # 140s deadline (10s safety margin)
    
    def validate(U, V, W):
        """Check if decomposition is exact using integer/rational arithmetic."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def try_remove_column(U, V, W, c):
        """Check if column c can be removed while preserving validity."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for cc in range(rank):
                        if cc == c:
                            continue
                        s += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                    if s != 1:
                        return False
        return True

    def reduce_rank(U, V, W):
        """Iteratively remove columns that are redundant."""
        U = [list(row) for row in U]
        V = [list(row) for row in V]
        W = [list(row) for row in W]
        rank = len(U[0])
        
        while rank > 0 and time.time() < deadline - 15:
            removed = False
            for c in range(rank):
                if time.time() > deadline - 15:
                    break
                if try_remove_column(U, V, W, c):
                    U = [row[:c] + row[c+1:] for row in U]
                    V = [row[:c] + row[c+1:] for row in V]
                    W = [row[:c] + row[c+1:] for row in W]
                    rank -= 1
                    removed = True
                    break
            if not removed:
                break
        return U, V, W

    def reduce_rank(U, V, W, max_iter=200):
        """Try to reduce rank by removing linearly dependent columns."""
        U = [list(row) for row in U]
        V = [list(row) for row in V]
        W = [list(row) for row in W]
        rank = len(U[0])
        
        improved = True
        while improved and rank > 0:
            improved = False
            for c in range(rank):
                if time.time() > deadline - 10:
                    break
                # Try removing column c
                U_test = [row[:c] + row[c+1:] for row in U]
                V_test = [row[:c] + row[c+1:] for row in V]
                W_test = [row[:c] + row[c+1:] for row in W]
                if validate(U_test, V_test, W_test):
                    U, V, W = U_test, V_test, W_test
                    rank -= 1
                    improved = True
                    break
            if time.time() > deadline - 10:
                break
        return U, V, W

    # Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication (rank 7)."""
        # n=2, m=2, p=2
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a11+a22)*(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (a21+a22)*b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # M3 = a11*(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = a22*(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a11+a12)*b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (a21-a11)*(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (a12-a22)*(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1
        
        return U, V, W

    def schoolbook(n, m, p):
        """Return U, V, W for n x m x p schoolbook decomposition."""
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    def combine(U1, V1, W1, U2, V2, W2):
        """Combine two decompositions by concatenating columns."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        n1 = len(U1)
        m1 = len(U1[0]) # This is not m, it's rank. Need dimensions.
        # We need to pass dimensions or infer them.
        # U shape (n*m, rank).
        # Let's assume the caller passes compatible shapes.
        U = [row1 + row2 for row1, row2 in zip(U1, U2)]
        V = [row1 + row2 for row1, row2 in zip(V1, V2)]
        W = [row1 + row2 for row1, row2 in zip(W1, W2)]
        return U, V, W

    def construct_2x2x7():
        """Construct 2x2x7 decomposition. Rank <= 25."""
        # Split B (2x7) into B1(2x2), B2(2x2), B3(2x2), B4(2x1)
        # C = A B1 + A B2 + A B3 + A B4
        # U, V, W for 2x2x2 (Strassen)
        U_s, V_s, W_s = strassen_2x2x2()
        
        # We need to map the local indices of the sub-problems to the global indices.
        # Global n=2, m=2, p=7.
        # U size 4. V size 14. W size 14.
        
        U_total = [[0]*25 for _ in range(4)]
        V_total = [[0]*25 for _ in range(14)]
        W_total = [[0]*25 for _ in range(14)]
        
        # Term 0-6: Strassen for B1 (cols 0,1)
        for c in range(7):
            for r in range(4):
                U_total[r][c] = U_s[r][c]
            # V maps j*p+k. j in 0,1. k in 0,1 (local) -> global k in 0,1
            for j in range(2):
                for k in range(2):
                    V_total[j*7+k][c] = V_s[j*2+k][c]
            # W maps k*n+i. k in 0,1 (local) -> global k in 0,1. i in 0,1
            for k in range(2):
                for i in range(2):
                    W_total[k*2+i][c] = W_s[k*2+i][c]
                    
        # Term 7-13: Strassen for B2 (cols 2,3)
        for c in range(7, 14):
            lc = c - 7
            for r in range(4):
                U_total[r][c] = U_s[r][lc]
            for j in range(2):
                for k in range(2):
                    V_total[j*7+(k+2)][c] = V_s[j*2+k][lc]
            for k in range(2):
                for i in range(2):
                    W_total[(k+2)*2+i][c] = W_s[k*2+i][lc]
                    
        # Term 14-20: Strassen for B3 (cols 4,5)
        for c in range(14, 21):
            lc = c - 14
            for r in range(4):
                U_total[r][c] = U_s[r][lc]
            for j in range(2):
                for k in range(2):
                    V_total[j*7+(k+4)][c] = V_s[j*2+k][lc]
            for k in range(2):
                for i in range(2):
                    W_total[(k+4)*2+i][c] = W_s[k*2+i][lc]
                    
        # Term 21-24: Schoolbook for B4 (col 6)
        # 2x2x1: A(2x2) B(2x1) -> C(2x1)
        # Terms:
        # 21: A[0,0]*B[0,6] -> C[0,6]
        # 22: A[0,1]*B[1,6] -> C[0,6]
        # 23: A[1,0]*B[0,6] -> C[1,6]
        # 24: A[1,1]*B[1,6] -> C[1,6]
        
        U_total[0][21] = 1
        V_total[0*7+6][21] = 1
        W_total[6*2+0][21] = 1
        
        U_total[1][22] = 1
        V_total[1*7+6][22] = 1
        W_total[6*2+0][22] = 1
        
        U_total[2][23] = 1
        V_total[0*7+6][23] = 1
        W_total[6*2+1][23] = 1
        
        U_total[3][24] = 1
        V_total[1*7+6][24] = 1
        W_total[6*2+1][24] = 1
        
        return U_total, V_total, W_total

    def construct_1x2x7():
        """Construct 1x2x7 decomposition. Rank 2."""
        # n=1, m=2, p=7
        # U size 2. V size 14. W size 7.
        U = [[0]*2 for _ in range(2)]
        V = [[0]*2 for _ in range(14)]
        W = [[0]*2 for _ in range(7)]
        
        # Term 0: A[0,0] * B[0,k] -> C[0,k]
        U[0][0] = 1
        for k in range(7):
            V[0*7+k][0] = 1
            W[k*1+0][0] = 1
            
        # Term 1: A[0,1] * B[1,k] -> C[0,k]
        U[1][1] = 1
        for k in range(7):
            V[1*7+k][1] = 1
            W[k*1+0][1] = 1
            
        return U, V, W

    def construct_3x2x7():
        """Construct 3x2x7 decomposition. Rank 27."""
        # Split A (3x2) into A1(2x2) and A2(1x2)
        # C = A1 B + A2 B
        # U1, V1, W1 from 2x2x7
        # U2, V2, W2 from 1x2x7
        
        U1, V1, W1 = construct_2x2x7()
        U2, V2, W2 = construct_1x2x7()
        
        # Global n=3, m=2, p=7
        # U size 6. V size 14. W size 21.
        
        U_total = [[0]*27 for _ in range(6)]
        V_total = [[0]*27 for _ in range(14)]
        W_total = [[0]*27 for _ in range(21)]
        
        # Part 1: A1 (rows 0,1 of A) * B
        # U1 indices: 0..3 (i in 0,1; j in 0,1). Global U index i*m+j = i*2+j.
        for r in range(4):
            for c in range(25):
                U_total[r][c] = U1[r][c]
        # V1 indices: j*p+k. Global V index j*7+k. Same.
        for r in range(14):
            for c in range(25):
                V_total[r][c] = V1[r][c]
        # W1 indices: k*n+i. Local n=2. Global n=3.
        # Local W index k*2+i. Global W index k*3+i.
        for k in range(7):
            for i in range(2):
                local_idx = k*2 + i
                global_idx = k*3 + i
                for c in range(25):
                    W_total[global_idx][c] = W1[local_idx][c]
                    
        # Part 2: A2 (row 2 of A) * B
        # U2 indices: 0,1 (i=0, j in 0,1). Local n=1.
        # Global U index: i_global=2, j in 0,1. Index 2*2+j = 4+j.
        # U2 has rank 2. Place in columns 25, 26 of U_total.
        for r in range(2):
            for c in range(2):
                U_total[4+r][25+c] = U2[r][c]
        # V2 indices: j*p+k. Same as V1.
        for r in range(14):
            for c in range(25, 27):
                V_total[r][c] = V2[r][c-25]
        # W2 indices: k*n+i. Local n=1. Global n=3.
        # Local W index k*1+0 = k. Global W index k*3+2.
        for k in range(7):
            local_idx = k
            global_idx = k*3 + 2
            for c in range(25, 27):
                W_total[global_idx][c] = W2[local_idx][c-25]
                
        return U_total, V_total, W_total

    def construct_3x4x7():
        """Construct 3x4x7 decomposition. Rank 54."""
        # Split A (3x4) into A1(3x2) and A2(3x2)
        # Split B (4x7) into B1(2x7) and B2(2x7)
        # C = A1 B1 + A2 B2
        
        U1, V1, W1 = construct_3x2x7()
        U2, V2, W2 = construct_3x2x7()
        
        # Global n=3, m=4, p=7
        # U size 12. V size 28. W size 21.
        
        U_total = [[0]*54 for _ in range(12)]
        V_total = [[0]*54 for _ in range(28)]
        W_total = [[0]*54 for _ in range(21)]
        
        # Part 1: A1 (cols 0,1 of A) * B1 (rows 0,1 of B)
        # U1 indices: i*m_local+j. m_local=2.
        # Global U index: i*4+j.
        # For i in 0..2, j in 0..1:
        # Local index i*2+j.
        for i in range(3):
            for j in range(2):
                local_idx = i*2 + j
                global_idx = i*4 + j
                for c in range(27):
                    U_total[global_idx][c] = U1[local_idx][c]
                    
        # V1 indices: j_local*p+k. j_local in 0,1.
        # Global V index: j_global*7+k. j_global in 0,1.
        # Same indices.
        for r in range(14):
            for c in range(27):
                V_total[r][c] = V1[r][c]
                
        # W1 indices: k*n+i. Same as global.
        for r in range(21):
            for c in range(27):
                W_total[r][c] = W1[r][c]
                
        # Part 2: A2 (cols 2,3 of A) * B2 (rows 2,3 of B)
        # U2 indices: i*m_local+j.
        # Global U index: i*4 + (j+2).
        for i in range(3):
            for j in range(2):
                local_idx = i*2 + j
                global_idx = i*4 + (j+2)
                for c in range(27, 54):
                    U_total[global_idx][c] = U2[local_idx][c-27]
                    
        # V2 indices: j_local*p+k. j_local in 0,1.
        # Global V index: (j_local+2)*7 + k.
        for j_local in range(2):
            for k in range(7):
                local_idx = j_local*7 + k
                global_idx = (j_local+2)*7 + k
                for c in range(27, 54):
                    V_total[global_idx][c] = V2[local_idx][c-27]
                    
        # W2 indices: k*n+i. Same as global.
        for r in range(21):
            for c in range(27, 54):
                W_total[r][c] = W2[r][c-27]
                
        return U_total, V_total, W_total

    # Construct and validate
    U, V, W = construct_3x4x7()
    
    if validate(U, V, W):
        # Try to reduce rank
        U, V, W = reduce_rank(U, V, W)
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
