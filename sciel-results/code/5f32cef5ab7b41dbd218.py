"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import fractions
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 2, 4, 6

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
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

    # Strassen 2x2x2 decomposition
    # M1 = (A11-A22)(B21+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B22)
    # M4 = A22(B22-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M5+M4-M3+M7
    # C12 = M1+M5
    # C21 = M2+M4
    # C22 = M3+M1-M2+M6

    def strassen_2x2x2():
        """Return U,V,W for 2x2x2 Strassen decomposition.
        Indices: a = i*2+j (i,j in {0,1}), b = j*2+k (j,k in {0,1}), c = k*2+i (k,i in {0,1})
        """
        # 7 terms
        U = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
        V = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
        W = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries

        # Term 0: M1 = (A11-A22)(B21+B22)
        # A11=a0, A22=a3, B21=b2, B22=b3
        # C11=c0, C12=c1, C21=c2, C22=c3
        U[0][0] = 1; U[3][0] = -1  # A11 - A22
        V[2][0] = 1; V[3][0] = 1   # B21 + B22
        W[0][0] = 1; W[1][0] = 1   # C11 += M1, C12 += M1

        # Term 1: M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1   # A21 + A22
        V[0][1] = 1               # B11
        W[2][1] = 1; W[3][1] = -1  # C21 += M2, C22 -= M2

        # Term 2: M3 = A11(B11-B22)
        U[0][2] = 1               # A11
        V[0][2] = 1; V[3][2] = -1 # B11 - B22
        W[0][2] = -1; W[3][2] = 1 # C11 -= M3, C22 += M3

        # Term 3: M4 = A22(B22-B11)
        U[3][3] = 1               # A22
        V[3][3] = 1; V[0][3] = -1 # B22 - B11
        W[0][3] = 1; W[2][3] = 1  # C11 += M4, C21 += M4

        # Term 4: M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1  # A11 + A12
        V[3][4] = 1               # B22
        W[0][4] = 1; W[1][4] = 1  # C11 += M5, C12 += M5

        # Term 5: M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1 # A21 - A11
        V[0][5] = 1; V[1][5] = 1  # B11 + B12
        W[3][5] = 1               # C22 += M6

        # Term 6: M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1 # A12 - A22
        V[2][6] = 1; V[3][6] = 1  # B21 + B22
        W[0][6] = 1               # C11 += M7

        return U, V, W

    # Build 2x4x6 from three 2x2x2 Strassen blocks for the 2x2x6 part
    # 2x2x6 = three 2x2x2 blocks: k=0,1; k=2,3; k=4,5
    # For each block, the A indices are the same (i,j in 2x2), 
    # B indices shift by 2 per block, C indices shift by 2 per block.
    
    def build_strassen_2x2x6():
        """Build Strassen-based decomposition for 2x2x6.
        A: 2x2 (indices 0-3), B: 2x6 (indices 0-11), C: 6x2 (indices 0-11)
        c = k*2 + i, so c index for block t (k=2t,2t+1) is 4t+i
        """
        U = [[0]*21 for _ in range(4)]   # 2x2 = 4
        V = [[0]*21 for _ in range(12)]  # 2x6 = 12
        W = [[0]*21 for _ in range(12)]  # 6x2 = 12

        sU, sV, sW = strassen_2x2x2()

        for t in range(3):  # 3 blocks of 2x2x2
            offset = t * 7
            # A indices unchanged
            for a in range(4):
                for r in range(7):
                    U[a][offset + r] = sU[a][r]
            # B indices: block t has j in {0,1}, k in {2t, 2t+1}
            # b = j*6 + k... wait, for 2x2x6: b = j*6 + k? No.
            # In the full 2x4x6: b = j*6 + k where j in 0..3, k in 0..5
            # For 2x2x6 subproblem: j in 0..1, k in 0..5, so b = j*6 + k
            # Block t: k in {2t, 2t+1}
            for j in range(2):
                for k in range(2):
                    b = j * 6 + (2*t + k)  # position in 2x6 B matrix
                    # In the 2x2x2 subproblem, local b = j*2 + k
                    local_b = j * 2 + k
                    for r in range(7):
                        V[b][offset + r] = sV[local_b][r]
            # C indices: c = k*2 + i
            # Block t: k in {2t, 2t+1}, so c in {4t, 4t+1, 4t+2, 4t+3}
            for k in range(2):
                for i in range(2):
                    c = (2*t + k) * 2 + i
                    local_c = k * 2 + i
                    for r in range(7):
                        W[c][offset + r] = sW[local_c][r]

        return U, V, W

    # Now build 2x4x6 from two 2x2x6 blocks
    # Block 1: j=0,1 (first 2 columns of A, first 12 rows of B)
    # Block 2: j=2,3 (last 2 columns of A, last 12 rows of B)
    # A: 2x4, indices a = i*4 + j
    # B: 4x6, indices b = j*6 + k
    # C: 6x2, indices c = k*2 + i (same for both blocks)

    def build_2x4x6_strassen():
        """Build 2x4x6 using Strassen on two 2x2x6 blocks. Rank = 42."""
        U = [[0]*42 for _ in range(8)]   # 2x4 = 8
        V = [[0]*42 for _ in range(24)]  # 4x6 = 24
        W = [[0]*42 for _ in range(12)]  # 6x2 = 12

        U1, V1, W1 = build_strassen_2x2x6()

        # Block 1: j=0,1
        for i in range(2):
            for j in range(2):
                a = i * 4 + j
                local_a = i * 2 + j
                for r in range(21):
                    U[a][r] = U1[local_a][r]
        for j in range(2):
            for k in range(6):
                b = j * 6 + k
                local_b = j * 6 + k  # same indexing in 2x6
                for r in range(21):
                    V[b][r] = V1[local_b][r]
        for c in range(12):
            for r in range(21):
                W[c][r] = W1[c][r]

        # Block 2: j=2,3
        for i in range(2):
            for j in range(2, 4):
                a = i * 4 + j
                local_a = i * 2 + (j - 2)
                for r in range(21):
                    U[a][21 + r] = U1[local_a][r]
        for j in range(2, 4):
            for k in range(6):
                b = j * 6 + k
                local_b = (j - 2) * 6 + k
                for r in range(21):
                    V[b][21 + r] = V1[local_b][r]
        for c in range(12):
            for r in range(21):
                W[c][21 + r] = W1[c][r]

        return {"U": U, "V": V, "W": W}

    best_result = build_2x4x6_strassen()

    # Verify the Strassen construction
    if verify(best_result["U"], best_result["V"], best_result["W"]):
        # Try to improve: attempt to find a lower rank by merging
        # For now, return rank 42
        return best_result

    # Fallback to schoolbook if Strassen construction has a bug
    best_result = schoolbook()
    return best_result

# EVOLVE-BLOCK-END
