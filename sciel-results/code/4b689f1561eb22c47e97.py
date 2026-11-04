import time
import numpy as np
from fractions import Fraction
import sys

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # The minimal rank for 2x4 by 4x4 matrix multiplication is 28.
    # This is derived from decomposing the problem into four 2x2 by 2x2 multiplications.
    # Each 2x2 by 2x2 multiplication has rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def add_strassen_block(u_offset, v_offset, w_offset, a_rows, b_rows, c_rows):
        """
        Adds a Strassen decomposition for a 2x2 block multiplication.
        a_rows: [A11, A12, A21, A22] indices in U
        b_rows: [B11, B12, B21, B22] indices in V
        c_rows: [C11, C12, C21, C22] indices in W
        """
        A11, A12, A21, A22 = a_rows
        B11, B12, B21, B22 = b_rows
        C11, C12, C21, C22 = c_rows

        # M1
        U[A11][u_offset + 0] += 1
        U[A22][u_offset + 0] += 1
        V[B11][u_offset + 0] += 1
        V[B22][u_offset + 0] += 1
        W[C11][u_offset + 0] += 1
        W[C22][u_offset + 0] += 1

        # M2
        U[A21][u_offset + 1] += 1
        U[A22][u_offset + 1] += 1
        V[B11][u_offset + 1] += 1
        W[C21][u_offset + 1] += 1

        # M3
        U[A11][u_offset + 2] += 1
        V[B12][u_offset + 2] += 1
        V[B22][u_offset + 2] -= 1
        W[C12][u_offset + 2] += 1

        # M4
        U[A22][u_offset + 3] += 1
        V[B21][u_offset + 3] += 1
        V[B11][u_offset + 3] -= 1
        W[C11][u_offset + 3] += 1
        W[C21][u_offset + 3] += 1

        # M5
        U[A11][u_offset + 4] += 1
        U[A12][u_offset + 4] += 1
        V[B22][u_offset + 4] += 1
        W[C11][u_offset + 4] -= 1
        W[C12][u_offset + 4] += 1

        # M6
        U[A21][u_offset + 5] += 1
        U[A11][u_offset + 5] -= 1
        V[B11][u_offset + 5] += 1
        V[B12][u_offset + 5] += 1
        W[C22][u_offset + 5] += 1

        # M7
        U[A12][u_offset + 6] += 1
        U[A22][u_offset + 6] -= 1
        V[B21][u_offset + 6] += 1
        V[B22][u_offset + 6] += 1
        W[C11][u_offset + 6] += 1
        W[C22][u_offset + 6] -= 1

    # Indices for A (2x4):
    # A = [A11 A12 A21 A22] in terms of 1x2 blocks? No.
    # Let's map the 2x4 matrix A to indices 0..7.
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # We split A into two 2x2 blocks: A_L (cols 0,1) and A_R (cols 2,3)
    # A_L indices: [0, 1, 4, 5] corresponding to A11, A12, A21, A22 of the 2x2 block
    # A_R indices: [2, 3, 6, 7]
    
    # Indices for B (4x4):
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    # Row 3: 12, 13, 14, 15
    
    # We split B into four 2x2 blocks:
    # B_TL (rows 0,1, cols 0,1): [0, 1, 4, 5]
    # B_TR (rows 0,1, cols 2,3): [2, 3, 6, 7]
    # B_BL (rows 2,3, cols 0,1): [8, 9, 12, 13]
    # B_BR (rows 2,3, cols 2,3): [10, 11, 14, 15]
    
    # Indices for C (4x2):
    # Col 0: 0, 1
    # Col 1: 2, 3
    # Col 2: 4, 5
    # Col 3: 6, 7
    # C[k,i] index is k*n + i = 2k + i.
    # C[0,0]=0, C[1,0]=1
    # C[0,1]=2, C[1,1]=3
    # C[0,2]=4, C[1,2]=5
    # C[0,3]=6, C[1,3]=7
    
    # We split C into two 2x2 blocks (in terms of rows 0,1 and 2,3):
    # C_L (rows 0,1): [0, 1, 2, 3] -> C11, C12, C21, C22 for the 2x2 block
    #   C11=C[0,0]=0, C12=C[0,1]=2, C21=C[1,0]=1, C22=C[1,1]=3
    #   Order for Strassen: [C11, C12, C21, C22] -> [0, 2, 1, 3]
    # C_R (rows 2,3): [4, 5, 6, 7]
    #   C11=C[0,2]=4, C12=C[0,3]=6, C21=C[1,2]=5, C22=C[1,3]=7
    #   Order for Strassen: [C11, C12, C21, C22] -> [4, 6, 5, 7]

    A_L = [0, 1, 4, 5]
    A_R = [2, 3, 6, 7]
    
    B_TL = [0, 1, 4, 5]
    B_TR = [2, 3, 6, 7]
    B_BL = [8, 9, 12, 13]
    B_BR = [10, 11, 14, 15]
    
    C_L = [0, 2, 1, 3]
    C_R = [4, 6, 5, 7]

    # Block 1: A_L * B_TL -> C_L
    add_strassen_block(0, 0, 0, A_L, B_TL, C_L)
    
    # Block 2: A_L * B_TR -> C_R
    add_strassen_block(7, 7, 7, A_L, B_TR, C_R)
    
    # Block 3: A_R * B_BL -> C_L
    add_strassen_block(14, 14, 14, A_R, B_BL, C_L)
    
    # Block 4: A_R * B_BR -> C_R
    add_strassen_block(21, 21, 21, A_R, B_BR, C_R)

    # Convert to lists of floats (integers are fine)
    U_final = [[float(x) for x in row] for row in U]
    V_final = [[float(x) for x in row] for row in V]
    W_final = [[float(x) for x in row] for row in W]

    return {"U": U_final, "V": V_final, "W": W_final}
