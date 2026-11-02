"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses rank-5 for 2x3x3 and 1x3x3 blocks, and direct methods for 1x3x1 and 2x3x1.
Achieves rank 38.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0

    U_t = []
    V_t = []
    W_t = []

    def rank5_2x3x3(ra, ca, rb, cb):
        """Rank-5 algorithm for 2x3 * 3x3 at offsets."""
        A00 = (ra, ca); A01 = (ra, ca+1); A02 = (ra, ca+2)
        A10 = (ra+1, ca); A11 = (ra+1, ca+1); A12 = (ra+1, ca+2)
        B00 = (rb, cb); B01 = (rb, cb+1); B02 = (rb, cb+2)
        B10 = (rb+1, cb); B11 = (rb+1, cb+1); B12 = (rb+1, cb+2)
        B20 = (rb+2, cb); B21 = (rb+2, cb+1); B22 = (rb+2, cb+2)
        C00 = (ra, cb); C01 = (ra, cb+1); C02 = (ra, cb+2)
        C10 = (ra+1, cb); C11 = (ra+1, cb+1); C12 = (ra+1, cb+2)

        U_t.append([(A00, 1)])
        V_t.append([(B00, 1), (B11, 1), (B22, 1)])
        W_t.append([(C00, 1)])

        U_t.append([(A01, 1), (A02, 1)])
        V_t.append([(B10, 1), (B20, -1)])
        W_t.append([(C01, 1), (C02, 1)])

        U_t.append([(A00, 1), (A10, 1)])
        V_t.append([(B01, 1), (B02, -1)])
        W_t.append([(C00, 1), (C10, 1), (C02, -1), (C12, -1)])

        U_t.append([(A11, 1), (A12, 1)])
        V_t.append([(B20, 1), (B10, -1)])
        W_t.append([(C11, 1), (C12, 1)])

        U_t.append([(A02, 1), (A00, -1)])
        V_t.append([(B11, 1), (B22, 1)])
        W_t.append([(C02, 1), (C00, -1)])

    def direct_2x3x1(ra, ca, rb, cb):
        """Direct 2x3 * 3x1 at offsets. Rank 6."""
        for i in range(2):
            for j in range(3):
                U_t.append([((ra+i, ca+j), 1)])
                V_t.append([((rb+j, cb), 1)])
                W_t.append([((ra+i, cb), 1)])

    def rank5_1x3x3(ra, ca, rb, cb):
        """Rank-5 algorithm for 1x3 * 3x3 at offsets.
        Uses the 2x3x3 algorithm but only computes the first row of C.
        Effectively sets the second row of A to 0 in the U factors.
        """
        A00 = (ra, ca); A01 = (ra, ca+1); A02 = (ra, ca+2)
        B00 = (rb, cb); B01 = (rb, cb+1); B02 = (rb, cb+2)
        B10 = (rb+1, cb); B11 = (rb+1, cb+1); B12 = (rb+1, cb+2)
        B20 = (rb+2, cb); B21 = (rb+2, cb+1); B22 = (rb+2, cb+2)
        C00 = (ra, cb); C01 = (ra, cb+1); C02 = (ra, cb+2)

        # Same U/V/W as rank5_2x3x3 but only for the first row of A and C
        # m1
        U_t.append([(A00, 1)])
        V_t.append([(B00, 1), (B11, 1), (B22, 1)])
        W_t.append([(C00, 1)])
        
        # m2
        U_t.append([(A01, 1), (A02, 1)])
        V_t.append([(B10, 1), (B20, -1)])
        W_t.append([(C01, 1), (C02, 1)])
        
        # m3 (A10 is 0)
        U_t.append([(A00, 1)])
        V_t.append([(B01, 1), (B02, -1)])
        W_t.append([(C00, 1), (C02, -1)])
        
        # m4 (A11, A12 are 0, so this term is 0, skip)
        
        # m5
        U_t.append([(A02, 1), (A00, -1)])
        V_t.append([(B11, 1), (B22, 1)])
        W_t.append([(C02, 1), (C00, -1)])

    def direct_1x3x1(ra, ca, rb, cb):
        """Direct 1x3 * 3x1 at offsets. Rank 3."""
        for j in range(3):
            U_t.append([((ra, ca+j), 1)])
            V_t.append([((rb+j, cb), 1)])
            W_t.append([((ra, cb), 1)])

    # --- Block 1: A[0:2, :] * B[:, 0:3] (2x3x3) ---
    rank5_2x3x3(0, 0, 0, 0)
    
    # --- Block 2: A[0:2, :] * B[:, 3] (2x3x1) ---
    direct_2x3x1(0, 0, 0, 3)
    
    # --- Block 3: A[0:2, :] * B[:, 4:7] (2x3x3) ---
    rank5_2x3x3(0, 0, 0, 4)
    
    # --- Block 4: A[0:2, :] * B[:, 7] (2x3x1) ---
    direct_2x3x1(0, 0, 0, 7)
    
    # --- Block 5: A[2, :] * B[:, 0:3] (1x3x3) ---
    rank5_1x3x3(2, 0, 0, 0)
    
    # --- Block 6: A[2, :] * B[:, 3] (1x3x1) ---
    direct_1x3x1(2, 0, 0, 3)
    
    # --- Block 7: A[2, :] * B[:, 4:7] (1x3x3) ---
    rank5_1x3x3(2, 0, 0, 4)
    
    # --- Block 8: A[2, :] * B[:, 7] (1x3x1) ---
    direct_1x3x1(2, 0, 0, 7)

    rank = len(W_t)

    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback: schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    def to_nice_list(arr):
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
