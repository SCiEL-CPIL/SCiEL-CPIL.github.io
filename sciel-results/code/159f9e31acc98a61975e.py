"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W):
        """Fast numpy-based verification."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k]) * 1.0
                    # Actually need sum of products of all three
                    s = 0.0
                    for r in range(R):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B):
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B):
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    def try_strassen_shared_rank13():
        """
        Attempt rank-13 by merging the (a00+a11) terms from both Strassen blocks.
        
        Standard Strassen for 2x2x2:
        m1 = (a00+a11)(b00+b11)
        m2 = (a10+a11)b00
        m3 = a00(b01-b11)
        m4 = a11(b10-b00)
        m5 = (a00+a01)b11
        m6 = (a10-a00)(b00+b01)
        m7 = (a01-a11)(b10+b11)
        
        c00 = m1 - m4 - m5 + m7
        c10 = m2 + m4
        c01 = m3 + m5
        c11 = m1 - m3 + m6 - m2
        
        For the left block (B cols 0,1) and right block (B cols 2,3),
        we can merge m1_left and m1_right into a single term:
        m1_merged = (a00+a11)(b00+b11+b02+b13)
        
        This saves 1 rank, giving rank 13 total.
        """
        if time.time() > deadline - 10:
            return None
        
        R = 13
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Term 0: merged m1 = (a00+a11)(b00+b11+b02+b13)
        # Contributes to C[0][0], C[1][1], C[0][2], C[1][3]
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1; V[2][0] = 1; V[7][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[4][0] = 1; W[7][0] = 1
        
        # Left block remaining terms (m2-m7):
        # Term 1: m2_left = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1
        
        # Term 2: m3_left = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = -1
        
        # Term 3: m4_left = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1
        
        # Term 4: m5_left = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[2][4] = 1; W[3][4] = -1
        
        # Term 5: m6_left = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1
        
        # Term 6: m7_left = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1; W[1][6] = 1
        
        # Right block remaining terms (m2-m7):
        # Term 7: m2_right = (a10+a11)b02
        U[2][7] = 1; U[3][7] = 1
        V[2][7] = 1
        W[5][7] = 1
        
        # Term 8: m3_right = a00(b03-b13)
        U[0][8] = 1
        V[3][8] = 1; V[7][8] = -1
        W[6][8] = 1; W[7][8] = -1
        
        # Term 9: m4_right = a11(b12-b02)
        U[3][9] = 1
        V[6][9] = 1; V[2][9] = -1
        W[4][9] = -1; W[5][9] = 1
        
        # Term 10: m5_right = (a00+a01)b13
        U[0][10] = 1; U[1][10] = 1
        V[7][10] = 1
        W[6][10] = 1; W[7][10] = -1
        
        # Term 11: m6_right = (a10-a00)(b02+b03)
        U[2][11] = 1; U[0][11] = -1
        V[2][11] = 1; V[3][11] = 1
        W[6][11] = 1
        
        # Term 12: m7_right = (a01-a11)(b12+b13)
        U[1][12] = 1; U[3][12] = -1
        V[6][12] = 1; V[7][12] = 1
        W[4][12] = 1; W[5][12] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    # Try rank-13 first
    result = try_strassen_shared_rank13()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
