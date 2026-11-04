import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank allocation
    # Block 1 (cols 0,1): M322 rank 12
    # Block 2 (cols 2,3): M322 rank 12
    # Block 3 (col 4):    M312 rank 6
    # Block 4 (cols 5,6): M322 rank 12
    # Total rank = 42
    
    rank_total = 42
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    def add_m322(rank_offset, i0, j0, k0):
        # Adds M322 for A[i0:i0+3, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # Rank 12
        # Terms:
        # 0: A00 B00 -> C00
        # 1: A01 B10 -> C01
        # 2: A10 B00 -> C10
        # 3: A11 B10 -> C11
        # 4: A20 B00 -> C20
        # 5: A21 B10 -> C21
        # 6: A00 B01 -> C02
        # 7: A01 B11 -> C03
        # 8: A10 B01 -> C12
        # 9: A11 B11 -> C13
        # 10: A20 B01 -> C22
        # 11: A21 B11 -> C23
        
        u_loc = [[0]*12 for _ in range(6)]
        v_loc = [[0]*12 for _ in range(4)]
        w_loc = [[0]*12 for _ in range(6)]
        
        # r=0: A00 B00
        u_loc[0][0] = 1
        v_loc[0][0] = 1
        w_loc[0][0] = 1 # C00 (k=0, i=0)
        
        # r=1: A01 B10
        u_loc[1][1] = 1
        v_loc[2][1] = 1 # B10 (j=1, k=0)
        w_loc[1][1] = 1 # C01 (k=0, i=0? No, C is 3x2. C[i,k]. C[0,1] is i=0, k=1? No.
        # Wait. C is 3x2. C[i, k].
        # A is 3x2. B is 2x2.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # C[0,0] = A00 B00 + A01 B10.
        # C[0,1] = A00 B01 + A01 B11.
        # C[1,0] = A10 B00 + A11 B10.
        # C[1,1] = A10 B01 + A11 B11.
        # C[2,0] = A20 B00 + A21 B10.
        # C[2,1] = A20 B01 + A21 B11.
        
        # So my terms above were slightly mixed up in description but let's stick to the 12 terms.
        # Term 0: A00 B00. Contributes to C00.
        # Term 1: A01 B10. Contributes to C00.
        # Term 2: A00 B01. Contributes to C01.
        # Term 3: A01 B11. Contributes to C01.
        # Term 4: A10 B00. Contributes to C10.
        # Term 5: A11 B10. Contributes to C10.
        # Term 6: A10 B01. Contributes to C11.
        # Term 7: A11 B11. Contributes to C11.
        # Term 8: A20 B00. Contributes to C20.
        # Term 9: A21 B10. Contributes to C20.
        # Term 10: A20 B01. Contributes to C21.
        # Term 11: A21 B11. Contributes to C21.
        
        # Let's fill u, v, w accordingly.
        # U index: i*m_s + j. m_s=2.
        # A00: 0. A01: 1. A10: 2. A11: 3. A20: 4. A21: 5.
        
        # V index: j*p_s + k. p_s=2.
        # B00: 0. B01: 1. B10: 2. B11: 3.
        
        # W index: k*n_s + i. n_s=3.
        # C00: 0*3+0 = 0.
        # C01: 1*3+0 = 3.
        # C10: 0*3+1 = 1.
        # C11: 1*3+1 = 4.
        # C20: 0*3+2 = 2.
        # C21: 1*3+2 = 5.
        
        # r=0: A00 B00 -> C00
        u_loc[0][0] = 1
        v_loc[0][0] = 1
        w_loc[0][0] = 1
        
        # r=1: A01 B10 -> C00
        u_loc[1][1] = 1
        v_loc[2][1] = 1
        w_loc[0][1] = 1
        
        # r=2: A00 B01 -> C01
        u_loc[0][2] = 1
        v_loc[1][2] = 1
        w_loc[3][2] = 1
        
        # r=3: A01 B11 -> C01
        u_loc[1][3] = 1
        v_loc[3][3] = 1
        w_loc[3][3] = 1
        
        # r=4: A10 B00 -> C10
        u_loc[2][4] = 1
        v_loc[0][4] = 1
        w_loc[1][4] = 1
        
        # r=5: A11 B10 -> C10
        u_loc[3][5] = 1
        v_loc[2][5] = 1
        w_loc[1][5] = 1
        
        # r=6: A10 B01 -> C11
        u_loc[2][6] = 1
        v_loc[1][6] = 1
        w_loc[4][6] = 1
        
        # r=7: A11 B11 -> C11
        u_loc[3][7] = 1
        v_loc[3][7] = 1
        w_loc[4][7] = 1
        
        # r=8: A20 B00 -> C20
        u_loc[4][8] = 1
        v_loc[0][8] = 1
        w_loc[2][8] = 1
        
        # r=9: A21 B10 -> C20
        u_loc[5][9] = 1
        v_loc[2][9] = 1
        w_loc[2][9] = 1
        
        # r=10: A20 B01 -> C21
        u_loc[4][10] = 1
        v_loc[1][10] = 1
        w_loc[5][10] = 1
        
        # r=11: A21 B11 -> C21
        u_loc[5][11] = 1
        v_loc[3][11] = 1
        w_loc[5][11] = 1
        
        # Map to global
        for i_l in range(3):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(12):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(12):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(3):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(12):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*3 + i_l][r]

    def add_m312(rank_offset, i0, j0, k0):
        # Adds M312 for A[i0:i0+3, j0:j0+1] * B[j0:j0+1, k0:k0+2]
        # Rank 6
        # C[i,k] = A[i,0] B[0,k]
        
        u_loc = [[0]*6 for _ in range(3)]
        v_loc = [[0]*6 for _ in range(2)]
        w_loc = [[0]*6 for _ in range(6)]
        
        # r=0: A00 B00 -> C00
        u_loc[0][0] = 1
        v_loc[0][0] = 1
        w_loc[0][0] = 1
        
        # r=1: A10 B00 -> C10
        u_loc[1][1] = 1
        v_loc[0][1] = 1
        w_loc[1][1] = 1
        
        # r=2: A20 B00 -> C20
        u_loc[2][2] = 1
        v_loc[0][2] = 1
        w_loc[2][2] = 1
        
        # r=3: A00 B01 -> C01
        u_loc[0][3] = 1
        v_loc[1][3] = 1
        w_loc[3][3] = 1
        
        # r=4: A10 B01 -> C11
        u_loc[1][4] = 1
        v_loc[1][4] = 1
        w_loc[4][4] = 1
        
        # r=5: A20 B01 -> C21
        u_loc[2][5] = 1
        v_loc[1][5] = 1
        w_loc[5][5] = 1
        
        # Map to global
        for i_l in range(3):
            for j_l in range(1):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(6):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*1 + j_l][r]
        
        for j_l in range(1):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(6):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(3):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(6):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*3 + i_l][r]

    # Build blocks
    # Block 1: cols 0,1 (j=0,1)
    add_m322(0, 0, 0, 0)
    
    # Block 2: cols 2,3 (j=2,3)
    add_m322(12, 0, 2, 0)
    
    # Block 3: col 4 (j=4)
    add_m312(24, 0, 4, 0)
    
    # Block 4: cols 5,6 (j=5,6)
    add_m322(30, 0, 5, 0)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
