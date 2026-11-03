import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Helper to validate a decomposition
    def validate(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def get_rank28_block_decomposition():
        """
        Constructs a rank-28 decomposition using block-based Strassen + schoolbook.
        This is a reliable baseline that beats the schoolbook rank 30.
        """
        rank = 28
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Helper to add a Strassen block for 2x2 * 2x2
        # A indices: i in 0..1, j in j_offset..j_offset+1
        # B indices: j in j_offset..j_offset+1, k in k_offset..k_offset+1
        # C indices: i in 0..1, k in k_offset..k_offset+1
        def add_strassen_block(rank_offset, j_offset, k_offset):
            # A vars
            a_idx = 0 * m + j_offset
            b_idx = 0 * m + (j_offset + 1)
            c_idx = 1 * m + j_offset
            d_idx = 1 * m + (j_offset + 1)
            
            # B vars
            e_idx = j_offset * p + k_offset
            f_idx = j_offset * p + (k_offset + 1)
            g_idx = (j_offset + 1) * p + k_offset
            h_idx = (j_offset + 1) * p + (k_offset + 1)
            
            # Ranks
            r0 = rank_offset
            r1 = rank_offset + 1
            r2 = rank_offset + 2
            r3 = rank_offset + 3
            r4 = rank_offset + 4
            r5 = rank_offset + 5
            r6 = rank_offset + 6
            
            # U (L_A)
            U[a_idx][r0] += 1; U[d_idx][r0] += 1
            U[c_idx][r1] += 1; U[d_idx][r1] += 1
            U[a_idx][r2] += 1
            U[d_idx][r3] += 1
            U[a_idx][r4] += 1; U[b_idx][r4] += 1
            U[c_idx][r5] += 1; U[a_idx][r5] -= 1
            U[b_idx][r6] += 1; U[d_idx][r6] -= 1
            
            # V (L_B)
            V[e_idx][r0] += 1; V[h_idx][r0] += 1
            V[e_idx][r1] += 1
            V[f_idx][r2] += 1; V[h_idx][r2] -= 1
            V[g_idx][r3] += 1; V[e_idx][r3] -= 1
            V[h_idx][r4] += 1
            V[e_idx][r5] += 1; V[f_idx][r5] += 1
            V[g_idx][r6] += 1; V[h_idx][r6] += 1
            
            # W (Coefficients)
            # C_00 (i=0, k=k_offset) -> idx k_offset*n + 0
            W[k_offset * n + 0][r0] += 1
            W[k_offset * n + 0][r3] += 1
            W[k_offset * n + 0][r4] -= 1
            W[k_offset * n + 0][r6] += 1
            
            # C_01 (i=0, k=k_offset+1) -> idx (k_offset+1)*n + 0
            W[(k_offset + 1) * n + 0][r2] += 1
            W[(k_offset + 1) * n + 0][r4] += 1
            
            # C_10 (i=1, k=k_offset) -> idx k_offset*n + 1
            W[k_offset * n + 1][r1] += 1
            W[k_offset * n + 1][r3] += 1
            
            # C_11 (i=1, k=k_offset+1) -> idx (k_offset+1)*n + 1
            W[(k_offset + 1) * n + 1][r0] += 1
            W[(k_offset + 1) * n + 1][r1] -= 1
            W[(k_offset + 1) * n + 1][r2] += 1
            W[(k_offset + 1) * n + 1][r5] += 1

        # Part 1: A_L (2x2) * B_T1 (2x2) -> C_1 (cols 0,1)
        # A cols 0,1. B rows 0,1 cols 0,1.
        add_strassen_block(0, 0, 0)

        # Part 2: A_L (2x2) * B_T2 (2x2) -> C_2 (cols 2,3)
        # A cols 0,1. B rows 0,1 cols 2,3.
        add_strassen_block(7, 0, 2)

        # Part 3: A_L (2x2) * B_T3 (2x1) -> C_3 (col 4)
        # A cols 0,1. B rows 0,1 col 4.
        # Schoolbook for 2x2 * 2x1 = 4 terms
        # A00*B04 -> C04
        U[0][14] += 1
        V[0*5 + 4][14] += 1
        W[4*2 + 0][14] += 1
        
        # A01*B14 -> C04
        U[1][15] += 1
        V[1*5 + 4][15] += 1
        W[4*2 + 0][15] += 1
        
        # A10*B04 -> C14
        U[3][16] += 1
        V[0*5 + 4][16] += 1
        W[4*2 + 1][16] += 1
        
        # A11*B14 -> C14
        U[4][17] += 1
        V[1*5 + 4][17] += 1
        W[4*2 + 1][17] += 1

        # Part 4: A_R (2x1) * B_B (1x5) -> C_4 (all cols)
        # A col 2. B row 2.
        # Schoolbook for 2x1 * 1x5 = 10 terms
        # A02*B20 -> C00
        U[2][18] += 1
        V[2*5 + 0][18] += 1
        W[0*2 + 0][18] += 1
        
        # A02*B21 -> C01
        U[2][19] += 1
        V[2*5 + 1][19] += 1
        W[1*2 + 0][19] += 1
        
        # A02*B22 -> C02
        U[2][20] += 1
        V[2*5 + 2][20] += 1
        W[2*2 + 0][20] += 1
        
        # A02*B23 -> C03
        U[2][21] += 1
        V[2*5 + 3][21] += 1
        W[3*2 + 0][21] += 1
        
        # A02*B24 -> C04
        U[2][22] += 1
        V[2*5 + 4][22] += 1
        W[4*2 + 0][22] += 1
        
        # A12*B20 -> C10
        U[5][23] += 1
        V[2*5 + 0][23] += 1
        W[0*2 + 1][23] += 1
        
        # A12*B21 -> C11
        U[5][24] += 1
        V[2*5 + 1][24] += 1
        W[1*2 + 1][24] += 1
        
        # A12*B22 -> C12
        U[5][25] += 1
        V[2*5 + 2][25] += 1
        W[2*2 + 1][25] += 1
        
        # A12*B23 -> C13
        U[5][26] += 1
        V[2*5 + 3][26] += 1
        W[3*2 + 1][26] += 1
        
        # A12*B24 -> C14
        U[5][27] += 1
        V[2*5 + 4][27] += 1
        W[4*2 + 1][27] += 1

        return {"U": U, "V": V, "W": W}

    # First, return the reliable rank-28 decomposition
    result = get_rank28_block_decomposition()
    
    # Validate it just in case
    if validate(result["U"], result["V"], result["W"]):
        return result

    # Fallback: Schoolbook (should never happen)
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

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Valid:", validate(res['U'], res['V'], res['W']))
