"""Matrix multiplication 2,4,8 tensor decomposition"""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is n x p
    
    # Rank 56 using Strassen for 2x2x2 blocks
    # Structure:
    # A: 2x4
    # B: 4x8
    # C: 2x8
    #
    # We split B into 4 blocks of 4x2 along the k dimension.
    # Each block computation is a 2x4x2 multiplication.
    # 2x4x2 is sum of two 2x2x2 multiplications (split A's columns).
    # Each 2x2x2 uses Strassen (rank 7).
    # So 2x4x2 uses rank 14.
    # 4 blocks * 14 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # Indices: U is (i, j), V is (j, k), W is (k, i)
    # Note: W index in output is k*n + i
    
    # M1: (a00+a11)(b00+b11) -> C00 +=, C11 +=
    s_u_1 = { (0,0): 1, (1,1): 1 }
    s_v_1 = { (0,0): 1, (1,1): 1 }
    s_w_1 = { (0,0): 1, (1,1): 1 }
    
    # M2: (a10+a11)b00 -> C10 +=, C11 -=
    s_u_2 = { (1,0): 1, (1,1): 1 }
    s_v_2 = { (0,0): 1 }
    s_w_2 = { (0,1): 1, (1,1): -1 }
    
    # M3: a00(b01-b11) -> C01 +=, C11 +=
    s_u_3 = { (0,0): 1 }
    s_v_3 = { (0,1): 1, (1,1): -1 }
    s_w_3 = { (1,0): 1, (1,1): 1 }
    
    # M4: a11(b10-b00) -> C00 +=, C10 +=
    s_u_4 = { (1,1): 1 }
    s_v_4 = { (1,0): 1, (0,0): -1 }
    s_w_4 = { (0,0): 1, (0,1): 1 }
    
    # M5: (a00+a01)b11 -> C00 -=, C01 +=
    s_u_5 = { (0,0): 1, (0,1): 1 }
    s_v_5 = { (1,1): 1 }
    s_w_5 = { (0,0): -1, (1,0): 1 }
    
    # M6: (a10-a00)(b00+b01) -> C11 +=
    s_u_6 = { (1,0): 1, (0,0): -1 }
    s_v_6 = { (0,0): 1, (0,1): 1 }
    s_w_6 = { (1,1): 1 }
    
    # M7: (a01-a11)(b10+b11) -> C00 +=
    s_u_7 = { (0,1): 1, (1,1): -1 }
    s_v_7 = { (1,0): 1, (1,1): 1 }
    s_w_7 = { (0,0): 1 }
    
    strassen_u = [s_u_1, s_u_2, s_u_3, s_u_4, s_u_5, s_u_6, s_u_7]
    strassen_v = [s_v_1, s_v_2, s_v_3, s_v_4, s_v_5, s_v_6, s_v_7]
    strassen_w = [s_w_1, s_w_2, s_w_3, s_w_4, s_w_5, s_w_6, s_w_7]
    
    # Iterate over k-blocks (4 blocks of size 2)
    for t in range(4):
        # Iterate over j-blocks (2 blocks of size 2)
        for s in range(2):
            # Iterate over Strassen multiplications
            for r in range(7):
                # Update U
                for (i_loc, j_loc), val in strassen_u[r].items():
                    i_glob = i_loc
                    j_glob = 2 * s + j_loc
                    U[i_glob * m + j_glob, col] += val
                
                # Update V
                for (j_loc, k_loc), val in strassen_v[r].items():
                    j_glob = 2 * s + j_loc
                    k_glob = 2 * t + k_loc
                    V[j_glob * p + k_glob, col] += val
                
                # Update W
                for (k_loc, i_loc), val in strassen_w[r].items():
                    k_glob = 2 * t + k_loc
                    i_glob = i_loc
                    W[k_glob * n + i_glob, col] += val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
