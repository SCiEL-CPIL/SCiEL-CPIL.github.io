"""Program for matrix multiplication 2x3 by 3x5 using Strassen decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 6
    V_dim = m * p  # 15
    W_dim = p * n  # 10

    rank = 17
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]

    # Helper to set a column
    def set_col(c, u_indices, v_indices, w_indices):
        for idx, val in u_indices:
            U[idx][c] += val
        for idx, val in v_indices:
            V[idx][c] += val
        for idx, val in w_indices:
            W[idx][c] += val

    c = 0

    # --- Block 1: Strassen for A[:, 0:2] * B[0:2, 0:2] ---
    # P1 = (A00+A11)(B00+B11) -> Z00, Z11
    set_col(c, [(0, 1), (5, 1)], [(0, 1), (6, 1)], [(0, 1), (3, 1)])
    c += 1
    
    # P2 = (A10+A11)B00 -> Z10, -Z11
    set_col(c, [(3, 1), (5, 1)], [(0, 1)], [(2, 1), (3, -1)])
    c += 1
    
    # P3 = A00(B01-B11) -> Z01, Z11
    set_col(c, [(0, 1)], [(1, 1), (6, -1)], [(1, 1), (3, 1)])
    c += 1
    
    # P4 = A11(B10-B00) -> Z00, Z10
    set_col(c, [(5, 1)], [(5, 1), (0, -1)], [(0, 1), (2, 1)])
    c += 1
    
    # P5 = (A00+A01)B11 -> -Z00, Z01
    set_col(c, [(0, 1), (1, 1)], [(6, 1)], [(0, -1), (1, 1)])
    c += 1
    
    # P6 = (A10-A00)(B00+B01) -> Z11
    set_col(c, [(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
    c += 1
    
    # P7 = (A01-A11)(B10+B11) -> Z00
    set_col(c, [(1, 1), (5, -1)], [(5, 1), (6, 1)], [(0, 1)])
    c += 1

    # --- Block 2: Rank 1 for A[:, 2] * B[2, 0:2] ---
    # P8 = A_i2 * B_2k for k in {0,1}
    set_col(c, [(2, 1), (5, 1)], [(10, 1), (11, 1)], [(0, 1), (1, 1), (2, 1), (3, 1)])
    c += 1

    # --- Block 3: Strassen for A[:, 0:2] * B[0:2, 2:4] ---
    # Note: V indices for B[0:2, 2:4] are:
    # B02 -> V[0*5+2]=2, B03 -> V[3]
    # B12 -> V[1*5+2]=7, B13 -> V[8]
    # W indices for C[:, 2:4] are:
    # C02 -> W[2*2+0]=4, C03 -> W[6]
    # C12 -> W[5], C13 -> W[7]

    # P1' = (A00+A11)(B02+B13) -> Z02, Z13
    set_col(c, [(0, 1), (5, 1)], [(2, 1), (8, 1)], [(4, 1), (7, 1)])
    c += 1
    
    # P2' = (A10+A11)B02 -> Z12, -Z13
    set_col(c, [(3, 1), (5, 1)], [(2, 1)], [(5, 1), (7, -1)])
    c += 1
    
    # P3' = A00(B03-B13) -> Z03, Z13
    set_col(c, [(0, 1)], [(3, 1), (8, -1)], [(6, 1), (7, 1)])
    c += 1
    
    # P4' = A11(B12-B02) -> Z02, Z12
    set_col(c, [(5, 1)], [(7, 1), (2, -1)], [(4, 1), (5, 1)])
    c += 1
    
    # P5' = (A00+A01)B13 -> -Z02, Z03
    set_col(c, [(0, 1), (1, 1)], [(8, 1)], [(4, -1), (6, 1)])
    c += 1
    
    # P6' = (A10-A00)(B02+B03) -> Z13
    set_col(c, [(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)])
    c += 1
    
    # P7' = (A01-A11)(B12+B13) -> Z02
    set_col(c, [(1, 1), (5, -1)], [(7, 1), (8, 1)], [(4, 1)])
    c += 1

    # --- Block 4: Rank 1 for A[:, 0:2] * B[0:2, 4] ---
    # P9 = Sum_j A_ij B_j4
    # U indices for A[:, 0:2]:
    # A00->0, A01->1, A10->3, A11->4
    # V indices for B[0:2, 4]:
    # B04->4, B14->9
    # W indices for C[:, 4]:
    # C04->8, C14->9
    set_col(c, [(0, 1), (1, 1), (3, 1), (4, 1)], [(4, 1), (9, 1)], [(8, 1), (9, 1)])
    c += 1

    # --- Block 5: Rank 1 for A[:, 2] * B[2, 2:4] ---
    # P10 = A_i2 * B_2k for k in {2,3,4}
    # U: A02->2, A12->5
    # V: B22->12, B23->13, B24->14
    # W: C02->4, C12->5, C03->6, C13->7, C04->8, C14->9
    set_col(c, [(2, 1), (5, 1)], [(12, 1), (13, 1), (14, 1)], 
             [(4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)])
    c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
