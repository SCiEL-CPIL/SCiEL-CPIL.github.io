"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 8

    # Strassen's 7-term decomposition for 2x2 matrix multiplication
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # We define U (2x2 x 7), V (2x2 x 7), W (2x2 x 7) such that:
    # C[i,j] = sum_r U[i,j,r] * (A * U_factor) ... 
    # Actually, let's use the standard tensor decomposition form:
    # The multiplication tensor T has indices (i, k, j) where C[i,j] = sum_k A[i,k] B[k,j]
    # A rank-r decomposition is: T = sum_r u_r * v_r * w_r
    # where u_r is a vector over (i,k) indices (size 4), 
    #       v_r is a vector over (k,j) indices (size 4),
    #       w_r is a vector over (i,j) indices (size 4).
    # 
    # For 2x2:
    # U shape: (4, 7) -> index is (i*2+k)
    # V shape: (4, 7) -> index is (k*2+j)
    # W shape: (4, 7) -> index is (j*2+i)  [Note: problem says W[k*n+i, r] for general case, 
    #                                         but for 2x2x2 with n=2, p=2, m=2, 
    #                                         W index is (j*n + i) = (j*2 + i)]
    # Wait, the problem statement says:
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # For general n x m * m x p:
    # U is (n*m, rank), index (i*m + k) for A[i,k]
    # V is (m*p, rank), index (k*p + j) for B[k,j]
    # W is (p*n, rank), index (j*n + i) for C[i,j]
    # So for 2x2x2:
    # U index: i*2 + k
    # V index: k*2 + j
    # W index: j*2 + i

    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    U2x2 = np.zeros((4, 7))
    V2x2 = np.zeros((4, 7))
    W2x2 = np.zeros((4, 7))

    # M1: (A00 + A11)(B00 + B11)
    # U: A00->1, A11->1 => U[0*2+0]=1, U[1*2+1]=1
    U2x2[0, 0] = 1; U2x2[3, 0] = 1
    # V: B00->1, B11->1 => V[0*2+0]=1, V[1*2+1]=1
    V2x2[0, 0] = 1; V2x2[3, 0] = 1
    # W: C00->1, C11->1 => W[0*2+0]=1, W[1*2+1]=1
    W2x2[0, 0] = 1; W2x2[3, 0] = 1

    # M2: (A10 + A11)B00
    U2x2[2, 1] = 1; U2x2[3, 1] = 1
    V2x2[0, 1] = 1
    # C10->1, C11->-1 => W[0*2+1]=1, W[1*2+1]=-1
    W2x2[1, 1] = 1; W2x2[3, 1] = -1

    # M3: A00(B01 - B11)
    U2x2[0, 2] = 1
    V2x2[1, 2] = 1; V2x2[3, 2] = -1
    # C01->1, C11->1 => W[1*2+0]=1, W[1*2+1]=1
    W2x2[2, 2] = 1; W2x2[3, 2] = 1

    # M4: A11(B10 - B00)
    U2x2[3, 3] = 1
    V2x2[2, 3] = 1; V2x2[0, 3] = -1
    # C00->1, C10->1 => W[0*2+0]=1, W[0*2+1]=1
    W2x2[0, 3] = 1; W2x2[1, 3] = 1

    # M5: (A00 + A01)B11
    U2x2[0, 4] = 1; U2x2[1, 4] = 1
    V2x2[3, 4] = 1
    # C00->-1, C01->1 => W[0*2+0]=-1, W[1*2+0]=1
    W2x2[0, 4] = -1; W2x2[2, 4] = 1

    # M6: (A10 - A00)(B00 + B01)
    U2x2[2, 5] = 1; U2x2[0, 5] = -1
    V2x2[0, 5] = 1; V2x2[1, 5] = 1
    # C11->1 => W[1*2+1]=1
    W2x2[3, 5] = 1

    # M7: (A01 - A11)(B10 + B11)
    U2x2[1, 6] = 1; U2x2[3, 6] = -1
    V2x2[2, 6] = 1; V2x2[3, 6] = 1
    # C00->1 => W[0*2+0]=1
    W2x2[0, 6] = 1

    # Now for 2x4 * 4x8:
    # We can view the 4x4 middle block as four 2x2 blocks.
    # A is 2x4 = [A1 A2] where A1, A2 are 2x2
    # B is 4x8 = [B1 B2 B3 B4] stacked? No, B is 4x8.
    # Let's partition the indices:
    # A[i, k] where i in {0,1}, k in {0,1,2,3}
    # B[k, j] where k in {0,1,2,3}, j in {0..7}
    # C[i, j] where i in {0,1}, j in {0..7}
    #
    # C[i,j] = sum_{k=0}^3 A[i,k] B[k,j]
    # = A[i,0]B[0,j] + A[i,1]B[1,j] + A[i,2]B[2,j] + A[i,3]B[3,j]
    #
    # Group k in pairs: (0,1) and (2,3)
    # Let A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # Let B_top = B[0:2, :] (2x8), B_bot = B[2:4, :] (2x8)
    # C = A_left * B_top + A_right * B_bot
    #
    # B_top is 2x8, B_bot is 2x8.
    # We can further split B_top into four 2x2 blocks: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # Similarly for B_bot.
    #
    # So C = A_left * [B1 B2 B3 B4] + A_right * [B5 B6 B7 B8]
    # where B1..B8 are all 2x2.
    #
    # C[:, 0:2] = A_left*B1 + A_right*B5
    # C[:, 2:4] = A_left*B2 + A_right*B6
    # C[:, 4:6] = A_left*B3 + A_right*B7
    # C[:, 6:8] = A_left*B4 + A_right*B8
    #
    # Each 2x2 * 2x2 multiplication can be done in rank 7.
    # We have 8 such multiplications, but they share the A matrices (A_left and A_right).
    # This is a sum of 8 rank-7 tensors, but with shared U factors.
    #
    # Total rank = 8 * 7 = 56 if we just concatenate.
    # U will be (8, 56), V will be (32, 56), W will be (16, 56).
    
    rank = 56
    U = np.zeros((n*m, rank))  # (8, 56)
    V = np.zeros((m*p, rank))  # (32, 56)
    W = np.zeros((p*n, rank))  # (16, 56)

    # 8 blocks of 2x2 multiplications
    # Block t (t=0..7) corresponds to:
    #   A_block: 2x2, either A_left or A_right
    #   B_block: 2x2, column j_block of B
    #   C_block: 2x2, column j_block of C
    #
    # t=0: A_left, B1, C_col0
    # t=1: A_right, B5, C_col0
    # t=2: A_left, B2, C_col1
    # t=3: A_right, B6, C_col1
    # t=4: A_left, B3, C_col2
    # t=5: A_right, B7, C_col2
    # t=6: A_left, B4, C_col3
    # t=7: A_right, B8, C_col3

    # For each block t, we use the Strassen decomposition.
    # The Strassen U2x2, V2x2, W2x2 are for local 2x2 indices.
    # We need to map local indices to global indices.

    # A_local[i, k] -> A_global[i, k + a_offset]
    # B_local[k, j] -> B_global[k + b_offset, j + c_offset]
    # C_local[i, j] -> C_global[i, j + c_offset]

    # U global index: (i*m + k) = (i*4 + k)
    # V global index: (k*p + j) = (k*8 + j)
    # W global index: (j*p? No, W index is (j*n + i) = (j*2 + i))
    # Wait, for general case W index is (j*n + i) where j is column of C (0..p-1), i is row of C (0..n-1).
    # Here n=2, p=8. So W index = (j*2 + i) for j in 0..7, i in 0..1. Total 16.

    for t in range(8):
        # Determine A offset, B offset, C offset
        if t % 2 == 0:
            a_offset = 0  # A_left: columns 0,1
        else:
            a_offset = 2  # A_right: columns 2,3

        b_col_block = t // 2  # 0,0,1,1,2,2,3,3
        # B_top or B_bot:
        if t % 4 == 0 or t % 4 == 1:
            # t=0,1 -> B_top (rows 0,1), t=2,3 -> B_top, t=4,5 -> B_bot, t=6,7 -> B_bot
            pass
        # Actually:
        # t=0: A_left, B[0:2, 0:2] -> b_row_offset=0, c_offset=0
        # t=1: A_right, B[2:4, 0:2] -> b_row_offset=2, c_offset=0
        # t=2: A_left, B[0:2, 2:4] -> b_row_offset=0, c_offset=2
        # t=3: A_right, B[2:4, 2:4] -> b_row_offset=2, c_offset=2
        # t=4: A_left, B[0:2, 4:6] -> b_row_offset=0, c_offset=4
        # t=5: A_right, B[2:4, 4:6] -> b_row_offset=2, c_offset=4
        # t=6: A_left, B[0:2, 6:8] -> b_row_offset=0, c_offset=6
        # t=7: A_right, B[2:4, 6:8] -> b_row_offset=2, c_offset=6

        if t < 2:
            b_row_offset = 0 if t == 0 else 2
            c_offset = 0
        elif t < 4:
            b_row_offset = 0 if t == 2 else 2
            c_offset = 2
        elif t < 6:
            b_row_offset = 0 if t == 4 else 2
            c_offset = 4
        else:
            b_row_offset = 0 if t == 6 else 2
            c_offset = 6

        # Now fill in the 7 Strassen terms for this block
        for r in range(7):
            global_r = t * 7 + r

            # U: local (i, k) -> global (i*4 + (k + a_offset))
            for i in range(2):
                for k in range(2):
                    local_idx = i * 2 + k
                    val = U2x2[local_idx, r]
                    if val != 0:
                        global_i = i
                        global_k = k + a_offset
                        U[global_i * 4 + global_k, global_r] += val

            # V: local (k, j) -> global ((k + b_row_offset)*8 + (j + c_offset))
            for k in range(2):
                for j in range(2):
                    local_idx = k * 2 + j
                    val = V2x2[local_idx, r]
                    if val != 0:
                        global_k = k + b_row_offset
                        global_j = j + c_offset
                        V[global_k * 8 + global_j, global_r] += val

            # W: local (i, j) -> global (j*2 + i) but with c_offset for j
            # W local index is (j*2 + i) for local j in 0..1, i in 0..1
            # Global C column is j + c_offset
            # Global W index = (j_global * 2 + i) = ((j + c_offset) * 2 + i)
            for i in range(2):
                for j in range(2):
                    local_idx = j * 2 + i
                    val = W2x2[local_idx, r]
                    if val != 0:
                        global_j = j + c_offset
                        global_i = i
                        W[global_j * 2 + global_i, global_r] += val

    # Convert to lists for the return value
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
