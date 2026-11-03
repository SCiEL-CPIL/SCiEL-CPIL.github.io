import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # M1: (x00+x11)(y00+y11)
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # M2: (x10+x11)y00
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # M3: x00(y01-y11)
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # M4: x11(y10-y00)
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # M5: (x00+x01)y11
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # M6: (x10-x00)(y00+y01)
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # M7: (x01-x11)(y10+y11)
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L B_TL -> C_LL
    # A_L rows: 0,1 (U indices 0,1,4,5 for 2x2 block? No, U is 8x28)
    # U index = i*m + l = i*4 + l
    # A_L is A[:, 0:2] -> i in {0,1}, l in {0,1}
    # U indices: 0, 1, 4, 5
    # B_TL is B[0:2, 0:2] -> l in {0,1}, k in {0,1}
    # V index = l*p + k = l*4 + k
    # V indices: 0, 1, 4, 5
    # C_LL is C[:, 0:2] -> k in {0,1}, i in {0,1}
    # W index = k*n + i = k*2 + i
    # W indices: 0, 1, 2, 3
    
    # Mapping for add_block:
    # X_rows: [x00, x01, x10, x11] global indices
    # Y_rows: [y00, y01, y10, y11] global indices
    # Z_rows: [z00, z01, z10, z11] global indices

    # Block 1
    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 1, 2, 3] # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    # Wait, Z convention: W[k*n+i].
    # C_LL corresponds to C[0,0], C[0,1], C[1,0], C[1,1].
    # C[0,0] is k=0, i=0 -> W[0]
    # C[0,1] is k=1, i=0 -> W[2]
    # C[1,0] is k=0, i=1 -> W[1]
    # C[1,1] is k=1, i=1 -> W[3]
    # So Z1 should be [0, 2, 1, 3] to match x00,x01,x10,x11 order?
    # In add_block, Z_rows[0] is z00, Z_rows[1] is z01, etc.
    # z00 = C[0,0] -> W[0]
    # z01 = C[0,1] -> W[2]
    # z10 = C[1,0] -> W[1]
    # z11 = C[1,1] -> W[3]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    # Block 2: A_R B_BL -> C_LR
    # A_R is A[:, 2:4] -> i in {0,1}, l in {2,3}
    # U indices: 2, 3, 6, 7
    # B_BL is B[2:4, 0:2] -> l in {2,3}, k in {0,1}
    # V indices: 8, 9, 12, 13
    # C_LR is C[:, 0:2] -> same as C_LL
    # W indices: 0, 2, 1, 3
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    # Block 3: A_L B_TR -> C_RL
    # A_L is A[:, 0:2]
    # U indices: 0, 1, 4, 5
    # B_TR is B[0:2, 2:4] -> l in {0,1}, k in {2,3}
    # V indices: 2, 3, 6, 7
    # C_RL is C[:, 2:4] -> k in {2,3}, i in {0,1}
    # W indices:
    # C[0,2] is k=2, i=0 -> W[4]
    # C[0,3] is k=3, i=0 -> W[6]
    # C[1,2] is k=2, i=1 -> W[5]
    # C[1,3] is k=3, i=1 -> W[7]
    # Z3 = [4, 6, 5, 7]
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    # Block 4: A_R B_BR -> C_RR
    # A_R is A[:, 2:4]
    # U indices: 2, 3, 6, 7
    # B_BR is B[2:4, 2:4] -> l in {2,3}, k in {2,3}
    # V indices: 10, 11, 14, 15
    # C_RR is C[:, 2:4]
    # W indices: 4, 6, 5, 7
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    return {"U": U, "V": V, "W": W}
