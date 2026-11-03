import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # Rank 66 construction
    # 6 Strassen blocks (rank 7 each) + 2 Direct blocks (rank 12 each)
    rank = 6 * 7 + 2 * 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen_block(u_idx, v_idx, w_idx, offset):
        """
        u_idx: [u11, u12, u21, u22]
        v_idx: [v11, v12, v21, v22]
        w_idx: [w11, w12, w21, w22]
        """
        # M1 = (A11+A22)(B11+B22)
        U[u_idx[0]][offset] += 1
        U[u_idx[3]][offset] += 1
        V[v_idx[0]][offset] += 1
        V[v_idx[3]][offset] += 1
        W[w_idx[0]][offset] += 1
        W[w_idx[3]][offset] += 1
        W[w_idx[1]][offset] -= 1
        W[w_idx[2]][offset] -= 1
        
        # M2 = (A21+A22)B11
        U[u_idx[2]][offset+1] += 1
        U[u_idx[3]][offset+1] += 1
        V[v_idx[0]][offset+1] += 1
        W[w_idx[2]][offset+1] += 1
        W[w_idx[3]][offset+1] += 1
        
        # M3 = A11(B21-B22)
        U[u_idx[0]][offset+2] += 1
        V[v_idx[2]][offset+2] += 1
        V[v_idx[3]][offset+2] -= 1
        W[w_idx[1]][offset+2] += 1
        W[w_idx[3]][offset+2] += 1
        
        # M4 = A22(B12-B11)
        U[u_idx[3]][offset+3] += 1
        V[v_idx[1]][offset+3] += 1
        V[v_idx[0]][offset+3] -= 1
        W[w_idx[0]][offset+3] += 1
        W[w_idx[2]][offset+3] += 1
        
        # M5 = (A11+A12)B22
        U[u_idx[0]][offset+4] += 1
        U[u_idx[1]][offset+4] += 1
        V[v_idx[3]][offset+4] += 1
        W[w_idx[0]][offset+4] -= 1
        W[w_idx[1]][offset+4] += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[u_idx[2]][offset+5] += 1
        U[u_idx[0]][offset+5] -= 1
        V[v_idx[0]][offset+5] += 1
        V[v_idx[1]][offset+5] += 1
        W[w_idx[2]][offset+5] += 1
        W[w_idx[3]][offset+5] -= 1
        
        # M7 = (A12-A22)(B21+B22)
        U[u_idx[1]][offset+6] += 1
        U[u_idx[3]][offset+6] -= 1
        V[v_idx[2]][offset+6] += 1
        V[v_idx[3]][offset+6] += 1
        W[w_idx[1]][offset+6] -= 1
        W[w_idx[3]][offset+6] += 1

    def add_direct_1x2x6_block(u_idx, v_idx, w_idx, offset):
        """
        u_idx: [u1, u2]
        v_idx: [v11, v12, v13, v14, v15, v16, v21, v22, v23, v24, v25, v26]
        w_idx: [w1, w2, w3, w4, w5, w6]
        """
        for k in range(6):
            # Term j=1
            U[u_idx[0]][offset + k] += 1
            V[v_idx[k]][offset + k] += 1
            W[w_idx[k]][offset + k] += 1
            
            # Term j=2
            U[u_idx[1]][offset + 6 + k] += 1
            V[v_idx[6 + k]][offset + 6 + k] += 1
            W[w_idx[k]][offset + 6 + k] += 1

    # Indices for A (3x4)
    # A11: rows 0,1; cols 0,1 -> U[0], U[1], U[2], U[3]
    # A12: rows 0,1; cols 2,3 -> U[4], U[5], U[6], U[7]
    # A21: row 2; cols 0,1 -> U[8], U[9]
    # A22: row 2; cols 2,3 -> U[10], U[11]
    
    # Indices for B (4x6)
    # BL: rows 0,1; cols 0..5 -> V[0..5] (row 0), V[6..11] (row 1)
    # BR: rows 2,3; cols 0..5 -> V[12..17] (row 2), V[18..23] (row 3)
    
    # Indices for C (3x6)
    # C rows 0,1; cols 0..5 -> W indices: k*3+i
    # C row 2; cols 0..5 -> W indices: k*3+2
    
    # Block 1: A11 * BL[:, 0:2]
    # A11 U: [0, 1, 2, 3] (11, 12, 21, 22)
    # BL V: [0, 1, 6, 7] (11, 12, 21, 22)
    # C W: [0, 1, 3, 4] (11, 12, 21, 22)
    add_strassen_block([0, 1, 2, 3], [0, 1, 6, 7], [0, 1, 3, 4], 0)
    current_rank += 7

    # Block 2: A11 * BL[:, 2:4]
    # A11 U: [0, 1, 2, 3]
    # BL V: [2, 3, 8, 9]
    # C W: [6, 7, 9, 10]
    add_strassen_block([0, 1, 2, 3], [2, 3, 8, 9], [6, 7, 9, 10], 7)
    current_rank += 7

    # Block 3: A11 * BL[:, 4:6]
    # A11 U: [0, 1, 2, 3]
    # BL V: [4, 5, 10, 11]
    # C W: [12, 13, 15, 16]
    add_strassen_block([0, 1, 2, 3], [4, 5, 10, 11], [12, 13, 15, 16], 14)
    current_rank += 7

    # Block 4: A12 * BR[:, 0:2]
    # A12 U: [4, 5, 6, 7]
    # BR V: [12, 13, 18, 19]
    # C W: [0, 1, 3, 4]
    add_strassen_block([4, 5, 6, 7], [12, 13, 18, 19], [0, 1, 3, 4], 21)
    current_rank += 7

    # Block 5: A12 * BR[:, 2:4]
    # A12 U: [4, 5, 6, 7]
    # BR V: [14, 15, 20, 21]
    # C W: [6, 7, 9, 10]
    add_strassen_block([4, 5, 6, 7], [14, 15, 20, 21], [6, 7, 9, 10], 28)
    current_rank += 7

    # Block 6: A12 * BR[:, 4:6]
    # A12 U: [4, 5, 6, 7]
    # BR V: [16, 17, 22, 23]
    # C W: [12, 13, 15, 16]
    add_strassen_block([4, 5, 6, 7], [16, 17, 22, 23], [12, 13, 15, 16], 35)
    current_rank += 7

    # Block 7: A21 * BL (1x2 x 2x6)
    # A21 U: [8, 9]
    # BL V: [0..11]
    # C W: [18, 19, 20, 21, 22, 23] (Row 2 of C, cols 0..5)
    # W indices for row 2: k*3 + 2
    # k=0->2, k=1->5, k=2->8, k=3->11, k=4->14, k=5->17
    add_direct_1x2x6_block([8, 9], list(range(12)), [2, 5, 8, 11, 14, 17], 42)
    current_rank += 12

    # Block 8: A22 * BR (1x2 x 2x6)
    # A22 U: [10, 11]
    # BR V: [12..23]
    # C W: [2, 5, 8, 11, 14, 17]
    add_direct_1x2x6_block([10, 11], list(range(12, 24)), [2, 5, 8, 11, 14, 17], 54)
    current_rank += 12

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
