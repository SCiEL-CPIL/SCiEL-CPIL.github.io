import numpy as np

def construct():
    n, m, p = 3, 4, 6
    
    # We decompose C = A B
    # A is 3x4, B is 4x6
    # Split A into A_left (3x3) and A_right (3x1)
    # Split B into B_top (3x6) and B_bottom (1x6)
    # C = A_left B_top + A_right B_bottom
    
    # Part 1: A_right (3x1) * B_bottom (1x6) -> 3x6
    # Rank 3 decomposition found:
    # U shape (3*1, 3) = (3, 3)
    # V shape (1*6, 3) = (6, 3)
    # W shape (6*3, 3) = (18, 3)
    # U[i, r] = delta_{i,r}
    # V[k, r] = 1
    # W[k*n + i, r] = delta_{i,r}
    
    U_part2 = np.zeros((3 * 1, 3))
    V_part2 = np.zeros((1 * 6, 3))
    W_part2 = np.zeros((6 * 3, 3))
    
    for i in range(3):
        U_part2[i, i] = 1
        for k in range(6):
            V_part2[k, i] = 1
            W_part2[k * 3 + i, i] = 1
            
    # Part 2: A_left (3x3) * B_top (3x6) -> 3x6
    # Split B_top into B_top_left (3x3) and B_top_right (3x3)
    # C_left = A_left * B_top_left (3x3x3)
    # C_right = A_left * B_top_right (3x3x3)
    
    # Use schoolbook for 3x3x3 (Rank 27)
    # U shape (3*3, 27) = (9, 27)
    # V shape (3*3, 27) = (9, 27)
    # W shape (3*3, 27) = (9, 27)
    
    U_part3 = np.zeros((3 * 3, 27))
    V_part3 = np.zeros((3 * 3, 27))
    W_part3 = np.zeros((3 * 3, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_part3[i * 3 + j, col] = 1
                V_part3[j * 3 + k, col] = 1
                W_part3[k * 3 + i, col] = 1
                col += 1
                
    # We have two such parts: B_top_left and B_top_right
    # The indices for B_top_left correspond to j in 0..2, k in 0..2
    # The indices for B_top_right correspond to j in 0..2, k in 3..5
    # Wait, B is 4x6.
    # B_top is rows 0..2 of B (size 3x6).
    # B_top_left is cols 0..2 of B_top (size 3x3).
    # B_top_right is cols 3..5 of B_top (size 3x3).
    
    # The global indices for B are j*p + k.
    # For B_top_left: j in 0..2, k in 0..2.
    # For B_top_right: j in 0..2, k in 3..5.
    
    # We can reuse the same U_part3 for both, because A_left is the same.
    # But V and W will be different because the indices in B and C are different.
    
    # Let's construct the full U, V, W for the 57 rank decomposition.
    # Total Rank = 27 + 27 + 3 = 57.
    
    rank = 57
    U = np.zeros((3 * 4, rank))
    V = np.zeros((4 * 6, rank))
    W = np.zeros((6 * 3, rank))
    
    # Fill Part 2 (A_right * B_bottom) into columns 0..2
    # A_right is column 3 of A. Index in U: i*4 + 3.
    # B_bottom is row 3 of B. Index in V: 3*6 + k = 18 + k.
    # C is 3x6. Index in W: k*3 + i.
    
    for i in range(3):
        for r in range(3):
            if i == r:
                U[i * 4 + 3, r] = 1
                for k in range(6):
                    V[3 * 6 + k, r] = 1
                    W[k * 3 + i, r] = 1
                    
    # Fill Part 3a (A_left * B_top_left) into columns 3..29
    # A_left is cols 0..2 of A. Index in U: i*4 + j.
    # B_top_left is rows 0..2, cols 0..2 of B. Index in V: j*6 + k.
    # C_left is cols 0..2 of C. Index in W: k*3 + i.
    
    col_offset = 3
    col = col_offset
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 4 + j, col] = 1
                V[j * 6 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
                
    # Fill Part 3b (A_left * B_top_right) into columns 30..56
    # A_left is cols 0..2 of A. Index in U: i*4 + j.
    # B_top_right is rows 0..2, cols 3..5 of B. Index in V: j*6 + (k+3).
    # C_right is cols 3..5 of C. Index in W: (k+3)*3 + i.
    
    col_offset = 30
    col = col_offset
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 4 + j, col] = 1
                V[j * 6 + (k + 3), col] = 1
                W[(k + 3) * 3 + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
