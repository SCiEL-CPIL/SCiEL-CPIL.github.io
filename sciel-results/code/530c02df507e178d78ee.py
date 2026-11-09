import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Helper to create a zero matrix
    def zeros(r, c):
        return [[0] * c for _ in range(r)]

    # Helper to place a block into a larger matrix
    def place_block(dest, src, row_offset, col_offset):
        for i in range(len(src)):
            for j in range(len(src[0])):
                dest[row_offset + i][col_offset + j] = src[i][j]

    # Strassen 2x2x2 decomposition
    # U (4x7), V (4x7), W (4x7)
    # Inputs: a,b,c,d (rows 0-3 of U)
    #         e,f,g,h (rows 0-3 of V)
    # Outputs: c11,c12,c21,c22 (rows 0-3 of W)
    
    def get_strassen_2x2x2():
        U = [
            [1, 0, 1, 0, 1, 1, 0], # a
            [1, 0, 0, 1, 1, 0, 1], # b
            [0, 1, -1, 0, 0, -1, 0], # c
            [0, 1, 0, -1, 0, 0, -1]  # d
        ]
        V = [
            [1, 1, 1, 0, 0, 1, 0], # e
            [1, 1, -1, 0, -1, 1, 0], # f
            [0, 0, 0, 1, 0, 0, 1], # g
            [0, 0, 0, 1, 1, 0, 1]  # h
        ]
        W = [
            [1, -1, 1, 0, 0, 1, 0], # c11
            [0, 0, 1, 0, 1, 0, 0], # c12
            [0, 1, 0, 1, 0, -1, 1], # c21
            [-1, 1, -1, 0, 1, 0, -1] # c22
        ]
        return U, V, W

    # Schoolbook 2x2x1 decomposition
    # U (4x4), V (4x4), W (2x4)
    # Inputs: a,b,c,d
    #         e,f
    # Outputs: c1, c2
    
    def get_schoolbook_2x2x1():
        U = [
            [1, 0, 0, 0], # a
            [0, 1, 0, 0], # b
            [0, 0, 1, 0], # c
            [0, 0, 0, 1]  # d
        ]
        V = [
            [1, 0, 1, 0], # e
            [0, 1, 0, 1], # f
        ]
        W = [
            [1, 0, 1, 0], # c1 = ae + bf
            [0, 1, 0, 1]  # c2 = ce + df
        ]
        return U, V, W

    # Build the full decomposition
    # Rank 50
    rank = 50
    
    U = zeros(n * m, rank) # 8 x 50
    V = zeros(m * p, rank) # 28 x 50
    W = zeros(p * n, rank) # 14 x 50
    
    # Part 1: C[:, 0:4] from A[:, 0:4] * B[0:4, 0:4]
    # This involves 4 Strassen blocks.
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # Block 3: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2] (Added to Block 1)
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4] (Added to Block 2)
    
    U_str, V_str, W_str = get_strassen_2x2x2()
    
    # Block 1: A_rows 0,1; A_cols 0,1 -> U rows 0,1
    #          B_rows 0,1; B_cols 0,1 -> V rows 0,1
    #          C_rows 0,1; C_cols 0,1 -> W rows 0,1
    # Note: U index is i*m + j. i in 0..1, j in 0..1 -> 0,1,2,3.
    #        V index is j*p + k. j in 0..1, k in 0..1 -> 0,1,7,8.
    #        W index is k*n + i. k in 0..1, i in 0..1 -> 0,1,2,3.
    
    # We need to map the local 2x2 indices to global indices.
    # Local U rows: 0(a), 1(b), 2(c), 3(d)
    # Global U rows for A[:, 0:2]:
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    
    # Local V rows: 0(e), 1(f), 2(g), 3(h)
    # Global V rows for B[0:2, 0:2]:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 7
    # j=1, k=1 -> 8
    
    # Local W rows: 0(c11), 1(c12), 2(c21), 3(c22)
    # Global W rows for C[:, 0:2]:
    # k=0, i=0 -> 0
    # k=1, i=0 -> 1
    # k=0, i=1 -> 2
    # k=1, i=1 -> 3
    
    # Mapping for Block 1
    u_map = [0, 1, 2, 3]
    v_map = [0, 1, 7, 8]
    w_map = [0, 1, 2, 3]
    
    col_off = 0
    for l in range(7):
        for i in range(4):
            U[u_map[i]][col_off + l] = U_str[i][l]
            V[v_map[i]][col_off + l] = V_str[i][l]
            W[w_map[i]][col_off + l] = W_str[i][l]
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # A same as Block 1.
    # B rows 0,1; cols 2,3.
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 9
    # j=1, k=3 -> 10
    # C rows 0,1; cols 2,3.
    # k=2, i=0 -> 2
    # k=3, i=0 -> 3
    # k=2, i=1 -> 4
    # k=3, i=1 -> 5
    
    col_off += 7
    u_map = [0, 1, 2, 3]
    v_map = [2, 3, 9, 10]
    w_map = [2, 3, 4, 5]
    
    for l in range(7):
        for i in range(4):
            U[u_map[i]][col_off + l] = U_str[i][l]
            V[v_map[i]][col_off + l] = V_str[i][l]
            W[w_map[i]][col_off + l] = W_str[i][l]
    
    # Block 3: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # A rows 0,1; cols 2,3.
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    # B rows 2,3; cols 0,1.
    # j=2, k=0 -> 14
    # j=2, k=1 -> 15
    # j=3, k=0 -> 21
    # j=3, k=1 -> 22
    # C same as Block 1.
    
    col_off += 7
    u_map = [2, 3, 6, 7]
    v_map = [14, 15, 21, 22]
    w_map = [0, 1, 2, 3]
    
    for l in range(7):
        for i in range(4):
            U[u_map[i]][col_off + l] = U_str[i][l]
            V[v_map[i]][col_off + l] = V_str[i][l]
            W[w_map[i]][col_off + l] = W_str[i][l]
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # A same as Block 3.
    # B rows 2,3; cols 2,3.
    # j=2, k=2 -> 16
    # j=2, k=3 -> 17
    # j=3, k=2 -> 23
    # j=3, k=3 -> 24
    # C same as Block 2.
    
    col_off += 7
    u_map = [2, 3, 6, 7]
    v_map = [16, 17, 23, 24]
    w_map = [2, 3, 4, 5]
    
    for l in range(7):
        for i in range(4):
            U[u_map[i]][col_off + l] = U_str[i][l]
            V[v_map[i]][col_off + l] = V_str[i][l]
            W[w_map[i]][col_off + l] = W_str[i][l]
            
    # Part 2: C[:, 4:7] from A[:, 0:4] * B[0:4, 4:7]
    # This involves 2 blocks of 2x2x3.
    # Block 5: A[:, 0:2] * B[0:2, 4:7] -> C[:, 4:7]
    # Block 6: A[:, 2:4] * B[2:4, 4:7] -> C[:, 4:7]
    
    # Each 2x2x3 is split into 2x2x2 (Strassen) + 2x2x1 (Schoolbook)
    
    U_sb, V_sb, W_sb = get_schoolbook_2x2x1()
    
    # Block 5a: A[:, 0:2] * B[0:2, 4:6] (2x2x2 part)
    # A same as Block 1.
    # B rows 0,1; cols 4,5.
    # j=0, k=4 -> 4
    # j=0, k=5 -> 5
    # j=1, k=4 -> 11
    # j=1, k=5 -> 12
    # C rows 0,1; cols 4,5.
    # k=4, i=0 -> 4
    # k=5, i=0 -> 5
    # k=4, i=1 -> 6
    # k=5, i=1 -> 7
    
    col_off += 7
    u_map = [0, 1, 2, 3]
    v_map = [4, 5, 11, 12]
    w_map = [4, 5, 6, 7]
    
    for l in range(7):
        for i in range(4):
            U[u_map[i]][col_off + l] = U_str[i][l]
            V[v_map[i]][col_off + l] = V_str[i][l]
            W[w_map[i]][col_off + l] = W_str[i][l]
            
    # Block 5b: A[:, 0:2] * B[0:2, 6:7] (2x2x1 part)
    # A same as Block 1.
    # B rows 0,1; col 6.
    # j=0, k=6 -> 6
    # j=1, k=6 -> 13
    # C rows 0,1; col 6.
    # k=6, i=0 -> 6
    # k=6, i=1 -> 8
    
    # V for 2x2x1 has 2 rows.
    # Local V rows: 0(e), 1(f).
    # Global V rows: 6, 13.
    # Local W rows: 0(c1), 1(c2).
    # Global W rows: 6, 8.
    
    col_off += 7
    u_map = [0, 1, 2, 3]
    v_map = [6, 13]
    w_map = [6, 8]
    
    for l in range(4):
        for i in range(4):
            U[u_map[i]][col_off + l] = U_sb[i][l]
        for i in range(2):
            V[v_map[i]][col_off + l] = V_sb[i][l]
            W[w_map[i]][col_off + l] = W_sb[i][l]
            
    # Block 6a: A[:, 2:4] * B[2:4, 4:6] (2x2x2 part)
    # A same as Block 3.
    # B rows 2,3; cols 4,5.
    # j=2, k=4 -> 18
    # j=2, k=5 -> 19
    # j=3, k=4 -> 25
    # j=3, k=5 -> 26
    # C same as Block 5a.
    
    col_off += 4
    u_map = [2, 3, 6, 7]
    v_map = [18, 19, 25, 26]
    w_map = [4, 5, 6, 7]
    
    for l in range(7):
        for i in range(4):
            U[u_map[i]][col_off + l] = U_str[i][l]
            V[v_map[i]][col_off + l] = V_str[i][l]
            W[w_map[i]][col_off + l] = W_str[i][l]
            
    # Block 6b: A[:, 2:4] * B[2:4, 6:7] (2x2x1 part)
    # A same as Block 3.
    # B rows 2,3; col 6.
    # j=2, k=6 -> 20
    # j=3, k=6 -> 27
    # C same as Block 5b.
    
    col_off += 7
    u_map = [2, 3, 6, 7]
    v_map = [20, 27]
    w_map = [6, 8]
    
    for l in range(4):
        for i in range(4):
            U[u_map[i]][col_off + l] = U_sb[i][l]
        for i in range(2):
            V[v_map[i]][col_off + l] = V_sb[i][l]
            W[w_map[i]][col_off + l] = W_sb[i][l]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
