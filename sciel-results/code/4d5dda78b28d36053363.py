"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    R = 36
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    def add_strassen(u_idx, v_idx, w_idx, col):
        # u_idx: [x11, x12, x21, x22]
        # v_idx: [y11, y12, y21, y22]
        # w_idx: [z11, z12, z21, z22]
        u11, u12, u21, u22 = u_idx
        v11, v12, v21, v22 = v_idx
        w11, w12, w21, w22 = w_idx
        
        # m1
        U[u11][col] += 1
        U[u22][col] += 1
        V[v11][col] += 1
        V[v22][col] += 1
        W[w11][col] += 1
        W[w22][col] += 1
        
        # m2
        U[u21][col] += 1
        U[u22][col] += 1
        V[v11][col] += 1
        W[w21][col] += 1
        W[w22][col] -= 1
        
        # m3
        U[u11][col] += 1
        V[v12][col] += 1
        V[v22][col] -= 1
        W[w12][col] += 1
        W[w22][col] += 1
        
        # m4
        U[u22][col] += 1
        V[v21][col] += 1
        V[v11][col] -= 1
        W[w11][col] += 1
        W[w21][col] += 1
        
        # m5
        U[u11][col] += 1
        U[u12][col] += 1
        V[v22][col] += 1
        W[w11][col] -= 1
        W[w12][col] += 1
        
        # m6
        U[u21][col] += 1
        U[u11][col] -= 1
        V[v11][col] += 1
        V[v12][col] += 1
        W[w22][col] += 1
        
        # m7
        U[u12][col] += 1
        U[u22][col] -= 1
        V[v21][col] += 1
        V[v22][col] += 1
        W[w11][col] += 1

    def add_schoolbook_2x2x1(u_idx, v_idx, w_idx, col):
        # u_idx: [x11, x12, x21, x22]
        # v_idx: [y1, y2]
        # w_idx: [z1, z2]
        u11, u12, u21, u22 = u_idx
        v1, v2 = v_idx
        w1, w2 = w_idx
        
        # x11 * y1 -> z1
        U[u11][col] += 1
        V[v1][col] += 1
        W[w1][col] += 1
        col += 1
        
        # x12 * y2 -> z1
        U[u12][col] += 1
        V[v2][col] += 1
        W[w1][col] += 1
        col += 1
        
        # x21 * y1 -> z2
        U[u21][col] += 1
        V[v1][col] += 1
        W[w2][col] += 1
        col += 1
        
        # x22 * y2 -> z2
        U[u22][col] += 1
        V[v2][col] += 1
        W[w2][col] += 1
        col += 1
        
        return col

    # Indices for A (2x4)
    # A1: i=0,1 j=0,1 -> 0, 1, 4, 5
    # A2: i=0,1 j=2,3 -> 2, 3, 6, 7
    u_A1 = [0, 1, 4, 5]
    u_A2 = [2, 3, 6, 7]
    
    # Indices for B (4x5)
    # B1: j=0,1
    #   B11: k=0,1 -> 0, 1, 5, 6
    #   B12: k=2,3 -> 2, 3, 7, 8
    #   B13: k=4   -> 4, 9
    # B2: j=2,3
    #   B21: k=0,1 -> 10, 11, 15, 16
    #   B22: k=2,3 -> 12, 13, 17, 18
    #   B23: k=4   -> 14, 19
    
    v_B11 = [0, 1, 5, 6]
    v_B12 = [2, 3, 7, 8]
    v_B13 = [4, 9]
    v_B21 = [10, 11, 15, 16]
    v_B22 = [12, 13, 17, 18]
    v_B23 = [14, 19]
    
    # Indices for C (2x5) -> W[k*n + i] = W[k*2 + i]
    # C11: k=0,1 -> 0, 1, 2, 3
    # C12: k=2,3 -> 4, 5, 6, 7
    # C13: k=4   -> 8, 9
    # C21: k=0,1 -> 0, 1, 2, 3
    # C22: k=2,3 -> 4, 5, 6, 7
    # C23: k=4   -> 8, 9
    
    w_C11 = [0, 1, 2, 3]
    w_C12 = [4, 5, 6, 7]
    w_C13 = [8, 9]
    w_C21 = [0, 1, 2, 3]
    w_C22 = [4, 5, 6, 7]
    w_C23 = [8, 9]
    
    col = 0
    # Block 1: A1 * B11
    col += 7
    add_strassen(u_A1, v_B11, w_C11, col-7)
    
    # Block 2: A1 * B12
    col += 7
    add_strassen(u_A1, v_B12, w_C12, col-7)
    
    # Block 3: A1 * B13
    col = add_schoolbook_2x2x1(u_A1, v_B13, w_C13, col)
    
    # Block 4: A2 * B21
    col += 7
    add_strassen(u_A2, v_B21, w_C21, col-7)
    
    # Block 5: A2 * B22
    col += 7
    add_strassen(u_A2, v_B22, w_C22, col-7)
    
    # Block 6: A2 * B23
    col = add_schoolbook_2x2x1(u_A2, v_B23, w_C23, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
