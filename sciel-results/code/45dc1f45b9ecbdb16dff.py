import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x4)
    # A[i, j] -> i*m + j
    # A1: [0,0], [0,1], [1,0], [1,1] -> 0, 1, 4, 5
    # A2: [0,2], [0,3], [1,2], [1,3] -> 2, 3, 6, 7
    
    # Indices for B (4x8)
    # B[i, j] -> i*p + j
    # B_top (rows 0,1):
    # Block j (cols 2j, 2j+1):
    # [0, 2j] -> 2j
    # [0, 2j+1] -> 2j+1
    # [1, 2j] -> 8+2j
    # [1, 2j+1] -> 8+2j+1
    # B_bottom (rows 2,3):
    # Block j (cols 2j, 2j+1):
    # [2, 2j] -> 16+2j
    # [2, 2j+1] -> 16+2j+1
    # [3, 2j] -> 24+2j
    # [3, 2j+1] -> 24+2j+1
    
    # Indices for C (2x8)
    # C[i, j] -> j*n + i
    # Block j (cols 2j, 2j+1):
    # [0, 2j] -> 4j
    # [0, 2j+1] -> 4j+1
    # [1, 2j] -> 4j+2
    # [1, 2j+1] -> 4j+3

    def add_strassen_block(U, V, W, col_offset, A_idx, B_idx, C_idx):
        # A_idx: [a, b, c, d]
        # B_idx: [e, f, g, h]
        # C_idx: [c11, c12, c21, c22]
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # P1 = (a+d)(e+h)
        col = col_offset
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c11][col] += 1
        W[c22][col] += 1
        
        # P2 = (c+d)e
        col += 1
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c21][col] += 1
        W[c22][col] -= 1
        
        # P3 = a(f-g)
        col += 1
        U[a][col] += 1
        V[f][col] += 1
        V[g][col] -= 1
        W[c12][col] += 1
        W[c22][col] += 1
        
        # P4 = d(h-g)
        col += 1
        U[d][col] += 1
        V[h][col] += 1
        V[g][col] -= 1
        W[c11][col] -= 1
        W[c12][col] += 1
        W[c22][col] -= 1
        
        # P5 = (a+b)h
        col += 1
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c22][col] += 1
        
        # P6 = (c-a)(e+f)
        col += 1
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c11][col] += 1
        
        # P7 = (b-d)(g+h)
        col += 1
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c11][col] -= 1
        W[c21][col] += 1

    A1_idx = [0, 1, 4, 5]
    A2_idx = [2, 3, 6, 7]
    
    col_offset = 0
    for j in range(4):
        # B_top block j
        B_top_idx = [2*j, 2*j+1, 8+2*j, 8+2*j+1]
        C_idx = [4*j, 4*j+1, 4*j+2, 4*j+3]
        add_strassen_block(U, V, W, col_offset, A1_idx, B_top_idx, C_idx)
        col_offset += 7
        
        # B_bottom block j
        B_bot_idx = [16+2*j, 16+2*j+1, 24+2*j, 24+2*j+1]
        add_strassen_block(U, V, W, col_offset, A2_idx, B_bot_idx, C_idx)
        col_offset += 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
