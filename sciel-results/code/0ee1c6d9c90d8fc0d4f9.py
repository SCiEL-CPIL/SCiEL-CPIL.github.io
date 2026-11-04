import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # Rank 51 decomposition
    rank = 51
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # 3 blocks of 3 columns each
    for block_idx in range(3):
        # k_offset is the starting column index in the 3x9 matrix
        k_offset = block_idx * 3
        
        # Part 1: Strassen for A[2x2] * B[2x2] (first 2 cols of block)
        # A[2x2] uses A[0,0], A[0,1], A[1,0], A[1,1]
        # B[2x2] uses B[0, k_offset], B[0, k_offset+1], B[1, k_offset], B[1, k_offset+1]
        
        # M1 = (a+b)(x1+y1)
        U[0][col_offset] = 1  # a
        U[1][col_offset] = 1  # b
        V[0 * p + k_offset][col_offset] = 1  # x1
        V[1 * p + k_offset][col_offset] = 1  # y1
        # W contributions for C11, C12, C21, C22
        # C11 (k_offset, 0)
        W[(k_offset) * n + 0][col_offset] = 1
        # C12 (k_offset+1, 0)
        W[(k_offset+1) * n + 0][col_offset] = -1
        # C21 (k_offset, 1)
        W[(k_offset) * n + 1][col_offset] = 1
        # C22 (k_offset+1, 1)
        W[(k_offset+1) * n + 1][col_offset] = 1
        col_offset += 1
        
        # M2 = a(x2+y2)
        U[0][col_offset] = 1  # a
        V[0 * p + k_offset + 1][col_offset] = 1  # x2
        V[1 * p + k_offset + 1][col_offset] = 1  # y2
        # W contributions
        # C11
        W[(k_offset) * n + 0][col_offset] = -1
        # C12
        W[(k_offset+1) * n + 0][col_offset] = 1
        col_offset += 1
        
        # M3 = b(x1-y1)
        U[1][col_offset] = 1  # b
        V[0 * p + k_offset][col_offset] = 1  # x1
        V[1 * p + k_offset][col_offset] = -1  # y1
        # W contributions
        # C21
        W[(k_offset) * n + 1][col_offset] = 1
        # C22
        W[(k_offset+1) * n + 1][col_offset] = -1
        col_offset += 1
        
        # M4 = d(x1+y2)
        U[3][col_offset] = 1  # d (index 1*2+1)
        V[0 * p + k_offset][col_offset] = 1  # x1
        V[1 * p + k_offset + 1][col_offset] = 1  # y2
        # W contributions
        # C11
        W[(k_offset) * n + 0][col_offset] = 1
        # C22
        W[(k_offset+1) * n + 1][col_offset] = 1
        col_offset += 1
        
        # M5 = (a+d)(y2-x2)
        U[0][col_offset] = 1  # a
        U[3][col_offset] = 1  # d
        V[0 * p + k_offset + 1][col_offset] = -1  # x2
        V[1 * p + k_offset + 1][col_offset] = 1  # y2
        # W contributions
        # C11
        W[(k_offset) * n + 0][col_offset] = -1
        # C12
        W[(k_offset+1) * n + 0][col_offset] = 1
        col_offset += 1
        
        # M6 = (c+d)(x1-y1)
        U[2][col_offset] = 1  # c (index 1*2+0)
        U[3][col_offset] = 1  # d
        V[0 * p + k_offset][col_offset] = 1  # x1
        V[1 * p + k_offset][col_offset] = -1  # y1
        # W contributions
        # C21
        W[(k_offset) * n + 1][col_offset] = 1
        # C22
        W[(k_offset+1) * n + 1][col_offset] = 1
        col_offset += 1
        
        # M7 = (b-c)(y1+x2)
        U[1][col_offset] = 1  # b
        U[2][col_offset] = -1  # c
        V[0 * p + k_offset + 1][col_offset] = 1  # x2
        V[1 * p + k_offset][col_offset] = 1  # y1
        # W contributions
        # C22
        W[(k_offset+1) * n + 1][col_offset] = 1
        col_offset += 1
        
        # Part 2: 2x2x1 for 3rd col of B[2x3] (col k_offset+2)
        # C13 = a*x3 + b*y3
        # C23 = c*x3 + d*y3
        
        # M8 = a*x3
        U[0][col_offset] = 1
        V[0 * p + k_offset + 2][col_offset] = 1
        W[(k_offset+2) * n + 0][col_offset] = 1
        col_offset += 1
        
        # M9 = b*y3
        U[1][col_offset] = 1
        V[1 * p + k_offset + 2][col_offset] = 1
        W[(k_offset+2) * n + 0][col_offset] = 1
        col_offset += 1
        
        # M10 = c*x3
        U[2][col_offset] = 1
        V[0 * p + k_offset + 2][col_offset] = 1
        W[(k_offset+2) * n + 1][col_offset] = 1
        col_offset += 1
        
        # M11 = d*y3
        U[3][col_offset] = 1
        V[1 * p + k_offset + 2][col_offset] = 1
        W[(k_offset+2) * n + 1][col_offset] = 1
        col_offset += 1
        
        # Part 3: Outer product a[2x1] * b[1x3]
        # a is col 2 of A (indices 2, 3)
        # b is row 2 of B_block (indices 2*p + k_offset, 2*p + k_offset+1, 2*p + k_offset+2)
        
        # M12 = u*z1 (u=A[0,2], z1=B[2, k_offset])
        U[2][col_offset] = 1
        V[2 * p + k_offset][col_offset] = 1
        W[(k_offset) * n + 0][col_offset] = 1
        col_offset += 1
        
        # M13 = u*z2
        U[2][col_offset] = 1
        V[2 * p + k_offset + 1][col_offset] = 1
        W[(k_offset+1) * n + 0][col_offset] = 1
        col_offset += 1
        
        # M14 = u*z3
        U[2][col_offset] = 1
        V[2 * p + k_offset + 2][col_offset] = 1
        W[(k_offset+2) * n + 0][col_offset] = 1
        col_offset += 1
        
        # M15 = v*z1 (v=A[1,2], z1=B[2, k_offset])
        U[3][col_offset] = 1
        V[2 * p + k_offset][col_offset] = 1
        W[(k_offset) * n + 1][col_offset] = 1
        col_offset += 1
        
        # M16 = v*z2
        U[3][col_offset] = 1
        V[2 * p + k_offset + 1][col_offset] = 1
        W[(k_offset+1) * n + 1][col_offset] = 1
        col_offset += 1
        
        # M17 = v*z3
        U[3][col_offset] = 1
        V[2 * p + k_offset + 2][col_offset] = 1
        W[(k_offset+2) * n + 1][col_offset] = 1
        col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
