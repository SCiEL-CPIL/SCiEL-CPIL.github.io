"""Program for finding rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose the 2x3x8 matrix multiplication tensor using a Strassen-based approach.
    
    Strategy:
    - Split the 3x8 matrix B into B_top (2x8) and B_bottom (1x8)
    - Split the 2x3 matrix A into A_left (2x2) and A_right (2x1)
    - C = A_left * B_top + A_right * B_bottom
    
    For A_left (2x2) * B_top (2x8):
    - Split B_top into four 2x2 blocks: B_tl, B_tr, B_bl, B_br
    - Apply Strassen (rank 7) to each 2x2x2 block
    - Total: 4 * 7 = 28 multiplications
    
    For A_right (2x1) * B_bottom (1x8):
    - Schoolbook: 2 * 1 * 8 = 16 multiplications
    
    Total rank: 28 + 16 = 44
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A_left (2x2) * B_top (2x8) using Strassen on each 2x2 block
    # A_left indices: U[0]=a00, U[1]=a01, U[2]=a10, U[3]=a11
    # B_top: rows 0,1 of B; columns k, k+1 for each block
    # C indices: W[k*n + i] = c_{i,k}
    
    for k_start in range(0, 8, 2):
        # Block B_{:, k_start:k_start+2}
        # b00 = V[0*p + k_start], b01 = V[0*p + k_start+1]
        # b10 = V[1*p + k_start], b11 = V[1*p + k_start+1]
        # c00 = W[k_start*n + 0], c01 = W[k_start*n + 1]
        # c10 = W[(k_start+1)*n + 0], c11 = W[(k_start+1)*n + 1]
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[0][col] = 1
        U[3][col] = 1
        V[0 * p + k_start][col] = 1
        V[1 * p + k_start + 1][col] = 1
        W[k_start * n + 0][col] = 1
        W[k_start * n + 1][col] = 1
        W[(k_start + 1) * n + 0][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[2][col] = 1
        U[3][col] = 1
        V[0 * p + k_start][col] = 1
        W[k_start * n + 0][col] = 1
        W[(k_start + 1) * n + 0][col] = 1
        col += 1
        
        # M3 = a00 * (b10 - b00)
        U[0][col] = 1
        V[1 * p + k_start][col] = 1
        V[0 * p + k_start][col] = -1
        W[k_start * n + 1][col] = 1
        W[(k_start + 1) * n + 1][col] = -1
        col += 1
        
        # M4 = a11 * (b01 - b11)
        U[3][col] = 1
        V[0 * p + k_start + 1][col] = 1
        V[1 * p + k_start + 1][col] = -1
        W[k_start * n + 0][col] = 1
        W[(k_start + 1) * n + 0][col] = -1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + k_start + 1][col] = 1
        W[k_start * n + 1][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[2][col] = 1
        U[0][col] = -1
        V[0 * p + k_start][col] = 1
        V[0 * p + k_start + 1][col] = 1
        W[(k_start + 1) * n + 0][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[1][col] = 1
        U[3][col] = -1
        V[1 * p + k_start][col] = 1
        V[1 * p + k_start + 1][col] = 1
        W[k_start * n + 0][col] = -1
        W[k_start * n + 1][col] = 1
        col += 1
    
    # Part 2: A_right (2x1) * B_bottom (1x8) - schoolbook
    # A_right: U[4]=a02, U[5]=a12
    # B_bottom: row 2 of B, V[2*p + k] = b2k
    
    for k in range(8):
        # c0k += a02 * b2k
        U[4][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # c1k += a12 * b2k
        U[5][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
