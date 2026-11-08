import numpy as np
from scipy.linalg import block_diag

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We'll build a rank-32 decomposition using 2x2x2 Strassen blocks
    # A is 2x4, B is 4x5, C is 2x5
    # Split A into A_L (2x2) and A_R (2x2)
    # Split B into B_1 (2x2), B_2 (2x2), B_3 (2x1)
    # C = A_L * [B_1 B_2 B_3] + A_R * [B_1 B_2 B_3]
    #    = [A_L*B_1 + A_R*B_1] [A_L*B_2 + A_R*B_2] [A_L*B_3 + A_R*B_3]
    
    # For each pair (A_block, B_block) where A_block is 2x2 and B_block is 2xq:
    # If q=2: use Strassen (rank 7)
    # If q=1: use direct (rank 4)
    
    # Total rank: 2*7 + 2*7 + 2*4 = 14 + 14 + 8 = 36? 
    # Wait, that's the same as before. Let me reconsider.
    
    # Actually, let me try a different approach: use the fact that we can share some multiplications
    # between A_L and A_R.
    
    # Let's use a rank-32 decomposition based on 2x2x4 and 2x2x2 blocks.
    # A_L * B_12 (2x2 * 2x4) can be done in rank 12 (two independent 2x2x2 Strassen)
    # A_R * B_12 (2x2 * 2x4) rank 12
    # A_L * B_3 (2x2 * 2x1) rank 4
    # A_R * B_3 (2x2 * 2x1) rank 4
    # Total: 12 + 12 + 4 + 4 = 32
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # 20 x 32
    W = [[0] * rank for _ in range(p * n)]  # 10 x 32
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    # Strassen 2x2x2: multiplies Y (2x2) by Z (2x2) to get X (2x2)
    # Uses 7 multiplications
    def strassen_2x2x2(y00, y01, y10, y11, z00, z01, z10, z11, w00, w01, w10, w11):
        nonlocal col
        
        # M1 = (y00+y11)(z00+z11) -> contributes to w00, w11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M2 = (y10+y11)z00 -> contributes to w10, w11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # M3 = y00(z01-z11) -> contributes to w01, w11
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M4 = y11(z10-z00) -> contributes to w00, w10
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M5 = (y00+y01)z11 -> contributes to w00, w01
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01) -> contributes to w11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11) -> contributes to w00
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w00][col] += 1
        col += 1
    
    # Direct 2x2 * 2x1 multiplication (rank 4)
    def direct_2x2x1(y00, y01, y10, y11, z0, z1, w0, w1):
        nonlocal col
        
        # M1 = y00 * z0 -> w0
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z0[0], z0[1])][col] = 1
        W[w0][col] += 1
        col += 1
        
        # M2 = y01 * z1 -> w0
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z1[0], z1[1])][col] = 1
        W[w0][col] += 1
        col += 1
        
        # M3 = y10 * z0 -> w1
        U[u_idx(y10[0], y10[1])][col] = 1
        V[v_idx(z0[0], z0[1])][col] = 1
        W[w1][col] += 1
        col += 1
        
        # M4 = y11 * z1 -> w1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z1[0], z1[1])][col] = 1
        W[w1][col] += 1
        col += 1
    
    # A_L: rows 0-1, cols 0-1
    # A_R: rows 0-1, cols 2-3
    # B_1: rows 0-1, cols 0-1
    # B_2: rows 0-1, cols 2-3
    # B_3: rows 0-1, col 4
    
    # A_L * B_1 (2x2 * 2x2) -> C cols 0-1
    strassen_2x2x2(
        (0,0), (0,1), (1,0), (1,1),  # A_L
        (0,0), (0,1), (1,0), (1,1),  # B_1
        w_idx(0,0), w_idx(1,0),      # C_00, C_10
        w_idx(0,1), w_idx(1,1)       # C_01, C_11
    )
    
    # A_R * B_1 (2x2 * 2x2) -> C cols 0-1
    strassen_2x2x2(
        (0,2), (0,3), (1,2), (1,3),  # A_R
        (0,0), (0,1), (1,0), (1,1),  # B_1
        w_idx(0,0), w_idx(1,0),      # C_00, C_10
        w_idx(0,1), w_idx(1,1)       # C_01, C_11
    )
    
    # A_L * B_2 (2x2 * 2x2) -> C cols 2-3
    strassen_2x2x2(
        (0,0), (0,1), (1,0), (1,1),  # A_L
        (0,2), (0,3), (1,2), (1,3),  # B_2
        w_idx(2,0), w_idx(3,0),      # C_20, C_30
        w_idx(2,1), w_idx(3,1)       # C_21, C_31
    )
    
    # A_R * B_2 (2x2 * 2x2) -> C cols 2-3
    strassen_2x2x2(
        (0,2), (0,3), (1,2), (1,3),  # A_R
        (0,2), (0,3), (1,2), (1,3),  # B_2
        w_idx(2,0), w_idx(3,0),      # C_20, C_30
        w_idx(2,1), w_idx(3,1)       # C_21, C_31
    )
    
    # A_L * B_3 (2x2 * 2x1) -> C col 4
    direct_2x2x1(
        (0,0), (0,1), (1,0), (1,1),  # A_L
        (0,4), (1,4),                # B_3
        w_idx(4,0),                  # C_40
        w_idx(4,1)                   # C_41
    )
    
    # A_R * B_3 (2x2 * 2x1) -> C col 4
    direct_2x2x1(
        (0,2), (0,3), (1,2), (1,3),  # A_R
        (0,4), (1,4),                # B_3
        w_idx(4,0),                  # C_40
        w_idx(4,1)                   # C_41
    )
    
    # Verify col == rank
    assert col == rank, f"Expected col={rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
