"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen for 2x2 blocks
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We split B (4x6) into three 4x2 blocks.
    # Block j (0, 1, 2) corresponds to columns 2*j, 2*j+1 of B.
    # For each block, we compute C_block = A * B_block (2x2 result).
    # A is 2x4, split into A_L (2x2) and A_R (2x2).
    # B_block is 4x2, split into B_T (2x2) and B_B (2x2).
    # C_block = A_L * B_T + A_R * B_B.
    # We use Strassen for A_L * B_T and A_R * B_B.
    
    # Strassen factors for X * Y (2x2 * 2x2)
    # X indices: 00, 01, 10, 11
    # Y indices: 00, 01, 10, 11
    # M1..M7
    # We need to map these to the global U, V, W indices.
    
    # Helper to add a Strassen term
    # x_off: offset for X rows in U (i, j_x)
    # y_off: offset for Y rows in V (j_y, k)
    # z_off: offset for Z rows in W (k, i)
    # base_col: column offset in U, V, W
    
    def add_strassen(x_i, x_j, y_j, y_k, z_k, z_i, base_col):
        # X is 2x2 at U rows:
        # (x_i, x_j), (x_i, x_j+1), (x_i+1, x_j), (x_i+1, x_j+1)
        # Y is 2x2 at V rows:
        # (y_j, y_k), (y_j, y_k+1), (y_j+1, y_k), (y_j+1, y_k+1)
        # Z is 2x2 at W rows:
        # (z_k, z_i), (z_k+1, z_i), (z_k, z_i+1), (z_k+1, z_i+1)
        
        # M1 = (X00 + X11)(Y00 + Y11)
        U[x_i * m + x_j][base_col] += 1
        U[(x_i+1) * m + (x_j+1)][base_col] += 1
        V[y_j * p + y_k][base_col] += 1
        V[(y_j+1) * p + (y_k+1)][base_col] += 1
        W[z_k * n + z_i][base_col] += 1
        W[(z_k+1) * n + (z_i+1)][base_col] += 1
        # Z00 += M1, Z11 += M1
        
        # M2 = (X10 + X11)Y00
        U[(x_i+1) * m + x_j][base_col+1] += 1
        U[(x_i+1) * m + (x_j+1)][base_col+1] += 1
        V[y_j * p + y_k][base_col+1] += 1
        W[(z_k+1) * n + z_i][base_col+1] += 1
        W[(z_k+1) * n + (z_i+1)][base_col+1] += 1
        # Z10 += M2, Z11 += M2? 
        # Wait, Strassen formulas:
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        # So M2 contributes to Z10 and Z11.
        
        # M3 = X00(Y01 - Y11)
        U[x_i * m + x_j][base_col+2] += 1
        V[y_j * p + (y_k+1)][base_col+2] += 1
        V[(y_j+1) * p + (y_k+1)][base_col+2] -= 1
        W[z_k * n + (z_i+1)][base_col+2] += 1
        W[(z_k+1) * n + (z_i+1)][base_col+2] += 1
        # Z01 += M3, Z11 += M3
        
        # M4 = X11(Y10 - Y00)
        U[(x_i+1) * m + (x_j+1)][base_col+3] += 1
        V[(y_j+1) * p + y_k][base_col+3] += 1
        V[y_j * p + y_k][base_col+3] -= 1
        W[(z_k+1) * n + z_i][base_col+3] += 1
        W[z_k * n + z_i][base_col+3] += 1
        # Z10 += M4, Z00 += M4
        
        # M5 = (X00 + X01)Y11
        U[x_i * m + x_j][base_col+4] += 1
        U[x_i * m + (x_j+1)][base_col+4] += 1
        V[(y_j+1) * p + (y_k+1)][base_col+4] += 1
        W[z_k * n + (z_i+1)][base_col+4] -= 1
        W[(z_k+1) * n + (z_i+1)][base_col+4] -= 1
        # Z00 -= M5, Z11 -= M5?
        # Z00 = M1 + M4 - M5 + M7
        # Z11 = M1 - M2 + M3 + M6
        # Wait, M5 is (X00+X01)Y11.
        # Z01 = M3 + M5. So Z01 += M5.
        # Z00 -= M5.
        
        # M6 = (X10 - X00)(Y00 + Y01)
        U[(x_i+1) * m + x_j][base_col+5] += 1
        U[x_i * m + x_j][base_col+5] -= 1
        V[y_j * p + y_k][base_col+5] += 1
        V[y_j * p + (y_k+1)][base_col+5] += 1
        W[(z_k+1) * n + (z_i+1)][base_col+5] += 1
        # Z11 += M6
        
        # M7 = (X01 - X11)(Y10 + Y11)
        U[x_i * m + (x_j+1)][base_col+6] += 1
        U[(x_i+1) * m + (x_j+1)][base_col+6] -= 1
        V[(y_j+1) * p + y_k][base_col+6] += 1
        V[(y_j+1) * p + (y_k+1)][base_col+6] += 1
        W[z_k * n + z_i][base_col+6] += 1
        # Z00 += M7

    # Loop over the 3 blocks of B (width 2 each)
    for block in range(3):
        # B block columns: 2*block, 2*block+1
        y_k = 2 * block
        
        # A_L * B_T
        # A_L is cols 0,1 of A. x_j = 0.
        # B_T is rows 0,1 of B_block. y_j = 0.
        # Result is top-left 2x2 of C_block? No, C_block is 2x2.
        # C_block[i, k] where k is 0,1 relative to block.
        # Global k is 2*block + k_rel.
        # Global i is i.
        
        # x_i = 0 (row of A)
        # x_j = 0 (col of A)
        # y_j = 0 (row of B)
        # y_k = 2*block (col of B)
        # z_k = 2*block (col of C)
        # z_i = 0 (row of C)
        
        add_strassen(0, 0, 0, y_k, y_k, 0, block * 14)
        
        # A_R * B_B
        # A_R is cols 2,3 of A. x_j = 2.
        # B_B is rows 2,3 of B_block. y_j = 2.
        # Result adds to same C_block.
        
        add_strassen(0, 2, 2, y_k, y_k, 0, block * 14 + 7)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
