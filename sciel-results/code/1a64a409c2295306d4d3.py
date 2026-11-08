"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Strategy: decompose 2x5x6 into sub-problems and use Kronecker structure
    # Part 1: 2x2x6 (columns 0,1 of A)
    #   Split k into 3 pairs, use Strassen 2x2x2 for each -> 3*7 = 21
    # Part 2: 2x3x6 (columns 2,3,4 of A)
    #   Sub-part 2a: 2x2x6 (columns 2,3) -> 3*7 = 21
    #   Sub-part 2b: 2x1x6 (column 4) -> use 2x1x2 Strassen for 3 pairs of k
    #     For 2x1x2: minimal rank is 4 (not 6 as naive)
    #     So 3 * 4 = 12
    # Total: 21 + 21 + 12 = 54
    
    # But we can do better for 2x1x2:
    # C_0,0 = A_0,j * B_j,0
    # C_0,1 = A_0,j * B_j,1
    # C_1,0 = A_1,j * B_j,0
    # C_1,1 = A_1,j * B_j,1
    # This is just a Kronecker product of 2x1 and 1x2, rank = 2*2 = 4
    # We need 4 terms, each multiplying one A element with one B element.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank for _ in range(p * n)]  # 12 x 54
    
    def strassen_2x2x2(U, V, W, j0, k0, col):
        """Add Strassen decomposition for 2x2x2 block."""
        # P1 = (A_0,j0 + A_1,j0+1)(B_j0,k0 + B_j0+1,k0+1)
        U[0*m + j0][col] += 1
        U[1*m + j0+1][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + 0][col] += 1
        W[(k0+1)*n + 1][col] += 1
        col += 1
        
        # P2 = (A_1,j0 + A_1,j0+1) B_j0,k0
        U[1*m + j0][col] += 1
        U[1*m + j0+1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + 1][col] += 1
        W[(k0+1)*n + 1][col] -= 1
        col += 1
        
        # P3 = A_0,j0 (B_j0,k0+1 - B_j0+1,k0+1)
        U[0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + 0][col] += 1
        W[(k0+1)*n + 1][col] += 1
        col += 1
        
        # P4 = A_1,j0+1 (B_j0+1,k0 - B_j0,k0)
        U[1*m + j0+1][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + 0][col] += 1
        W[k0*n + 1][col] += 1
        col += 1
        
        # P5 = (A_0,j0 + A_0,j0+1) B_j0+1,k0+1
        U[0*m + j0][col] += 1
        U[0*m + j0+1][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + 0][col] -= 1
        W[(k0+1)*n + 0][col] += 1
        col += 1
        
        # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k0+1)
        U[1*m + j0][col] += 1
        U[0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + 1][col] += 1
        col += 1
        
        # P7 = (A_0,j0+1 - A_1,j0+1)(B_j0+1,k0 + B_j0+1,k0+1)
        U[0*m + j0+1][col] += 1
        U[1*m + j0+1][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + 0][col] += 1
        col += 1
        
        return col
    
    def naive_2x1x2(U, V, W, j, k0, col):
        """Add naive decomposition for 2x1x2 block at A col j; B row j; k=k0,k0+1"""
        for k_offset in range(2):
            k = k0 + k_offset
            for row in range(2):
                U[row*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + row][col] += 1
                col += 1
        return col
    
    col = 0
    # Part 1: 2x2x6 for j=0,1
    for k0 in [0, 2, 4]:
        col = strassen_2x2x2(U, V, W, 0, k0, col)
    
    # Part 2: 2x3x6 for j=2,3,4
    # Sub-part 2a: 2x2x6 for j=2,3
    for k0 in [0, 2, 4]:
        col = strassen_2x2x2(U, V, W, 2, k0, col)
    
    # Sub-part 2b: 2x1x6 for j=4
    for k0 in [0, 2, 4]:
        col = naive_2x1x2(U, V, W, 4, k0, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
