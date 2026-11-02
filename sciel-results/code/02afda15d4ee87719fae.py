"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0

    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v00 = b_row * p + b_col
        v01 = b_row * p + b_col + 1
        v10 = (b_row + 1) * p + b_col
        v11 = (b_row + 1) * p + b_col + 1
        
        w00 = c_col * n + c_row
        w01 = (c_col + 1) * n + c_row
        w10 = c_col * n + (c_row + 1)
        w11 = (c_col + 1) * n + (c_row + 1)

        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1

        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1

        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1

        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1

        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1

        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_2x2x1(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v0 = b_row * p + b_col
        v1 = (b_row + 1) * p + b_col
        
        w0 = c_col * n + c_row
        w1 = c_col * n + (c_row + 1)

        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1; col += 1
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1; col += 1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1; col += 1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1; col += 1

    # C = A @ B1 + A @ B2 where B1 = B[:, 0:2], B2 = B[:, 2:5]
    
    # Part 1: A @ B1 (2x4 * 4x2 -> 2x2)
    add_strassen(0, 0, 0, 0, 0, 0)  # A[:,0:2] @ B[0:2, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0)  # A[:,2:4] @ B[2:4, 0:2]
    
    # Part 2: A @ B2 (2x4 * 4x3 -> 2x3)
    # Split B2 into B2[:,0] (4x1) and B2[:,1:3] (4x2)
    
    # Subpart 2a: A @ B2[:,0] (2x4 * 4x1 -> 2x1)
    add_2x2x1(0, 0, 0, 2, 0, 2)  # A[:,0:2] @ B[0:2, 2]
    add_2x2x1(0, 2, 2, 2, 0, 2)  # A[:,2:4] @ B[2:4, 2]
    
    # Subpart 2b: A @ B2[:,1:3] (2x4 * 4x2 -> 2x2)
    add_strassen(0, 0, 0, 3, 0, 3)  # A[:,0:2] @ B[0:2, 3:5]
    add_strassen(0, 2, 2, 3, 0, 3)  # A[:,2:4] @ B[2:4, 3:5]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
