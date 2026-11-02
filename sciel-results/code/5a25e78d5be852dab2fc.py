"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Split m=4 into 2+2, p=5 into 1+4
    # Block 1: A(:, 0:2) * B(0:2, 0:1) -> 2x2x1 (rank 2)
    # Block 2: A(:, 0:2) * B(0:2, 1:5) -> 2x2x4 (rank 14, two 2x2x2 Strassen)
    # Block 3: A(:, 2:4) * B(2:4, 0:1) -> 2x2x1 (rank 2)
    # Block 4: A(:, 2:4) * B(2:4, 1:5) -> 2x2x4 (rank 14, two 2x2x2 Strassen)
    # Total rank: 2 + 14 + 2 + 14 = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        u00 = (a_row_off + 0) * m + (a_col_off + 0)
        u01 = (a_row_off + 0) * m + (a_col_off + 1)
        u10 = (a_row_off + 1) * m + (a_col_off + 0)
        u11 = (a_row_off + 1) * m + (a_col_off + 1)
        
        v00 = (b_row_off + 0) * p + (b_col_off + 0)
        v01 = (b_row_off + 0) * p + (b_col_off + 1)
        v10 = (b_row_off + 1) * p + (b_col_off + 0)
        v11 = (b_row_off + 1) * p + (b_col_off + 1)
        
        w00 = (c_col_off + 0) * n + (c_row_off + 0)
        w01 = (c_col_off + 1) * n + (c_row_off + 0)
        w10 = (c_col_off + 0) * n + (c_row_off + 1)
        w11 = (c_col_off + 1) * n + (c_row_off + 1)
        
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

    def add_2x2x1(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        u00 = (a_row_off + 0) * m + (a_col_off + 0)
        u01 = (a_row_off + 1) * m + (a_col_off + 0)
        u10 = (a_row_off + 0) * m + (a_col_off + 1)
        u11 = (a_row_off + 1) * m + (a_col_off + 1)
        
        v0 = (b_row_off + 0) * p + b_col_off
        v1 = (b_row_off + 1) * p + b_col_off
        
        w0 = (c_col_off) * n + (c_row_off + 0)
        w1 = (c_col_off) * n + (c_row_off + 1)
        
        # C0 = a00*b0 + a01*b1
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        col += 1
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1
        col += 1
        
        # C1 = a10*b0 + a11*b1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1
        col += 1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1
        col += 1

    def add_2x2x4(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        # Split B cols into 2+2
        add_strassen_2x2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
        add_strassen_2x2x2x2(a_row_off, a_col_off, b_row_off, b_col_off + 2, c_row_off, c_col_off + 2)

    # Block 1: A(:, 0:2) * B(0:2, 0:1)
    # a_row_off=0, a_col_off=0
    # b_row_off=0, b_col_off=0
    # c_row_off=0, c_col_off=0
    add_2x2x1(0, 0, 0, 0, 0, 0)
    
    # Block 2: A(:, 0:2) * B(0:2, 1:5)
    # a_row_off=0, a_col_off=0
    # b_row_off=0, b_col_off=1
    # c_row_off=0, c_col_off=1
    add_2x2x4(0, 0, 0, 1, 0, 1)
    
    # Block 3: A(:, 2:4) * B(2:4, 0:1)
    # a_row_off=0, a_col_off=2
    # b_row_off=2, b_col_off=0
    # c_row_off=0, c_col_off=0
    add_2x2x1(0, 2, 2, 0, 0, 0)
    
    # Block 4: A(:, 2:4) * B(2:4, 1:5)
    # a_row_off=0, a_col_off=2
    # b_row_off=2, b_col_off=1
    # c_row_off=0, c_col_off=1
    add_2x2x4(0, 2, 2, 1, 0, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
