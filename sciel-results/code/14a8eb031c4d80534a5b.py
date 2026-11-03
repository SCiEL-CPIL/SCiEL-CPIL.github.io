"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    def add_strassen_block(A_idx, B_idx, C_idx, col_start):
        """
        Add a Strassen 2x2 x 2x2 block starting at column col_start.
        A_idx: list of 4 indices into U rows for the 2x2 A matrix (row-major: [a00, a01, a10, a11])
        B_idx: list of 4 indices into V rows for the 2x2 B matrix (row-major: [b00, b01, b10, b11])
        C_idx: list of 4 indices into W rows for the 2x2 C matrix (row-major: [c00, c01, c10, c11])
        """
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        c00, c01, c10, c11 = C_idx
        
        # M1 = (x00+x11)*(y00+y11)
        col = col_start
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M2 = (x10+x11)*y00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3 = x00*(y01-y11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M4 = x11*(y10-y00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M5 = (x00+x01)*y11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M6 = (x10-x00)*(y00+y01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (x01-x11)*(y10+y11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] -= 1
        col += 1
        
        return col_start + 7
    
    # A is 2x4: A[i,j] for i in 0..1, j in 0..3
    # U row index for A[i,j] = i*m + j = i*4 + j
    # A1 = A[0:2, 0:2] -> U rows: [0*4+0, 0*4+1, 1*4+0, 1*4+1] = [0, 1, 4, 5]
    # A2 = A[0:2, 2:4] -> U rows: [0*4+2, 0*4+3, 1*4+2, 1*4+3] = [2, 3, 6, 7]
    
    # B is 4x4: B[j,k] for j in 0..3, k in 0..3
    # V row index for B[j,k] = j*p + k = j*4 + k
    # B1 = B[0:2, 0:2] -> V rows: [0*4+0, 0*4+1, 1*4+0, 1*4+1] = [0, 1, 4, 5]
    # B2 = B[0:2, 2:4] -> V rows: [0*4+2, 0*4+3, 1*4+2, 1*4+3] = [2, 3, 6, 7]
    # B3 = B[2:4, 0:2] -> V rows: [2*4+0, 2*4+1, 3*4+0, 3*4+1] = [8, 9, 12, 13]
    # B4 = B[2:4, 2:4] -> V rows: [2*4+2, 2*4+3, 3*4+2, 3*4+3] = [10, 11, 14, 15]
    
    # C is 2x4: C[i,k] for i in 0..1, k in 0..3
    # W row index for C[i,k] = k*n + i = k*2 + i
    # C1 = C[0:2, 0:2] -> W rows: [0*2+0, 0*2+1, 1*2+0, 1*2+1] = [0, 1, 2, 3]
    # C2 = C[0:2, 2:4] -> W rows: [2*2+0, 2*2+1, 3*2+0, 3*2+1] = [4, 5, 6, 7]
    
    A1 = [0, 1, 4, 5]
    A2 = [2, 3, 6, 7]
    B1 = [0, 1, 4, 5]
    B2 = [2, 3, 6, 7]
    B3 = [8, 9, 12, 13]
    B4 = [10, 11, 14, 15]
    C1 = [0, 1, 2, 3]
    C2 = [4, 5, 6, 7]
    
    col = 0
    col = add_strassen_block(A1, B1, C1, col)
    col = add_strassen_block(A2, B3, C1, col)
    col = add_strassen_block(A1, B2, C2, col)
    col = add_strassen_block(A2, B4, C2, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
