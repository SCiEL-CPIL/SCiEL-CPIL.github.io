"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_idx: (i1, i2) indices in U for A
    # b_idx: (j1, j2) indices in V for B
    # c_idx: (k1, k2) indices in W for C (k*n+i)
    # Note: U index is i*m+j, V index is j*p+k, W index is k*n+i
    
    def add_strassen(a_i1, a_i2, a_j1, a_j2, b_j1, b_j2, b_k1, b_k2, c_i1, c_i2, c_k1, c_k2):
        nonlocal col
        # A submatrix indices in U:
        # A00: i1, j1 -> idx_a00 = a_i1*m + a_j1
        # A01: i1, j2 -> idx_a01 = a_i1*m + a_j2
        # A10: i2, j1 -> idx_a10 = a_i2*m + a_j1
        # A11: i2, j2 -> idx_a11 = a_i2*m + a_j2
        
        # B submatrix indices in V:
        # B00: j1, k1 -> idx_b00 = b_j1*p + b_k1
        # B01: j1, k2 -> idx_b01 = b_j1*p + b_k2
        # B10: j2, k1 -> idx_b10 = b_j2*p + b_k1
        # B11: j2, k2 -> idx_b11 = b_j2*p + b_k2
        
        # C submatrix indices in W:
        # C00: k1, i1 -> idx_c00 = c_k1*n + c_i1
        # C01: k1, i2 -> idx_c01 = c_k1*n + c_i2
        # C10: k2, i1 -> idx_c10 = c_k2*n + c_i1
        # C11: k2, i2 -> idx_c11 = c_k2*n + c_i2
        
        idx_a00 = a_i1 * m + a_j1
        idx_a01 = a_i1 * m + a_j2
        idx_a10 = a_i2 * m + a_j1
        idx_a11 = a_i2 * m + a_j2
        
        idx_b00 = b_j1 * p + b_k1
        idx_b01 = b_j1 * p + b_k2
        idx_b10 = b_j2 * p + b_k1
        idx_b11 = b_j2 * p + b_k2
        
        idx_c00 = c_k1 * n + c_i1
        idx_c01 = c_k1 * n + c_i2
        idx_c10 = c_k2 * n + c_i1
        idx_c11 = c_k2 * n + c_i2
        
        # Strassen 7 multiplications
        # M1 = (a00+a11)(b00+b11)
        U[idx_a00][col] += 1
        U[idx_a11][col] += 1
        V[idx_b00][col] += 1
        V[idx_b11][col] += 1
        W[idx_c00][col] += 1
        W[idx_c01][col] += 1
        W[idx_c10][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[idx_a10][col] += 1
        U[idx_a11][col] += 1
        V[idx_b00][col] += 1
        W[idx_c10][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[idx_a00][col] += 1
        V[idx_b01][col] += 1
        V[idx_b11][col] -= 1
        W[idx_c01][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[idx_a11][col] += 1
        V[idx_b10][col] += 1
        V[idx_b00][col] -= 1
        W[idx_c00][col] += 1
        W[idx_c10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[idx_a00][col] += 1
        U[idx_a01][col] += 1
        V[idx_b11][col] += 1
        W[idx_c00][col] -= 1
        W[idx_c01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[idx_a10][col] += 1
        U[idx_a00][col] -= 1
        V[idx_b00][col] += 1
        V[idx_b01][col] += 1
        W[idx_c11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[idx_a01][col] += 1
        U[idx_a11][col] -= 1
        V[idx_b10][col] += 1
        V[idx_b11][col] += 1
        W[idx_c00][col] += 1
        W[idx_c11][col] -= 1
        col += 1

    # 1. Add 4 Strassen blocks for the 2x4x4 part
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1}
    # C indices: k in {0,1}, i in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    
    # Block 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # A indices: i in {0,1}, j in {2,3}
    # B indices: j in {2,3}, k in {0,1}
    # C indices: k in {0,1}, i in {0,1}
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)
    
    # Block 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {2,3}
    # C indices: k in {2,3}, i in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # A indices: i in {0,1}, j in {2,3}
    # B indices: j in {2,3}, k in {2,3}
    # C indices: k in {2,3}, i in {0,1}
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)
    
    # 2. Add 8 schoolbook terms for the 2x4x1 part (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
