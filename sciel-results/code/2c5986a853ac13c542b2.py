# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=4
    
    # Target rank 34
    R = 34
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # a_idx: list of 4 (i,j) indices for A block
    # b_idx: list of 4 (j,k) indices for B block
    # c_idx: list of 4 (k,i) indices for C block
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Indices in 2x2 block:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # We need to map these to global indices.
        # a_idx[0] is (i0, j0), a_idx[1] is (i0, j1), a_idx[2] is (i1, j0), a_idx[3] is (i1, j1)
        # b_idx[0] is (j0, k0), b_idx[1] is (j0, k1), b_idx[2] is (j1, k0), b_idx[3] is (j1, k1)
        # c_idx[0] is (k0, i0), c_idx[1] is (k0, i1), c_idx[2] is (k1, i0), c_idx[3] is (k1, i1)
        
        # M1
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M2
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M3
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M4
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M5
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M6
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # M7
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
    # Block 1: A11 (2x2) * B1 (2x4)
    # A11: i in {0,1}, j in {0,1}
    # B1: j in {0,1}, k in {0,1,2,3}
    # Split B1 into B11 (j in {0,1}, k in {0,1}) and B12 (j in {0,1}, k in {2,3})
    
    # A11 indices
    a11_idx = [0*3+0, 0*3+1, 1*3+0, 1*3+1] # (0,0), (0,1), (1,0), (1,1)
    
    # B11 indices (k in 0,1)
    b11_idx = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # (0,0), (0,1), (1,0), (1,1)
    
    # C11 indices (k in 0,1, i in 0,1)
    c11_idx = [0*3+0, 0*3+1, 1*3+0, 1*3+1] # (0,0), (0,1), (1,0), (1,1)
    
    add_strassen(a11_idx, b11_idx, c11_idx)
    
    # B12 indices (k in 2,3)
    b12_idx = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # (0,2), (0,3), (1,2), (1,3)
    
    # C12 indices (k in 2,3, i in 0,1)
    c12_idx = [2*3+0, 2*3+1, 3*3+0, 3*3+1] # (2,0), (2,1), (3,0), (3,1)
    
    add_strassen(a11_idx, b12_idx, c12_idx)
    
    # Block 2: A12 (2x1) * B2 (1x4)
    # A12: i in {0,1}, j=2
    # B2: j=2, k in {0,1,2,3}
    # C: i in {0,1}, k in {0,1,2,3}
    for i in range(2):
        for k in range(4):
            U[i*3+2][col] = 1
            V[2*4+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
            
    # Block 3: A21 (1x2) * B1 (2x4)
    # A21: i=2, j in {0,1}
    # B1: j in {0,1}, k in {0,1,2,3}
    # C: i=2, k in {0,1,2,3}
    for j in range(2):
        for k in range(4):
            U[2*3+j][col] = 1
            V[j*4+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
            
    # Block 4: A22 (1x1) * B2 (1x4)
    # A22: i=2, j=2
    # B2: j=2, k in {0,1,2,3}
    # C: i=2, k in {0,1,2,3}
    for k in range(4):
        U[2*3+2][col] = 1
        V[2*4+k][col] = 1
        W[k*3+2][col] = 1
        col += 1
        
    # Check col == R
    assert col == R, f"Expected rank {R}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
