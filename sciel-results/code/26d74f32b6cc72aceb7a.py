"""Program for matrix multiplication 3x3x4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

def construct():
    """
    Construct a rank-reduced tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix using a hybrid Strassen/Schoolbook approach.
    
    Rank 34 decomposition:
    1. A[0:2, 0:2] * B[0:2, :] (2x2 * 2x4) -> Rank 14 (2x Strassen 2x2x2)
    2. A[0:2, 2:3] * B[2:3, :] (2x1 * 1x4) -> Rank 8 (Outer product)
    3. A[2:3, 0:2] * B[0:2, :] (1x2 * 2x4) -> Rank 8 (Dot products)
    4. A[2:3, 2:3] * B[2:3, :] (1x1 * 1x4) -> Rank 4 (Scalar mult)
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Helper to add a product term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if coeff != 0:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff
        col += 1

    # --- Part 1: A[0:2, 0:2] * B[0:2, 0:4] ---
    # This is 2x2 matrix A11 times 2x4 matrix B1.
    # Split B1 into B1_left (2x2) and B1_right (2x2).
    # Use Strassen for each.
    
    # Strassen 1: A11 * B1_left
    # A11 indices: (0,0), (0,1), (1,0), (1,1)
    # B1_left indices: (0,0), (0,1), (1,0), (1,1)
    
    # P1 = (A00+A11)(B00+B11)
    # U: A00, A11
    # V: B00, B11
    # W: C00, C11 (with coefficients)
    
    # We need to be careful with the W coefficients.
    # Instead of adding to U/V/W immediately with complex logic, 
    # let's define the 7 Strassen products explicitly.
    
    def strassen_2x2x2(a_base, b_base, c_base, start_col):
        # a_base: (i, j) offset for A
        # b_base: (j, k) offset for B
        # c_base: (i, k) offset for C
        # Returns end_col
        
        # Indices in local 2x2 block:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        
        col_local = start_col
        
        # P1 = (A00+A11)(B00+B11)
        U[a_base[0]*m + a_base[1]][col_local] = 1
        U[(a_base[0]+1)*m + (a_base[1]+1)][col_local] = 1
        V[b_base[0]*p + b_base[1]][col_local] = 1
        V[(b_base[0]+1)*p + (b_base[1]+1)][col_local] = 1
        W[c_base[0]*n + c_base[1]][col_local] = 1
        W[(c_base[0]+1)*n + (c_base[1]+1)][col_local] = 1
        col_local += 1
        
        # P2 = (A10+A11)B00
        U[(a_base[0]+1)*m + a_base[1]][col_local] = 1
        U[(a_base[0]+1)*m + (a_base[1]+1)][col_local] = 1
        V[b_base[0]*p + b_base[1]][col_local] = 1
        W[(c_base[0]+1)*n + c_base[1]][col_local] = 1
        col_local += 1
        
        # P3 = A00(B01-B11)
        U[a_base[0]*m + a_base[1]][col_local] = 1
        V[b_base[0]*p + (b_base[1]+1)][col_local] = 1
        V[(b_base[0]+1)*p + (b_base[1]+1)][col_local] = -1
        W[c_base[0]*n + (c_base[1]+1)][col_local] = 1
        col_local += 1
        
        # P4 = A11(B10-B00)
        U[(a_base[0]+1)*m + (a_base[1]+1)][col_local] = 1
        V[(b_base[0]+1)*p + b_base[1]][col_local] = 1
        V[b_base[0]*p + b_base[1]][col_local] = -1
        W[(c_base[0]+1)*n + c_base[1]][col_local] = 1
        col_local += 1
        
        # P5 = (A00+A01)B11
        U[a_base[0]*m + a_base[1]][col_local] = 1
        U[a_base[0]*m + (a_base[1]+1)][col_local] = 1
        V[(b_base[0]+1)*p + (b_base[1]+1)][col_local] = 1
        W[c_base[0]*n + (c_base[1]+1)][col_local] = 1
        W[(c_base[0]+1)*n + (c_base[1]+1)][col_local] = -1
        col_local += 1
        
        # P6 = (A10-A00)(B00+B01)
        U[(a_base[0]+1)*m + a_base[1]][col_local] = 1
        U[a_base[0]*m + a_base[1]][col_local] = -1
        V[b_base[0]*p + b_base[1]][col_local] = 1
        V[b_base[0]*p + (b_base[1]+1)][col_local] = 1
        W[(c_base[0]+1)*n + (c_base[1]+1)][col_local] = 1
        col_local += 1
        
        # P7 = (A01-A11)(B10+B11)
        U[a_base[0]*m + (a_base[1]+1)][col_local] = 1
        U[(a_base[0]+1)*m + (a_base[1]+1)][col_local] = -1
        V[(b_base[0]+1)*p + b_base[1]][col_local] = 1
        V[(b_base[0]+1)*p + (b_base[1]+1)][col_local] = 1
        W[c_base[0]*n + (c_base[1]+1)][col_local] = -1
        W[(c_base[0]+1)*n + c_base[1]][col_local] = 1
        col_local += 1
        
        return col_local

    # Call Strassen for B_left (k=0,1)
    col = strassen_2x2x2((0,0), (0,0), (0,0), col)
    # Call Strassen for B_right (k=2,3)
    col = strassen_2x2x2((0,0), (0,2), (0,2), col)
    
    # --- Part 2: A[0:2, 2:3] * B[2:3, 0:4] ---
    # A indices: i in {0,1}, j=2
    # B indices: j=2, k in {0,1,2,3}
    # C indices: i in {0,1}, k in {0,1,2,3}
    for i in range(2):
        for k in range(4):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # --- Part 3: A[2:3, 0:2] * B[0:2, 0:4] ---
    # A indices: i=2, j in {0,1}
    # B indices: j in {0,1}, k in {0,1,2,3}
    # C indices: i=2, k in {0,1,2,3}
    for j in range(2):
        for k in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    # --- Part 4: A[2:3, 2:3] * B[2:3, 0:4] ---
    # A indices: i=2, j=2
    # B indices: j=2, k in {0,1,2,3}
    # C indices: i=2, k in {0,1,2,3}
    for k in range(4):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
