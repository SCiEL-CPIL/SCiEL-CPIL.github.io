import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-66 tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    The decomposition is based on partitioning the matrices:
    A (3x4) -> Rows 1+2, Cols 2+2
    B (4x6) -> Rows 2+2, Cols 2+2+2
    C (3x6) -> Rows 1+2, Cols 2+2+2
    
    This yields:
    - 6 blocks of 1x2 * 2x2 (rank 4 each) for the first row of C.
    - 6 blocks of 2x2 * 2x2 (rank 7 each, Strassen) for the remaining two rows of C.
    
    Total rank = 6*4 + 6*7 = 24 + 42 = 66.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = 3, 4, 6
    rank = 66
    
    # Initialize factor matrices
    # U: (n*m, rank) = (12, 66)
    # V: (m*p, rank) = (24, 66)
    # W: (p*n, rank) = (18, 66)
    # Indices:
    # A[i, j] is stored in U[i*m + j, :]
    # B[j, k] is stored in V[j*p + k, :]
    # C[i, k] is stored in W[k*n + i, :] (Trace convention: W[k*n + i, r])
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Add 1x2 * 2x2 block.
        A is 1x2 (row a_r, cols a_c, a_c+1)
        B is 2x2 (rows b_r, b_r+1, cols b_c, b_c+1)
        C is 1x2 (row c_r, cols c_c, c_c+1)
        Rank 4.
        """
        nonlocal col
        if time.time() > deadline:
            return False
            
        # A indices
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        
        # B indices
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (W[k*n + i])
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # M1: A0 * B00 -> C0
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2: A1 * B10 -> C0
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3: A0 * B01 -> C1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4: A1 * B11 -> C1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1
        
        return col < rank

    def add_strassen_2x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Add Strassen 2x2 * 2x2 block.
        A is 2x2 (rows a_r, a_r+1, cols a_c, a_c+1)
        B is 2x2 (rows b_r, b_r+1, cols b_c, b_c+1)
        C is 2x2 (rows c_r, c_r+1, cols c_c, c_c+1)
        Rank 7.
        """
        nonlocal col
        if time.time() > deadline:
            return False
            
        # A indices
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        # B indices
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (W[k*n + i])
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
        
        return col < rank

    # --- Build Decomposition ---
    
    # Row 0 of C (1x6)
    # C[0, 0:2] = A[0, 0:2] * B[0:2, 0:2] + A[0, 2:4] * B[2:4, 0:2]
    if not add_1x2_2x2(0, 0, 0, 0, 0, 0): return {"U": U, "V": V, "W": W}
    if not add_1x2_2x2(0, 2, 2, 0, 0, 0): return {"U": U, "V": V, "W": W}
    
    # C[0, 2:4] = A[0, 0:2] * B[0:2, 2:4] + A[0, 2:4] * B[2:4, 2:4]
    if not add_1x2_2x2(0, 0, 0, 2, 0, 2): return {"U": U, "V": V, "W": W}
    if not add_1x2_2x2(0, 2, 2, 2, 0, 2): return {"U": U, "V": V, "W": W}
    
    # C[0, 4:6] = A[0, 0:2] * B[0:2, 4:6] + A[0, 2:4] * B[2:4, 4:6]
    if not add_1x2_2x2(0, 0, 0, 4, 0, 4): return {"U": U, "V": V, "W": W}
    if not add_1x2_2x2(0, 2, 2, 4, 0, 4): return {"U": U, "V": V, "W": W}
    
    # Rows 1-2 of C (2x6)
    # C[1:3, 0:2] = A[1:3, 0:2] * B[0:2, 0:2] + A[1:3, 2:4] * B[2:4, 0:2]
    if not add_strassen_2x2_2x2(1, 0, 0, 0, 1, 0): return {"U": U, "V": V, "W": W}
    if not add_strassen_2x2_2x2(1, 2, 2, 0, 1, 0): return {"U": U, "V": V, "W": W}
    
    # C[1:3, 2:4] = A[1:3, 0:2] * B[0:2, 2:4] + A[1:3, 2:4] * B[2:4, 2:4]
    if not add_strassen_2x2_2x2(1, 0, 0, 2, 1, 2): return {"U": U, "V": V, "W": W}
    if not add_strassen_2x2_2x2(1, 2, 2, 2, 1, 2): return {"U": U, "V": V, "W": W}
    
    # C[1:3, 4:6] = A[1:3, 0:2] * B[0:2, 4:6] + A[1:3, 2:4] * B[2:4, 4:6]
    if not add_strassen_2x2_2x2(1, 0, 0, 4, 1, 4): return {"U": U, "V": V, "W": W}
    if not add_strassen_2x2_2x2(1, 2, 2, 4, 1, 4): return {"U": U, "V": V, "W": W}
    
    assert col == 66, f"Expected 66 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
