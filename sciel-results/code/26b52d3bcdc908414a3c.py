"""Matrix multiplication 4x4x7 with rank reduction via 2x2x2 Strassen blocks."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Build a list of (U_col, V_col, W_col) where each is a dict from row_idx to value
    # U has rows 0..n*m-1 = 0..15
    # V has rows 0..m*p-1 = 0..27
    # W has rows 0..p*n-1 = 0..27
    
    terms = []  # list of (U_dict, V_dict, W_dict)
    
    def add_term(U_dict, V_dict, W_dict):
        terms.append((U_dict, V_dict, W_dict))
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen 2x2x2 decomposition for block at (i0, j0, k0).
        This handles A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        """
        # Row indices for U (i*m + j)
        U_i0_j0 = i0 * m + j0
        U_i0_j1 = i0 * m + (j0 + 1)
        U_i1_j0 = (i0 + 1) * m + j0
        U_i1_j1 = (i0 + 1) * m + (j0 + 1)
        
        # Row indices for V (j*p + k)
        V_j0_k0 = j0 * p + k0
        V_j0_k1 = j0 * p + (k0 + 1)
        V_j1_k0 = (j0 + 1) * p + k0
        V_j1_k1 = (j0 + 1) * p + (k0 + 1)
        
        # Row indices for W (k*n + i)
        W_k0_i0 = k0 * n + i0
        W_k0_i1 = k0 * n + (i0 + 1)
        W_k1_i0 = (k0 + 1) * n + i0
        W_k1_i1 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (A11 - A22) * (B21 + B22)
        add_term(
            {U_i0_j0: 1, U_i1_j1: -1},
            {V_j0_k1: 1, V_j1_k1: 1},
            {W_k1_i0: 1, W_k1_i1: 1, W_k0_i1: 1, W_k0_i0: 1}
        )
        
        # M2 = (A21 + A22) * B11
        add_term(
            {U_i1_j0: 1, U_i1_j1: 1},
            {V_j0_k0: 1},
            {W_k0_i0: 1, W_k0_i1: 1}
        )
        
        # M3 = A11 * (B12 - B22)
        add_term(
            {U_i0_j0: 1},
            {V_j0_k1: 1, V_j1_k1: -1},
            {W_k1_i0: 1, W_k1_i1: 1}
        )
        
        # M4 = A22 * (B11 - B12)
        add_term(
            {U_i1_j1: 1},
            {V_j0_k0: 1, V_j0_k1: -1},
            {W_k0_i1: 1, W_k0_i0: -1}
        )
        
        # M5 = (A11 + A22) * (B22 - B11)
        add_term(
            {U_i0_j0: 1, U_i1_j1: 1},
            {V_j1_k1: 1, V_j0_k0: -1},
            {W_k1_i0: -1, W_k1_i1: -1, W_k0_i1: -1, W_k0_i0: -1}
        )
        
        # M6 = (A21 - A11) * (B11 + B12)
        add_term(
            {U_i1_j0: 1, U_i0_j0: -1},
            {V_j0_k0: 1, V_j0_k1: 1},
            {W_k0_i1: 1, W_k0_i0: 1}
        )
        
        # M7 = (A12 - A22) * (B21 + B22)
        add_term(
            {U_i0_j1: 1, U_i1_j1: -1},
            {V_j1_k0: 1, V_j1_k1: 1},
            {W_k1_i0: 1, W_k1_i1: 1}
        )
    
    # We have 4x4x7. Split into blocks of 2x2x2 and 2x2x1.
    # For the k dimension (size 7), we can do:
    # - 3 blocks of 2x2x2 covering k=0,1; k=2,3; k=4,5
    # - 1 block of 2x2x1 covering k=6
    
    # For i dimension (size 4): blocks at i0=0, i0=2
    # For j dimension (size 4): blocks at j0=0, j0=2
    # For k dimension (size 7): blocks at k0=0, k0=2, k0=4 for 2x2x2, and k0=6 for 2x2x1
    
    # 2x2x2 blocks: 2 (i) * 2 (j) * 3 (k pairs) = 12 blocks, each rank 7 = 84
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen_2x2x2(i0, j0, k0)
    
    # 2x2x1 blocks for k=6: 2 (i) * 2 (j) = 4 blocks, each rank 4 (schoolbook for 2x2x1) = 16
    # For 2x2x1: C = A * B where A is 2x2, B is 2x1, C is 2x1
    # Schoolbook: 4 multiplications
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            U_i0_j0 = i0 * m + j0
            U_i0_j1 = i0 * m + (j0 + 1)
            U_i1_j0 = (i0 + 1) * m + j0
            U_i1_j1 = (i0 + 1) * m + (j0 + 1)
            
            k = 6
            V_j0_k = j0 * p + k
            V_j1_k = (j0 + 1) * p + k
            
            W_k_i0 = k * n + i0
            W_k_i1 = k * n + (i0 + 1)
            
            # C[i0, k] = A[i0, j0] * B[j0, k] + A[i0, j1] * B[j1, k]
            # C[i1, k] = A[i1, j0] * B[j0, k] + A[i1, j1] * B[j1, k]
            
            # Term 1: A[i0,j0] * B[j0,k]
            add_term(
                {U_i0_j0: 1},
                {V_j0_k: 1},
                {W_k_i0: 1}
            )
            
            # Term 2: A[i0,j1] * B[j1,k]
            add_term(
                {U_i0_j1: 1},
                {V_j1_k: 1},
                {W_k_i0: 1}
            )
            
            # Term 3: A[i1,j0] * B[j0,k]
            add_term(
                {U_i1_j0: 1},
                {V_j0_k: 1},
                {W_k_i1: 1}
            )
            
            # Term 4: A[i1,j1] * B[j1,k]
            add_term(
                {U_i1_j1: 1},
                {V_j1_k: 1},
                {W_k_i1: 1}
            )
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(terms):
        for row, val in U_dict.items():
            U[row][col] = val
        for row, val in V_dict.items():
            V[row][col] = val
        for row, val in W_dict.items():
            W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
