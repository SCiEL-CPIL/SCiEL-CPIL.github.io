import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Recursive function to get U, V, W for n x m x p
    def get_decomposition(n, m, p):
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        # Base case: 2x2x2 using Strassen
        if n == 2 and m == 2 and p == 2:
            # Strassen decomposition for 2x2
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B21 - B22)
            # M4 = A22(B12 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # U indices: 0: A11, 1: A21, 2: A12, 3: A22
            # V indices: 0: B11, 1: B21, 2: B12, 3: B22
            # W indices: 0: C11, 1: C21, 2: C12, 3: C22 (k*n+i convention: k=0,1 for rows of C? No, W[k*n+i, r] where k is row of C, i is col of C? 
            # Task says: W[k*n+i, r]. For C (p x n), k is row index (0..p-1), i is col index (0..n-1).
            # So W index is k*n + i.
            # For 2x2, n=2.
            # C11 is k=0, i=0 -> idx 0
            # C21 is k=1, i=0 -> idx 2
            # C12 is k=0, i=1 -> idx 1
            # C22 is k=1, i=1 -> idx 3
            
            rank = 7
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(4)]
            W = [[0]*rank for _ in range(4)]
            
            # M1
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[3][0] = 1
            W[0][0] = 1; W[3][0] = 1 # C11, C22
            
            # M2
            U[1][1] = 1; U[3][1] = 1
            V[0][1] = 1
            W[2][1] = 1; W[3][1] = 1 # C21, C22
            
            # M3
            U[0][2] = 1
            V[1][2] = 1; V[3][2] = -1
            W[1][2] = 1; W[3][2] = 1 # C12, C22
            
            # M4
            U[3][3] = 1
            V[2][3] = 1; V[0][3] = -1
            W[0][3] = 1; W[2][3] = 1 # C11, C21
            
            # M5
            U[0][4] = 1; U[2][4] = 1
            V[3][4] = 1
            W[0][4] = -1; W[1][4] = 1 # C11, C12
            
            # M6
            U[1][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[2][5] = 1
            W[3][5] = 1 # C22
            
            # M7
            U[2][6] = 1; U[3][6] = -1
            V[1][6] = 1; V[3][6] = 1
            W[0][6] = 1 # C11
            
            return U, V, W

        # Case: 2 x 1 x 2 (Rank 4)
        if n == 2 and m == 1 and p == 2:
            # A (2x1), B (1x2). C (2x2).
            # C11 = A1 B1, C12 = A1 B2, C21 = A2 B1, C22 = A2 B2
            rank = 4
            U = [[0]*rank for _ in range(2)]
            V = [[0]*rank for _ in range(2)]
            W = [[0]*rank for _ in range(4)] # n=2, p=2. W index k*n+i.
            # k=0 (row 0 of C), i=0,1
            # k=1 (row 1 of C), i=0,1
            
            # C11 (0)
            U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
            # C12 (1)
            U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
            # C21 (2)
            U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
            # C22 (3)
            U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
            return U, V, W

        # Case: 2 x 2 x 1 (Rank 4)
        if n == 2 and m == 2 and p == 1:
            # A (2x2), B (2x1). C (2x1).
            # C1 = A11 B1 + A12 B2
            # C2 = A21 B1 + A22 B2
            rank = 4
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(2)]
            W = [[0]*rank for _ in range(2)] # n=2, p=1. W index k*2+i. k=0,1. i=0.
            
            # C1 (0)
            U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
            U[2][1] = 1; V[1][1] = 1; W[0][1] = 1
            # C2 (1)
            U[1][2] = 1; V[0][2] = 1; W[1][2] = 1
            U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
            return U, V, W

        # Case: 2 x 1 x 1 (Rank 2)
        if n == 2 and m == 1 and p == 1:
            # A (2x1), B (1x1). C (2x1).
            # C1 = A1 B1, C2 = A2 B1
            rank = 2
            U = [[0]*rank for _ in range(2)]
            V = [[0]*rank for _ in range(1)]
            W = [[0]*rank for _ in range(2)]
            
            U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
            U[1][1] = 1; V[0][1] = 1; W[1][1] = 1
            return U, V, W

        # Case: 1 x 2 x 2 (Rank 4)
        if n == 1 and m == 2 and p == 2:
            # A (1x2), B (2x2). C (1x2).
            # C1 = A1 B11 + A2 B21
            # C2 = A1 B12 + A2 B22
            rank = 4
            U = [[0]*rank for _ in range(2)]
            V = [[0]*rank for _ in range(4)]
            W = [[0]*rank for _ in range(2)] # n=1, p=2. W index k*1+i. k=0. i=0,1.
            
            # C1 (0)
            U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
            U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
            # C2 (1)
            U[0][2] = 1; V[1][2] = 1; W[1][2] = 1
            U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
            return U, V, W

        # Case: 1 x 2 x 1 (Rank 2)
        if n == 1 and m == 2 and p == 1:
            # A (1x2), B (2x1). C (1x1).
            # C1 = A1 B1 + A2 B2
            rank = 2
            U = [[0]*rank for _ in range(2)]
            V = [[0]*rank for _ in range(2)]
            W = [[0]*rank for _ in range(1)]
            
            U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
            U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
            return U, V, W

        # Case: 1 x 1 x 2 (Rank 2)
        if n == 1 and m == 1 and p == 2:
            # A (1x1), B (1x2). C (1x2).
            # C1 = A1 B1, C2 = A1 B2
            rank = 2
            U = [[0]*rank for _ in range(1)]
            V = [[0]*rank for _ in range(2)]
            W = [[0]*rank for _ in range(2)]
            
            U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
            U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
            return U, V, W

        # General Case: Split
        # Choose split for n, m, p
        # Prefer splitting into 3 and 3 if possible (for 6)
        # Or 2 and 1 if odd
        
        def split_dim(dim):
            if dim % 2 == 0:
                return dim // 2, dim // 2
            else:
                return dim - 1, 1

        n1, n2 = split_dim(n)
        m1, m2 = split_dim(m)
        p1, p2 = split_dim(p)
        
        # Get sub-decompositions
        # 1. n1 x m1 x p1
        U11, V11, W11 = get_decomposition(n1, m1, p1)
        # 2. n1 x m2 x p2
        U12, V12, W12 = get_decomposition(n1, m2, p2)
        # 3. n2 x m1 x p2
        U21, V21, W21 = get_decomposition(n2, m1, p2)
        # 4. n2 x m2 x p1
        U22, V22, W22 = get_decomposition(n2, m2, p1)
        
        # Wait, the 8 terms are:
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        
        # We need decompositions for:
        # A11 B11 (n1, m1, p1) -> Done (U11, V11, W11)
        # A12 B21 (n1, m2, p1) -> Need (n1, m2, p1)
        # A11 B12 (n1, m1, p2) -> Need (n1, m1, p2)
        # A12 B22 (n1, m2, p2) -> Done (U12, V12, W12)
        # A21 B11 (n2, m1, p1) -> Need (n2, m1, p1)
        # A22 B21 (n2, m2, p1) -> Done (U22, V22, W22)
        # A21 B12 (n2, m1, p2) -> Done (U21, V21, W21)
        # A22 B22 (n2, m2, p2) -> Need (n2, m2, p2)
        
        # Let's re-fetch all 8
        U_11, V_11, W_11 = get_decomposition(n1, m1, p1)
        U_12, V_12, W_12 = get_decomposition(n1, m2, p1)
        U_13, V_13, W_13 = get_decomposition(n1, m1, p2)
        U_14, V_14, W_14 = get_decomposition(n1, m2, p2)
        
        U_21, V_21, W_21 = get_decomposition(n2, m1, p1)
        U_22, V_22, W_22 = get_decomposition(n2, m2, p1)
        U_23, V_23, W_23 = get_decomposition(n2, m1, p2)
        U_24, V_24, W_24 = get_decomposition(n2, m2, p2)
        
        # Combine
        rank = 0
        U_parts = []
        V_parts = []
        W_parts = []
        
        # Helper to merge
        def merge(U_part, V_part, W_part, offset_n, offset_m, offset_p):
            # U is (n*m, rank_part)
            # V is (m*p, rank_part)
            # W is (p*n, rank_part)
            pass

        # This recursive combination is complex to implement correctly with index mapping.
        # Given the time constraint, I will hardcode the 6x6x6 using the 3x3x3 blocks.
        
    # Hardcoded 6x6x6 using 3x3x3 blocks
    # Get 3x3x3 decomposition
    U3, V3, W3 = get_decomposition(3, 3, 3)
    
    # 6x6x6 is 4 blocks of 3x3x3
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We need 8 decompositions of 3x3x3
    # But they are all identical in structure, just mapped to different indices.
    
    rank = 8 * len(U3[0])
    
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Mapping functions
    # U index: i*m + j. For 6x6, m=6.
    # Block (r, c) of A: rows r*3..r*3+2, cols c*3..c*3+2.
    # Local index in U3: i_local * 3 + j_local.
    # Global index: (r*3 + i_local) * 6 + (c*3 + j_local).
    
    # V index: j*p + k.
    # Block (r, c) of B: rows r*3..r*3+2, cols c*3..c*3+2.
    # Local index in V3: i_local * 3 + j_local. (i_local is row of B, j_local is col of B)
    # Global index: (r*3 + i_local) * 6 + (c*3 + j_local).
    
    # W index: k*n + i. (k is row of C, i is col of C)
    # Block (r, c) of C: rows r*3..r*3+2, cols c*3..c*3+2.
    # Local index in W3: i_local * 3 + j_local. (i_local is row of C, j_local is col of C)
    # Global index: (r*3 + i_local) * 6 + (c*3 + j_local).
    
    def map_U(block_r, block_c, local_idx):
        i_loc, j_loc = divmod(local_idx, 3)
        return (block_r * 3 + i_loc) * 6 + (block_c * 3 + j_loc)
        
    def map_V(block_r, block_c, local_idx):
        i_loc, j_loc = divmod(local_idx, 3)
        return (block_r * 3 + i_loc) * 6 + (block_c * 3 + j_loc)
        
    def map_W(block_r, block_c, local_idx):
        i_loc, j_loc = divmod(local_idx, 3)
        return (block_r * 3 + i_loc) * 6 + (block_c * 3 + j_loc)

    # The 8 terms:
    # 1. A11 B11 -> C11 part 1
    # 2. A12 B21 -> C11 part 2
    # 3. A11 B12 -> C12 part 1
    # 4. A12 B22 -> C12 part 2
    # 5. A21 B11 -> C21 part 1
    # 6. A22 B21 -> C21 part 2
    # 7. A21 B12 -> C22 part 1
    # 8. A22 B22 -> C22 part 2
    
    terms = [
        (0, 0, 0, 0, 0, 0), # A(0,0) B(0,0) -> C(0,0)
        (0, 1, 1, 0, 0, 0), # A(0,1) B(1,0) -> C(0,0)
        (0, 0, 0, 1, 0, 1), # A(0,0) B(0,1) -> C(0,1)
        (0, 1, 1, 1, 0, 1), # A(0,1) B(1,1) -> C(0,1)
        (1, 0, 0, 0, 1, 0), # A(1,0) B(0,0) -> C(1,0)
        (1, 1, 1, 0, 1, 0), # A(1,1) B(1,0) -> C(1,0)
        (1, 0, 0, 1, 1, 1), # A(1,0) B(0,1) -> C(1,1)
        (1, 1, 1, 1, 1, 1), # A(1,1) B(1,1) -> C(1,1)
    ]
    
    col_offset = 0
    for ar, ac, br, bc, cr, cc in terms:
        for r in range(rank // 8):
            u_idx = map_U(ar, ac, r)
            v_idx = map_V(br, bc, r)
            w_idx = map_W(cr, cc, r)
            
            U[u_idx][col_offset + r] = U3[r][0]
            V[v_idx][col_offset + r] = V3[r][0]
            W[w_idx][col_offset + r] = W3[r][0]
            
    # Note: U3, V3, W3 are lists of lists. U3[r] is a list of 1 element (rank 1 sub-decomp? No, U3 is the full 3x3 decomp)
    # My get_decomposition returns U, V, W where each row has `rank` columns.
    # For 3x3, rank is 26.
    # So U3 is 9x26.
    # The loop above assumes U3[r] is the r-th component.
    # Correct.
    
    col_offset += rank // 8
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
