import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = 56
    
    # Initialize U, V, W as lists of lists to match the expected interface
    # U: (n*m, rank) -> (8, 56)
    # V: (m*p, rank) -> (32, 56)
    # W: (p*n, rank) -> (16, 56)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # We define the coefficients for U, V, W for a single 2x2x2 block.
    # Indices for local 2x2 matrix A: (0,0), (0,1), (1,0), (1,1) -> flat idx 0,1,2,3
    # Indices for local 2x2 matrix B: (0,0), (0,1), (1,0), (1,1) -> flat idx 0,1,2,3
    # Indices for local 2x2 matrix C: (0,0), (0,1), (1,0), (1,1) -> flat idx 0,1,2,3
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U (A factors), V (B factors), W (C factors)
    # We store as list of (index, value) pairs for each rank r (0 to 6)
    
    # r=0: M1
    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    
    # r=1: M2
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    
    # r=2: M3
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    
    # r=3: M4
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    
    # r=4: M5
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    
    # r=5: M6
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    
    # r=6: M7
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]
    
    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]
    
    # The problem is 2x4 * 4x8.
    # We can view this as 8 independent 2x2 * 2x2 multiplications? No.
    # A is 2x4. B is 4x8.
    # We can partition A into two 2x2 blocks horizontally: A_left (cols 0-1), A_right (cols 2-3).
    # We can partition B into four 2x2 blocks vertically: B_00 (rows 0-1, cols 0-1), B_01 (rows 0-1, cols 2-3), etc.
    # Actually, it's easier to think of it as 8 sub-blocks of C, each 2x2.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Let's partition J dimension (size 4) into two halves of size 2.
    # A is 2x4 -> split into A_L (2x2) and A_R (2x2).
    # B is 4x8 -> split rows into B_T (2x8) and B_B (2x8).
    # C = A_L @ B_T + A_R @ B_B.
    
    # A_L is 2x2. B_T is 2x8.
    # B_T can be split into 4 blocks of 2x2: B_T_0, B_T_1, B_T_2, B_T_3.
    # A_L @ B_T = [A_L @ B_T_0, A_L @ B_T_1, A_L @ B_T_2, A_L @ B_T_3].
    # Each A_L @ B_T_k is a 2x2 multiplication.
    # So we have 4 such multiplications from the A_L part.
    
    # Similarly, A_R is 2x2. B_B is 2x8.
    # B_B can be split into 4 blocks of 2x2: B_B_0, B_B_1, B_B_2, B_B_3.
    # A_R @ B_B = [A_R @ B_B_0, A_R @ B_B_1, A_R @ B_B_2, A_R @ B_B_3].
    # So we have 4 such multiplications from the A_R part.
    
    # Total 8 Strassen decompositions. Each uses rank 7. Total rank 56.
    
    def add_block(base_rank, i_offset, j_offset, k_offset):
        """
        Adds a Strassen decomposition for a 2x2 A block at (i_offset, j_offset)
        and a 2x2 B block at (j_offset, k_offset).
        i_offset: row offset in A (0 or 1 is not needed as A rows are 0-1, but i_offset for C/W)
        Actually, A is 2x4. Rows are 0-1. Cols are 0-3.
        B is 4x8. Rows are 0-3. Cols are 0-7.
        C is 2x8. Rows are 0-1. Cols are 0-7.
        
        The local 2x2 A block is always rows 0-1 of A, but cols vary.
        So for A_L, j_offset=0. For A_R, j_offset=2.
        The local 2x2 B block has row offset j_offset (must match A's col offset) and col offset k_offset.
        The local 2x2 C block has row offset 0 (since A has only 2 rows) and col offset k_offset.
        """
        for r in range(7):
            current_rank = base_rank + r
            
            # Update U (A factors)
            # A global index: row * m + col. m=4.
            # Local A indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i # A rows are always 0-1
                global_j = local_j + j_offset
                U[global_i * m + global_j][current_rank] += val
            
            # Update V (B factors)
            # B global index: row * p + col. p=8.
            # Local B indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * p + global_k][current_rank] += val
            
            # Update W (C factors)
            # C global index: col * n + row (Trace convention W[k*n+i, r])
            # Local C indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i # C rows are always 0-1
                W[global_k * n + global_i][current_rank] += val

    # Apply 8 blocks
    # Block 1: A_L (j=0) * B_T_0 (j=0, k=0)
    add_block(0, 0, 0)
    # Block 2: A_L (j=0) * B_T_1 (j=0, k=2)
    add_block(7, 0, 2)
    # Block 3: A_L (j=0) * B_T_2 (j=0, k=4)
    add_block(14, 0, 4)
    # Block 4: A_L (j=0) * B_T_3 (j=0, k=6)
    add_block(21, 0, 6)
    
    # Block 5: A_R (j=2) * B_B_0 (j=2, k=0)
    add_block(28, 2, 0)
    # Block 6: A_R (j=2) * B_B_1 (j=2, k=2)
    add_block(35, 2, 2)
    # Block 7: A_R (j=2) * B_B_2 (j=2, k=4)
    add_block(42, 2, 4)
    # Block 8: A_R (j=2) * B_B_3 (j=2, k=6)
    add_block(49, 2, 6)
    
    # Check deadline
    if time.time() > deadline:
        # Return partial or best known, but we constructed it already
        pass
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
