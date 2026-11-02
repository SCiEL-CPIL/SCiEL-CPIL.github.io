import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def get_strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7).
    U, V, W rows correspond to the 4 elements of the 2x2 matrices (row-major: 00, 01, 10, 11).
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (for A blocks): A11, A12, A21, A22
    # V vectors (for B blocks): B11, B12, B21, B22
    # W vectors (for C blocks): C11, C12, C21, C22
    
    # M1
    U[:, 0] = [1, 0, 0, 1]  # A11 + A22
    V[:, 0] = [1, 0, 0, 1]  # B11 + B22
    W[:, 0] = [1, 0, 0, 1]  # +C11 +C22
    
    # M2
    U[:, 1] = [0, 0, 1, 1]  # A21 + A22
    V[:, 1] = [1, 0, 0, 0]  # B11
    W[:, 1] = [0, 0, 1, -1]  # +C21 -C22
    
    # M3
    U[:, 2] = [1, 0, 0, 0]  # A11
    V[:, 2] = [0, 0, 1, -1]  # B21 - B22
    W[:, 2] = [0, 1, 0, 1]  # +C12 +C22
    
    # M4
    U[:, 3] = [0, 0, 0, 1]  # A22
    V[:, 3] = [0, 1, -1, 0]  # B12 - B11
    W[:, 3] = [1, 0, 1, 0]  # +C11 +C21
    
    # M5
    U[:, 4] = [1, 1, 0, 0]  # A11 + A12
    V[:, 4] = [0, 0, 0, 1]  # B22
    W[:, 4] = [-1, 1, 0, 0]  # -C11 +C12
    
    # M6
    U[:, 5] = [0, 0, 1, -1]  # A21 - A11
    V[:, 5] = [1, 1, 0, 0]  # B11 + B12
    W[:, 5] = [0, 0, 0, 1]  # +C22
    
    # M7
    U[:, 6] = [0, 1, 0, -1]  # A12 - A22
    V[:, 6] = [0, 0, 1, 1]  # B21 + B22
    W[:, 6] = [1, 0, 0, 0]  # +C11
    
    return U, V, W

def get_laderman_3x3():
    """
    Laderman's decomposition for 3x3 * 3x3 -> 3x3.
    Rank 15.
    Returns U (9x15), V (9x15), W (9x15).
    """
    rank = 15
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Laderman's 15 products for 3x3 matrix multiplication
    # Indices 0..8 correspond to 3x3 matrix entries (row-major)
    
    # The following is a known rank-15 decomposition for 3x3 matmul.
    # Reference: Laderman, "A non-Strassen algorithm for computing 3x3 matrix products", SIAM J. Algebraic Discrete Methods, 1976.
    
    # Let A, B be 3x3.
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A13+A23)(B21+B31)
    # P3 = (A11+A33)(B11+B33)
    # P4 = (A22+A33)(B22+B33)
    # P5 = (A13+A23+A33)(B11+B22+B33)
    # P6 = (A11+A13+A22+A23)(B11+B12+B21+B22)
    # P7 = (A11+A23)(B11+B22)
    # P8 = (A13+A22)(B12+B21)
    # P9 = (A11+A33)(B11+B22)
    # P10 = (A13+A33)(B12+B21)
    # P11 = (A22+A33)(B22+B33)
    # P12 = (A23+A33)(B21+B31)
    # P13 = (A11+A22+A33)(B11+B22+B33)
    # P14 = (A13+A22+A33)(B12+B21+B33)
    # P15 = (A11+A23+A33)(B11+B22+B13)
    
    # This specific set of products is one valid rank-15 decomposition.
    # Let's define the coefficients for U, V, W.
    # U[i, r] is coeff of A[i] in product r
    # V[j, r] is coeff of B[j] in product r
    # W[k, r] is coeff of C[k] in product r
    
    # Indices:
    # 0:11, 1:12, 2:13
    # 3:21, 4:22, 5:23
    # 6:31, 7:32, 8:33
    
    # P1: (A0+A4)(B0+B4)
    U[:, 0] = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    V[:, 0] = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    # C0 = P1 - P3 - P5 + P6 + P7 - P9 + P13 - P15 (This reconstruction is complex, let's use a verified set)
    
    # A simpler verified set for 3x3 rank 15:
    # See "Tensor decompositions and neural networks" or similar sources for explicit coefficients.
    # For safety and speed, if Laderman implementation is tricky, we can fall back to schoolbook (rank 27)
    # Total rank with schoolbook 3x3:
    # C11: 7 + 12 = 19
    # C12: 12 + 18 = 30
    # C21: 12 + 18 = 30
    # C22: 18 + 27 = 45
    # Total: 19 + 30 + 30 + 45 = 124. This is barely better than 125.
    # We MUST use Laderman to get a significant improvement.
    
    # Let's use the standard Laderman decomposition coefficients.
    # I will hardcode the known valid Laderman coefficients.
    
    # U (9x15)
    U[0, 0] = 1; U[4, 0] = 1; U[6, 0] = 1; U[7, 0] = 1; U[8, 0] = 1
    U[0, 1] = 1; U[4, 1] = 1; U[6, 1] = 1; U[7, 1] = 1; U[8, 1] = 1
    U[0, 2] = 1; U[4, 2] = 1; U[6, 2] = 1; U[7, 2] = 1; U[8, 2] = 1
    U[0, 3] = 1; U[4, 3] = 1; U[6, 3] = 1; U[7, 3] = 1; U[8, 3] = 1
    U[0, 4] = 1; U[4, 4] = 1; U[6, 4] = 1; U[7, 4] = 1; U[8, 4] = 1
    U[0, 5] = 1; U[4, 5] = 1; U[6, 5] = 1; U[7, 5] = 1; U[8, 5] = 1
    U[0, 6] = 1; U[4, 6] = 1; U[6, 6] = 1; U[7, 6] = 1; U[8, 6] = 1
    U[0, 7] = 1; U[4, 7] = 1; U[6, 7] = 1; U[7, 7] = 1; U[8, 7] = 1
    U[0, 8] = 1; U[4, 8] = 1; U[6, 8] = 1; U[7, 8] = 1; U[8, 8] = 1
    U[0, 9] = 1; U[4, 9] = 1; U[6, 9] = 1; U[7, 9] = 1; U[8, 9] = 1
    U[0, 10] = 1; U[4, 10] = 1; U[6, 10] = 1; U[7, 10] = 1; U[8, 10] = 1
    U[0, 11] = 1; U[4, 11] = 1; U[6, 11] = 1; U[7, 11] = 1; U[8, 11] = 1
    U[0, 12] = 1; U[4, 12] = 1; U[6, 12] = 1; U[7, 12] = 1; U[8, 12] = 1
    U[0, 13] = 1; U[4, 13] = 1; U[6, 13] = 1; U[7, 13] = 1; U[8, 13] = 1
    U[0, 14] = 1; U[4, 14] = 1; U[6, 14] = 1; U[7, 14] = 1; U[8, 14] = 1
    
    # This hardcoded attempt is likely incorrect without verification. 
    # Given the risk of an invalid decomposition, and the tight time limit, 
    # I will implement a robust schoolbook for 3x3 (rank 27) 
    # BUT I will try to optimize the 2x3 and 3x2 blocks if possible? No, they are minimal.
    # 
    # Wait, there is a simpler way. 
    # If I can't guarantee Laderman, I should stick to a decomposition I am 100% sure is correct.
    # Schoolbook 5x5 is 125.
    # Block decomposition with Strassen (7) and Schoolbook 3x3 (27):
    # C11: 7 + 12 = 19
    # C12: 12 + 18 = 30
    # C21: 12 + 18 = 30
    # C22: 18 + 27 = 45
    # Total: 124.
    # 
    # Is 124 better than 125? Yes.
    # Is it worth the complexity? Yes, it's a valid improvement.
    # 
    # Let's implement the block decomposition with Strassen for 2x2 and Schoolbook for everything else.
    
    # For now, let's just return schoolbook 3x3 to ensure correctness, 
    # but structure the code to allow swapping in Laderman later if verified.
    
    # Actually, let's try to use the known Laderman decomposition from a reliable source.
    # The following is a verified rank-15 decomposition for 3x3:
    
    U = np.zeros((9, 15))
    V = np.zeros((9, 15))
    W = np.zeros((9, 15))
    
    # Indices: 0,1,2,3,4,5,6,7,8
    # P1 = (A0+A4)(B0+B4)
    U[0,0]=1; U[4,0]=1; V[0,0]=1; V[4,0]=1; W[0,0]=1; W[4,0]=1
    # P2 = (A2+A5)(B3+B6)
    U[2,1]=1; U[5,1]=1; V[3,1]=1; V[6,1]=1; W[2,1]=1; W[5,1]=1
    # P3 = (A0+A8)(B0+B8)
    U[0,2]=1; U[8,2]=1; V[0,2]=1; V[8,2]=1; W[0,2]=1; W[8,2]=1
    # P4 = (A4+A8)(B4+B8)
    U[4,3]=1; U[8,3]=1; V[4,3]=1; V[8,3]=1; W[4,3]=1; W[8,3]=1
    # P5 = (A2+A5+A8)(B0+B4+B8)
    U[2,4]=1; U[5,4]=1; U[8,4]=1; V[0,4]=1; V[4,4]=1; V[8,4]=1; W[2,4]=1; W[5,4]=1; W[8,4]=1
    # P6 = (A0+A2+A4+A5)(B0+B1+B3+B4)
    U[0,5]=1; U[2,5]=1; U[4,5]=1; U[5,5]=1; V[0,5]=1; V[1,5]=1; V[3,5]=1; V[4,5]=1; W[0,5]=1; W[2,5]=1; W[4,5]=1; W[5,5]=1
    # P7 = (A0+A5)(B0+B4)
    U[0,6]=1; U[5,6]=1; V[0,6]=1; V[4,6]=1; W[0,6]=1; W[4,6]=1
    # P8 = (A2+A4)(B1+B3)
    U[2,7]=1; U[4,7]=1; V[1,7]=1; V[3,7]=1; W[2,7]=1; W[4,7]=1
    # P9 = (A0+A8)(B0+B4)
    U[0,8]=1; U[8,8]=1; V[0,8]=1; V[4,8]=1; W[0,8]=1; W[4,8]=1
    # P10 = (A2+A8)(B1+B3)
    U[2,9]=1; U[8,9]=1; V[1,9]=1; V[3,9]=1; W[2,9]=1; W[4,9]=1
    # P11 = (A4+A8)(B4+B8)
    U[4,10]=1; U[8,10]=1; V[4,10]=1; V[8,10]=1; W[4,10]=1; W[8,10]=1
    # P12 = (A5+A8)(B3+B6)
    U[5,11]=1; U[8,11]=1; V[3,11]=1; V[6,11]=1; W[5,11]=1; W[8,11]=1
    # P13 = (A0+A4+A8)(B0+B4+B8)
    U[0,12]=1; U[4,12]=1; U[8,12]=1; V[0,12]=1; V[4,12]=1; V[8,12]=1; W[0,12]=1; W[4,12]=1; W[8,12]=1
    # P14 = (A2+A4+A8)(B1+B3+B8)
    U[2,13]=1; U[4,13]=1; U[8,13]=1; V[1,13]=1; V[3,13]=1; V[8,13]=1; W[2,13]=1; W[4,13]=1; W[8,13]=1
    # P15 = (A0+A5+A8)(B0+B4+B1)
    U[0,14]=1; U[5,14]=1; U[8,14]=1; V[0,14]=1; V[4,14]=1; V[1,14]=1; W[0,14]=1; W[4,14]=1; W[1,14]=1
    
    # This specific set of products is NOT guaranteed to reconstruct C correctly without the correct linear combination in W.
    # The W matrix above just sets 1s where the product contributes, but the signs and coefficients for the final C are missing.
    # Implementing Laderman correctly is error-prone.
    
    # SAFETY FALLBACK: Use Schoolbook for 3x3.
    # Rank 27.
    # Total Rank: 124.
    
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + k, col] = 1
                V[k * 3 + j, col] = 1
                W[i * 3 + j, col] = 1
                col += 1
                
    return U, V, W

def get_schoolbook_rect(m, n, p):
    """
    Schoolbook decomposition for (m x n) * (n x p) -> (m x p).
    Returns U (m*n x m*n*p), V (n*p x m*n*p), W (m*p x m*n*p).
    Note: U, V, W rows are flattened row-major.
    """
    rank = m * n * p
    U = np.zeros((m * n, rank))
    V = np.zeros((n * p, rank))
    W = np.zeros((m * p, rank))
    
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j, col] = 1
                V[j * p + k, col] = 1
                W[i * p + k, col] = 1
                col += 1
                
    return U, V, W

def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    Uses block decomposition: 5 = 2 + 3.
    - 2x2*2x2: Strassen (rank 7)
    - 3x3*3x3: Schoolbook (rank 27) [Laderman is risky to implement correctly here]
    - Rectangular: Schoolbook
    
    Total Rank:
    C11 (2x2): 7 (Strassen) + 12 (2x3*3x2) = 19
    C12 (2x3): 12 (2x2*2x3) + 18 (2x3*3x3) = 30
    C21 (3x2): 12 (3x2*2x2) + 18 (3x3*3x2) = 30
    C22 (3x3): 18 (3x2*2x3) + 27 (3x3*3x3) = 45
    Total: 19 + 30 + 30 + 45 = 124
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS # 5, 5, 5
    
    # Global dimensions
    size_A = 25
    size_B = 25
    size_C = 25
    
    # Total rank
    rank_11 = 7 + 12
    rank_12 = 12 + 18
    rank_21 = 12 + 18
    rank_22 = 18 + 27
    total_rank = rank_11 + rank_12 + rank_21 + rank_22
    
    U = np.zeros((size_A, total_rank))
    V = np.zeros((size_B, total_rank))
    W = np.zeros((size_C, total_rank))
    
    # Helper to add a block decomposition
    # u_offset: start row in U
    # v_offset: start row in V
    # w_offset: start row in W
    # u_indices: list of global U indices corresponding to the block's local rows
    # v_indices: list of global V indices corresponding to the block's local rows
    # w_indices: list of global W indices corresponding to the block's local rows
    # u_block, v_block, w_block: local decomposition matrices
    
    def add_block(u_indices, v_indices, w_indices, U_block, V_block, W_block, start_col):
        local_rank = U_block.shape[1]
        for lr in range(local_rank):
            for lu in range(U_block.shape[0]):
                if U_block[lu, lr] != 0:
                    U[u_indices[lu], start_col + lr] = U_block[lu, lr]
            for lv in range(V_block.shape[0]):
                if V_block[lv, lr] != 0:
                    V[v_indices[lv], start_col + lr] = V_block[lv, lr]
            for lw in range(W_block.shape[0]):
                if W_block[lw, lr] != 0:
                    W[w_indices[lw], start_col + lr] = W_block[lw, lr]
        return start_col + local_rank
    
    # Define global indices for blocks
    # A (25x1) flattened row-major: A[i, j] -> i*5 + j
    # B (25x1) flattened row-major: B[j, k] -> j*5 + k
    # C (25x1) flattened trace: C[i, k] -> W[k*5 + i]
    
    # Block 1: A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # A11 indices: 0, 1, 5, 6
    # B11 indices: 0, 1, 5, 6
    # C11 indices (W): C[0,0]->0, C[0,1]->1, C[1,0]->5, C[1,1]->6
    # Wait, W[k*5+i]. 
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 5
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 6
    # So W indices for C11 are [0, 1, 5, 6] corresponding to local C rows [0, 1, 2, 3] (row-major C: 00, 01, 10, 11)
    # Local C row 0 (00) -> global 0
    # Local C row 1 (01) -> global 5
    # Local C row 2 (10) -> global 1
    # Local C row 3 (11) -> global 6
    
    u_idx_11 = [0, 1, 5, 6]
    v_idx_11 = [0, 1, 5, 6]
    w_idx_11 = [0, 5, 1, 6] # Mapping local 0,1,2,3 to global 0,5,1,6
    
    U_11, V_11, W_11 = get_strassen_2x2()
    col = add_block(u_idx_11, v_idx_11, w_idx_11, U_11, V_11, W_11, 0)
    
    # Block 2: A12 (2x3) * B21 (3x2) -> C11 (2x2)
    # A12 indices: A[0,2]=2, A[0,3]=3, A[0,4]=4, A[1,2]=7, A[1,3]=8, A[1,4]=9
    # B21 indices: B[2,0]=10, B[2,1]=11, B[3,0]=15, B[3,1]=16, B[4,0]=20, B[4,1]=21
    # C11 indices: same as above [0, 5, 1, 6]
    
    u_idx_12_11 = [2, 3, 4, 7, 8, 9]
    v_idx_12_11 = [10, 11, 15, 16, 20, 21]
    w_idx_12_11 = [0, 5, 1, 6]
    
    U_12_11, V_12_11, W_12_11 = get_schoolbook_rect(2, 3, 2)
    col = add_block(u_idx_12_11, v_idx_12_11, w_idx_12_11, U_12_11, V_12_11, W_12_11, col)
    
    # Block 3: A11 (2x2) * B12 (2x3) -> C12 (2x3)
    # A11 indices: [0, 1, 5, 6]
    # B12 indices: B[0,2]=2, B[0,3]=3, B[0,4]=4, B[1,2]=7, B[1,3]=8, B[1,4]=9
    # C12 indices: C[0,2]=10, C[0,3]=15, C[0,4]=20, C[1,2]=11, C[1,3]=16, C[1,4]=21
    # Local C (2x3) rows: 0(00), 1(01), 2(02), 3(10), 4(11), 5(12)
    # Global W indices for C12:
    # C[0,2] -> k=2, i=0 -> 10
    # C[0,3] -> k=3, i=0 -> 15
    # C[0,4] -> k=4, i=0 -> 20
    # C[1,2] -> k=2, i=1 -> 11
    # C[1,3] -> k=3, i=1 -> 16
    # C[1,4] -> k=4, i=1 -> 21
    # So w_idx is [10, 15, 20, 11, 16, 21]
    
    u_idx_11_12 = [0, 1, 5, 6]
    v_idx_11_12 = [2, 3, 4, 7, 8, 9]
    w_idx_11_12 = [10, 15, 20, 11, 16, 21]
    
    U_11_12, V_11_12, W_11_12 = get_schoolbook_rect(2, 2, 3)
    col = add_block(u_idx_11_12, v_idx_11_12, w_idx_11_12, U_11_12, V_11_12, W_11_12, col)
    
    # Block 4: A12 (2x3) * B22 (3x3) -> C12 (2x3)
    # A12 indices: [2, 3, 4, 7, 8, 9]
    # B22 indices: B[2,2]=12, B[2,3]=13, B[2,4]=14, B[3,2]=17, B[3,3]=18, B[3,4]=19, B[4,2]=22, B[4,3]=23, B[4,4]=24
    # C12 indices: same as above [10, 15, 20, 11, 16, 21]
    
    u_idx_12_12 = [2, 3, 4, 7, 8, 9]
    v_idx_12_12 = [12, 13, 14, 17, 18, 19, 22, 23, 24]
    w_idx_12_12 = [10, 15, 20, 11, 16, 21]
    
    U_12_12, V_12_12, W_12_12 = get_schoolbook_rect(2, 3, 3)
    col = add_block(u_idx_12_12, v_idx_12_12, w_idx_12_12, U_12_12, V_12_12, W_12_12, col)
    
    # Block 5: A21 (3x2) * B11 (2x2) -> C21 (3x2)
    # A21 indices: A[2,0]=10, A[2,1]=11, A[3,0]=15, A[3,1]=16, A[4,0]=20, A[4,1]=21
    # B11 indices: [0, 1, 5, 6]
    # C21 indices: C[2,0]=2, C[2,1]=7, C[3,0]=3, C[3,1]=8, C[4,0]=4, C[4,1]=9
    # Global W indices for C21:
    # C[2,0] -> k=0, i=2 -> 2
    # C[2,1] -> k=1, i=2 -> 7
    # C[3,0] -> k=0, i=3 -> 3
    # C[3,1] -> k=1, i=3 -> 8
    # C[4,0] -> k=0, i=4 -> 4
    # C[4,1] -> k=1, i=4 -> 9
    # w_idx: [2, 7, 3, 8, 4, 9]
    
    u_idx_21_11 = [10, 11, 15, 16, 20, 21]
    v_idx_21_11 = [0, 1, 5, 6]
    w_idx_21_11 = [2, 7, 3, 8, 4, 9]
    
    U_21_11, V_21_11, W_21_11 = get_schoolbook_rect(3, 2, 2)
    col = add_block(u_idx_21_11, v_idx_21_11, w_idx_21_11, U_21_11, V_21_11, W_21_11, col)
    
    # Block 6: A22 (3x3) * B21 (3x2) -> C21 (3x2)
    # A22 indices: [12, 13, 14, 17, 18, 19, 22, 23, 24]
    # B21 indices: [10, 11, 15, 16, 20, 21]
    # C21 indices: same as above [2, 7, 3, 8, 4, 9]
    
    u_idx_22_21 = [12, 13, 14, 17, 18, 19, 22, 23, 24]
    v_idx_22_21 = [10, 11, 15, 16, 20, 21]
    w_idx_22_21 = [2, 7, 3, 8, 4, 9]
    
    U_22_21, V_22_21, W_22_21 = get_schoolbook_rect(3, 3, 2)
    col = add_block(u_idx_22_21, v_idx_22_21, w_idx_22_21, U_22_21, V_22_21, W_22_21, col)
    
    # Block 7: A21 (3x2) * B12 (2x3) -> C22 (3x3)
    # A21 indices: [10, 11, 15, 16, 20, 21]
    # B12 indices: [2, 3, 4, 7, 8, 9]
    # C22 indices: C[2,2]=12, C[2,3]=17, C[2,4]=22, C[3,2]=13, C[3,3]=18, C[3,4]=23, C[4,2]=14, C[4,3]=19, C[4,4]=24
    # Global W indices for C22:
    # C[2,2] -> k=2, i=2 -> 12
    # C[2,3] -> k=3, i=2 -> 17
    # C[2,4] -> k=4, i=2 -> 22
    # C[3,2] -> k=2, i=3 -> 13
    # C[3,3] -> k=3, i=3 -> 18
    # C[3,4] -> k=4, i=3 -> 23
    # C[4,2] -> k=2, i=4 -> 14
    # C[4,3] -> k=3, i=4 -> 19
    # C[4,4] -> k=4, i=4 -> 24
    # w_idx: [12, 17, 22, 13, 18, 23, 14, 19, 24]
    
    u_idx_21_12 = [10, 11, 15, 16, 20, 21]
    v_idx_21_12 = [2, 3, 4, 7, 8, 9]
    w_idx_21_12 = [12, 17, 22, 13, 18, 23, 14, 19, 24]
    
    U_21_12, V_21_12, W_21_12 = get_schoolbook_rect(3, 2, 3)
    col = add_block(u_idx_21_12, v_idx_21_12, w_idx_21_12, U_21_12, V_21_12, W_21_12, col)
    
    # Block 8: A22 (3x3) * B22 (3x3) -> C22 (3x3)
    # A22 indices: [12, 13, 14, 17, 18, 19, 22, 23, 24]
    # B22 indices: [12, 13, 14, 17, 18, 19, 22, 23, 24]
    # C22 indices: same as above [12, 17, 22, 13, 18, 23, 14, 19, 24]
    
    u_idx_22_22 = [12, 13, 14, 17, 18, 19, 22, 23, 24]
    v_idx_22_22 = [12, 13, 14, 17, 18, 19, 22, 23, 24]
    w_idx_22_22 = [12, 17, 22, 13, 18, 23, 14, 19, 24]
    
    U_22_22, V_22_22, W_22_22 = get_laderman_3x3() # Using schoolbook fallback for safety
    col = add_block(u_idx_22_22, v_idx_22_22, w_idx_22_22, U_22_22, V_22_22, W_22_22, col)
    
    # Convert to lists for the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
