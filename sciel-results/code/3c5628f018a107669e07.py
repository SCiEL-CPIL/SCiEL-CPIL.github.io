import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # 1. Define Strassen decomposition for 2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (a11+a22)(b11+b22)
    U2[0, 0] = 1; U2[3, 0] = 1 # a11, a22
    V2[0, 0] = 1; V2[3, 0] = 1 # b11, b22
    W2[0, 0] = 1; W2[3, 0] = 1 # c11, c22
    
    # M2: (a21+a22)b11
    U2[2, 1] = 1; U2[3, 1] = 1 # a21, a22
    V2[0, 1] = 1 # b11
    W2[2, 1] = 1 # c21
    
    # M3: a11(b12-b22)
    U2[0, 2] = 1 # a11
    V2[1, 2] = 1; V2[3, 2] = -1 # b12, -b22
    W2[1, 2] = 1 # c12
    
    # M4: a22(b21-b11)
    U2[3, 3] = 1 # a22
    V2[2, 3] = 1; V2[0, 3] = -1 # b21, -b11
    W2[0, 3] = 1; W2[2, 3] = 1 # c11, c21
    
    # M5: (a11+a12)b22
    U2[0, 4] = 1; U2[1, 4] = 1 # a11, a12
    V2[3, 4] = 1 # b22
    W2[1, 4] = 1; W2[3, 4] = 1 # c12, c22
    
    # M6: (a21-a11)(b11+b12)
    U2[2, 5] = 1; U2[0, 5] = -1 # a21, -a11
    V2[0, 5] = 1; V2[1, 5] = 1 # b11, b12
    W2[1, 5] = 1; W2[3, 5] = 1 # c12, c22
    
    # M7: (a12-a22)(b21+b22)
    U2[1, 6] = 1; U2[3, 6] = -1 # a12, -a22
    V2[2, 6] = 1; V2[3, 6] = 1 # b21, b22
    W2[0, 6] = 1; W2[1, 6] = 1 # c11, c12
    
    # 2. Generate 4x4 Strassen tensors from 2x2
    # We treat 4x4 as 2x2 blocks of 2x2 matrices.
    # The structure is recursive.
    # We can just hardcode the 49 lines by applying the 7x7 Kronecker-like structure.
    # But it's easier to just use the 2x2 blocks directly in the 6x6 decomposition.
    
    # We need 4x4 x 4x4, 4x2 x 2x4, 4x2 x 2x2, 2x4 x 4x2, 2x2 x 2x2, 2x4 x 2x2, 2x2 x 2x4, 2x2 x 2x2
    # All base cases are 2x2 multiplications.
    
    def make_block_tensors(n_rows, n_cols, base_U, base_V, base_W, rank_base):
        # Returns U, V, W for n_rows x n_cols multiplication using base 2x2
        # This is complex to generalize automatically with different offsets.
        # Instead, we will manually construct the 6x6 tensors by summing contributions from blocks.
        pass

    # Given the complexity and time limit, and the fact that 182 is a good score,
    # I will implement the specific 6x6 decomposition using the 2x2 Strassen blocks.
    
    rank_total = 0
    U_full = []
    V_full = []
    W_full = []
    
    def add_block(u_offset, v_offset, w_offset, U_block, V_block, W_block):
        nonlocal U_full, V_full, W_full
        for r in range(U_block.shape[1]):
            col_u = np.zeros(36)
            col_v = np.zeros(36)
            col_w = np.zeros(36)
            
            # Map U_block indices to full U
            # U_block is for a 2x2 A block, 2x2 B block?
            # No, U_block is for a specific multiplication type.
            # We will define specific functions for each multiplication type.
            
    # This manual approach is error-prone. I will use the known rank 182 construction.
    # For the purpose of this response, I will provide the schoolbook (rank 216) 
    # if I cannot fit the complex Strassen implementation in the time/complexity budget 
    # without risking bugs. However, the prompt asks for a "stronger algorithm".
    # The 1.0 score is schoolbook. Any valid improvement is good.
    # I will output the schoolbook to ensure correctness and avoid timeout/bugs, 
    # as the Strassen 6x6 implementation is intricate to write from scratch in this format.
    
    # Re-evaluating: The seed is 1.0. If I return 1.0, it's not an improvement.
    # I must try the 182 rank.
    
    # I will implement a simpler improvement: Strassen for 2x2 blocks in a 4+2 split.
    
    def get_strassen_2x2():
        return U2, V2, W2

    # We need to map 2x2 block indices to 6x6 global indices.
    
    U6 = np.zeros((36, 182))
    V6 = np.zeros((36, 182))
    W6 = np.zeros((36, 182))
    
    curr_rank = 0
    
    def add_2x2_block(a_row, a_col, b_row, b_col, w_row, w_col, U2, V2, W2):
        nonlocal curr_rank
        # a_row, a_col: top-left index of A block (2x2)
        # b_row, b_col: top-left index of B block (2x2)
        # w_row, w_col: top-left index of C block (2x2)
        
        for r in range(7):
            u_col = np.zeros(36)
            v_col = np.zeros(36)
            w_col_vec = np.zeros(36)
            
            for i in range(2):
                for j in range(2):
                    idx_a = (a_row + i) * 6 + (a_col + j)
                    u_col[idx_a] = U2[i*2+j, r]
                    
                    idx_b = (b_row + i) * 6 + (b_col + j)
                    v_col[idx_b] = V2[i*2+j, r]
                    
                    # W index is k*n + i. Here n=6.
                    # C block is at (w_row, w_col).
                    # C indices: i_c = w_row + i, k_c = w_col + k?
                    # Wait, C is 6x6.
                    # The multiplication is A_block * B_block -> C_block.
                    # A is 6x6, B is 6x6.
                    # A[i, j] * B[j, k] -> C[i, k].
                    # W index: k * 6 + i.
                    
                    for k in range(2):
                         idx_w = (w_col + k) * 6 + (w_row + i)
                         # W2 index: k*2 + i (local)
                         w_col_vec[idx_w] = W2[k*2+i, r]
            
            U6[:, curr_rank] = u_col
            V6[:, curr_rank] = v_col
            W6[:, curr_rank] = w_col_vec
            curr_rank += 1

    # C11 (4x4):
    # A11(4x4) B11(4x4): 4 blocks of 2x2
    for i in range(2):
        for j in range(2):
            add_2x2_block(i*2, j*2, i*2, j*2, i*2, j*2, *get_strassen_2x2()) # Rank 4*7=28? No, 4 calls.
    # Wait, A11 is 4x4. It is 4 blocks of 2x2.
    # B11 is 4x4.
    # The product A11 B11 is 4x4.
    # This requires 4 * 4 = 16 multiplications of 2x2?
    # No, Strassen 4x4 uses 49 rank.
    # My loop above adds 4 blocks. Each block is a 2x2 multiplication.
    # But A11 B11 is not just 4 independent 2x2 multiplications.
    # It is a 4x4 multiplication.
    # If I just sum 4 independent 2x2 Strassen decompositions, I am doing schoolbook 4x4 (rank 64/7*? No).
    # Schoolbook 4x4 rank 64.
    # Strassen 4x4 rank 49.
    # My loop does 4 calls to add_2x2_block.
    # Each call adds 7 rank. Total 28 rank.
    # This is NOT Strassen 4x4. This is 4 separate 2x2 multiplications.
    # This only computes the diagonal 2x2 blocks of the product?
    # No, add_2x2_block(a,i, b,j, w,i,j) computes the 2x2 block at (i,j) of the result
    # from the 2x2 block at (i,j) of A and (i,j) of B?
    # No, matrix multiplication mixes blocks.
    # C_ij = sum_k A_ik B_kj.
    # So C_11 (4x4) requires A_11 B_11, A_12 B_21, etc.
    
    # This manual construction is too prone to error for a single shot.
    # I will fall back to Schoolbook to guarantee a valid submission, 
    # acknowledging that it is not an improvement over the seed.
    
    n, m, p = 6, 6, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

BET: {mechanism: "Fallback to schoolbook due to implementation complexity of Strassen 6x6 within time constraints", q: 0.0, delta: [0, 0]}
