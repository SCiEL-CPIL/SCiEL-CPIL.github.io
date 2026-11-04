import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s buffer for evaluator overhead
    
    # 1. Build the Matrix Multiplication Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    # Indices:
    # U index: i in [0, n), j in [0, m) -> u_idx = i*m + j
    # V index: j in [0, m), k in [0, p) -> v_idx = j*p + k
    # W index: k in [0, p), i in [0, n) -> w_idx = k*n + i
    
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # We represent the tensor T as a collection of slices.
    # T_{a, b, c} where a is U-index, b is V-index, c is W-index.
    # T is non-zero only if:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # for some i, j, k.
    
    # Let's create a sparse representation or just iterate.
    # Since dimensions are small (90, 30, 18), we can work with dense arrays if needed,
    # but rank minimization is better done via iterative projection.
    
    # Strategy:
    # Start with rank R = n*m*p (schoolbook).
    # We want to find coefficients U, V, W such that T = sum_{t=1}^R U_t \otimes V_t \otimes W_t.
    # This is equivalent to finding a rank-R decomposition.
    # We can try to reduce R by checking if the current tensor can be represented with lower rank.
    
    # A robust method for small tensors is to use the "Iterative Rank-1 Deflation" or 
    # "Randomized SVD based approach" on the matricizations.
    # However, for exact integer/half-integer solutions, random floating point might drift.
    
    # Alternative Strategy: Use the known Strassen-like algorithms for small blocks and combine them.
    # The previous attempt failed due to complexity. 
    # Let's try a different structural approach: 
    # Use the fact that M(3,5,6) can be decomposed using smaller known optimal decompositions.
    # Known ranks:
    # M(2,2,2) = 7
    # M(2,2,3) = 11? (Actually M(2,2,3) is not standard. M(2,2,2)=7, M(2,2,3) <= 11 via block? No, 2x2 by 2x3.
    # Let's stick to a verified algorithmic construction that is simpler than manual Strassen for 3x5x6.
    
    # Let's use a block decomposition that is easy to verify and construct programmatically.
    # Split n=3 into 1+2
    # Split m=5 into 2+3
    # Split p=6 into 3+3
    
    # Blocks:
    # C11 (1x3) = A11(1x2) B11(2x3) + A12(1x3) B21(3x3)
    # C12 (1x3) = A11(1x2) B12(2x3) + A12(1x3) B22(3x3)
    # C21 (2x3) = A21(2x2) B11(2x3) + A22(2x3) B21(3x3)
    # C22 (2x3) = A21(2x2) B12(2x3) + A22(2x3) B22(3x3)
    
    # We need decompositions for:
    # 1. M(1,2,3): Rank 6 (Schoolbook is optimal for 1xn or nx1)
    # 2. M(1,3,3): Rank 9
    # 3. M(2,2,3): Rank ? 
    #    2x2 by 2x3.
    #    Split 2x3 into two 2x2 and one 2x1? No.
    #    Split inner dim 2 into 1+1.
    #    C = A1 B1 + A2 B2.
    #    A1, A2 are 2x1. B1, B2 are 1x3.
    #    Product of 2x1 and 1x3 is rank 1 matrix (outer product).
    #    So M(2,2,3) <= 2 * M(2,1,3) = 2 * 6 = 12.
    #    Schoolbook is 2*2*3 = 12. So 12 is likely optimal or close.
    # 4. M(2,3,3): Rank ?
    #    Schoolbook 18.
    #    Split 3 into 2+1.
    #    C = A1 B1 + A2 B2.
    #    A1 (2x2), B1 (2x3). Rank M(2,2,3) = 12.
    #    A2 (2x1), B2 (1x3). Rank M(2,1,3) = 6.
    #    Total 18.
    #    So schoolbook is hard to beat easily without Strassen on 2x2x2 parts.
    
    # Let's try to use Strassen for the 2x2x2 sub-blocks within these.
    
    # This is getting complicated to implement correctly from scratch in one shot.
    # Let's try a **Randomized Search with Local Optimization** approach specifically designed to find a valid decomposition of rank R < 90.
    
    # We will attempt to find a decomposition of rank R by:
    # 1. Initializing U, V, W with random Gaussian values.
    # 2. Defining the error tensor E = T - sum(U_t \otimes V_t \otimes W_t).
    # 3. Using gradient descent or a simpler "rank-1 fitting" step to minimize ||E||_F.
    # 4. Since we need EXACT integer/half-integer entries, we will round the final result and verify.
    #    If rounding fails, we retry with different seeds.
    
    # However, gradient descent on a non-convex problem is risky.
    
    # Better Strategy: **Constructive Block Decomposition with Strassen for 2x2x2**
    # I will implement the Strassen decomposition for 2x2x2 carefully and use it to build larger blocks.
    
    def get_strassen_222():
        """
        Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
        U shape (4, 7), V shape (4, 7), W shape (4, 7).
        Indices:
        U: i*2 + j
        V: j*2 + k
        W: k*2 + i
        """
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # Strassen Formulas:
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
        
        # Map indices:
        # A11: U[0,0] (i=0,j=0)
        # A12: U[0,1] (i=0,j=1)
        # A21: U[1,0] (i=1,j=0)
        # A22: U[1,1] (i=1,j=1)
        
        # B11: V[0,0] (j=0,k=0)
        # B12: V[0,1] (j=0,k=1)
        # B21: V[1,0] (j=1,k=0)
        # B22: V[1,1] (j=1,k=1)
        
        # C11: W[0,0] (k=0,i=0)
        # C12: W[1,0] (k=1,i=0)
        # C21: W[0,1] (k=0,i=1)
        # C22: W[1,1] (k=1,i=1)
        
        # t=0: M1
        # U: A11(1), A22(1) -> U[0,0]=1, U[3,0]=1
        # V: B11(1), B22(1) -> V[0,0]=1, V[3,0]=1
        # W: C11(1), C22(1) -> W[0,0]=1, W[3,0]=1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # t=1: M2
        # U: A21(1), A22(1) -> U[2,1]=1, U[3,1]=1
        # V: B11(1) -> V[0,1]=1
        # W: C21(1), C22(-1) -> W[1,1]=1, W[3,1]=-1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1
        
        # t=2: M3
        # U: A11(1) -> U[0,2]=1
        # V: B21(1), B22(-1) -> V[2,2]=1, V[3,2]=-1
        # W: C12(1) -> W[2,0]=1
        U[0,2] = 1
        V[2,2] = 1; V[3,2] = -1
        W[2,0] = 1
        
        # t=3: M4
        # U: A22(1) -> U[3,3]=1
        # V: B12(1), B11(-1) -> V[1,3]=1, V[0,3]=-1
        # W: C11(1), C21(1) -> W[0,0]=1, W[1,1]=1
        U[3,3] = 1
        V[1,3] = 1; V[0,3] = -1
        W[0,0] += 1; W[1,1] += 1
        
        # t=4: M5
        # U: A11(1), A12(1) -> U[0,4]=1, U[1,4]=1
        # V: B22(1) -> V[3,4]=1
        # W: C11(-1), C12(1) -> W[0,0]=-1, W[2,0]=1
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,0] -= 1; W[2,0] += 1
        
        # t=5: M6
        # U: A21(1), A11(-1) -> U[2,5]=1, U[0,5]=-1
        # V: B11(1), B12(1) -> V[0,5]=1, V[1,5]=1
        # W: C22(1) -> W[3,1]=1
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,1] += 1
        
        # t=6: M7
        # U: A12(1), A22(-1) -> U[1,6]=1, U[3,6]=-1
        # V: B21(1), B22(1) -> V[2,6]=1, V[3,6]=1
        # W: C11(1) -> W[0,0]=1
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,0] += 1
        
        return U, V, W

    def get_schoolbook(n_b, m_b, p_b):
        """Returns U, V, W for schoolbook multiplication of n_b x m_b by m_b x p_b."""
        r = n_b * m_b * p_b
        U = np.zeros((n_b * m_b, r), dtype=int)
        V = np.zeros((m_b * p_b, r), dtype=int)
        W = np.zeros((p_b * n_b, r), dtype=int)
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j, c] = 1
                    V[j * p_b + k, c] = 1
                    W[k * n_b + i, c] = 1
                    c += 1
        return U, V, W

    def combine_blocks(blocks):
        """
        Combines a list of (U_b, V_b, W_b, i_off, j_off, k_off) into global U, V, W.
        """
        total_rank = sum(U_b.shape[1] for U_b, _, _, _, _, _ in blocks)
        U_global = np.zeros((n * m, total_rank), dtype=int)
        V_global = np.zeros((m * p, total_rank), dtype=int)
        W_global = np.zeros((p * n, total_rank), dtype=int)
        
        col_offset = 0
        for U_b, V_b, W_b, i_off, j_off, k_off in blocks:
            n_b = U_b.shape[0] // (j_off + 1) # This logic is fragile. Let's pass dimensions.
            # Better: pass n_b, m_b, p_b explicitly.
            pass
        # I need to refactor combine_blocks to accept dimensions.
        
        # Let's just build the specific blocks manually to avoid generic errors.
        pass

    # Given the complexity and time, I will construct the specific blocks for:
    # n=3, m=5, p=6
    # Split n: 1, 2
    # Split m: 2, 3
    # Split p: 3, 3
    
    # Block 1: n=1, m=2, p=3 (Schoolbook, Rank 6)
    U1, V1, W1 = get_schoolbook(1, 2, 3)
    # Block 2: n=1, m=3, p=3 (Schoolbook, Rank 9)
    U2, V2, W2 = get_schoolbook(1, 3, 3)
    # Block 3: n=2, m=2, p=3 (Schoolbook, Rank 12)
    U3, V3, W3 = get_schoolbook(2, 2, 3)
    # Block 4: n=2, m=3, p=3 (Schoolbook, Rank 18)
    U4, V4, W4 = get_schoolbook(2, 3, 3)
    
    # Total Rank = 6 + 9 + 12 + 18 = 45?
    # Wait, the blocks are:
    # C11 (1x3) = A11(1x2)B11(2x3) + A12(1x3)B21(3x3)
    #   Term 1: M(1,2,3) Rank 6
    #   Term 2: M(1,3,3) Rank 9
    # C12 (1x3) = A11(1x2)B12(2x3) + A12(1x3)B22(3x3)
    #   Term 3: M(1,2,3) Rank 6
    #   Term 4: M(1,3,3) Rank 9
    # C21 (2x3) = A21(2x2)B11(2x3) + A22(2x3)B21(3x3)
    #   Term 5: M(2,2,3) Rank 12
    #   Term 6: M(2,3,3) Rank 18
    # C22 (2x3) = A21(2x2)B12(2x3) + A22(2x3)B22(3x3)
    #   Term 7: M(2,2,3) Rank 12
    #   Term 8: M(2,3,3) Rank 18
    
    # Total Rank = 6+9+6+9+12+18+12+18 = 90.
    # This is just schoolbook again.
    
    # To improve, I need to use Strassen for the 2x2x2 parts inside these blocks.
    # Let's try to use Strassen for M(2,2,3) and M(2,3,3).
    
    # M(2,2,3):
    # Split p=3 into 1+2.
    # C = A B1 + A B2
    # B1 is 2x1, B2 is 2x2.
    # A B1: 2x2 by 2x1. Rank M(2,2,1) = 4 (Schoolbook).
    # A B2: 2x2 by 2x2. Rank M(2,2,2) = 7 (Strassen).
    # Total M(2,2,3) <= 4 + 7 = 11.
    
    # M(2,3,3):
    # Split m=3 into 1+2.
    # C = A1 B1 + A2 B2
    # A1 is 2x1, B1 is 1x3. Rank M(2,1,3) = 6.
    # A2 is 2x2, B2 is 2x3.
    # Split B2 p=3 into 1+2.
    # A2 B2 = A2 B21 + A2 B22.
    # A2 B21: 2x2 by 2x1. Rank 4.
    # A2 B22: 2x2 by 2x2. Rank 7.
    # So M(2,2,3) <= 11.
    # Total M(2,3,3) <= 6 + 11 = 17.
    
    # Let's recalculate total rank with these improvements:
    # Term 1: M(1,2,3) = 6
    # Term 2: M(1,3,3) = 9
    # Term 3: M(1,2,3) = 6
    # Term 4: M(1,3,3) = 9
    # Term 5: M(2,2,3) <= 11
    # Term 6: M(2,3,3) <= 17
    # Term 7: M(2,2,3) <= 11
    # Term 8: M(2,3,3) <= 17
    
    # Total = 6+9+6+9+11+17+11+17 = 86.
    # Improvement from 90 to 86.
    
    # I will implement these specific decompositions.
    
    def get_m223_strassen():
        """
        M(2,2,3) using Strassen for 2x2x2 part.
        A is 2x2, B is 2x3.
        Split B into B1 (2x1) and B2 (2x2).
        C1 = A B1 (Rank 4, Schoolbook)
        C2 = A B2 (Rank 7, Strassen)
        Total Rank 11.
        """
        U_list = []
        V_list = []
        W_list = []
        
        # Part 1: A B1 (2x2 by 2x1)
        # Schoolbook rank 4.
        U1, V1, W1 = get_schoolbook(2, 2, 1)
        U_list.append(U1)
        V_list.append(V1)
        W_list.append(W1)
        
        # Part 2: A B2 (2x2 by 2x2)
        # Strassen rank 7.
        U2, V2, W2 = get_strassen_222()
        U_list.append(U2)
        V_list.append(V2)
        W_list.append(W2)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        return U, V, W

    def get_m233_strassen():
        """
        M(2,3,3) using Strassen for 2x2x2 part.
        A is 2x3, B is 3x3.
        Split A into A1 (2x1) and A2 (2x2).
        C1 = A1 B1 (2x1 by 1x3). Rank 6.
        C2 = A2 B2 (2x2 by 2x3). Rank 11 (from M(2,2,3)).
        Total Rank 17.
        """
        U_list = []
        V_list = []
        W_list = []
        
        # Part 1: A1 B1 (2x1 by 1x3)
        # Schoolbook rank 6.
        U1, V1, W1 = get_schoolbook(2, 1, 3)
        U_list.append(U1)
        V_list.append(V1)
        W_list.append(W1)
        
        # Part 2: A2 B2 (2x2 by 2x3)
        # Use M(2,2,3) decomposition.
        U2, V2, W2 = get_m223_strassen()
        U_list.append(U2)
        V_list.append(V2)
        W_list.append(W2)
        
        U = np.hstack(U_list)
        V = np.hstack(V_list)
        W = np.hstack(W_list)
        return U, V, W

    # Now build the global blocks.
    
    # Block 1: n=1, m=2, p=3 (Term 1: A11 B11)
    # A11 is 1x2, B11 is 2x3.
    U_b1, V_b1, W_b1 = get_schoolbook(1, 2, 3)
    
    # Block 2: n=1, m=3, p=3 (Term 2: A12 B21)
    # A12 is 1x3, B21 is 3x3.
    U_b2, V_b2, W_b2 = get_schoolbook(1, 3, 3)
    
    # Block 3: n=1, m=2, p=3 (Term 3: A11 B12)
    # A11 is 1x2, B12 is 2x3.
    U_b3, V_b3, W_b3 = get_schoolbook(1, 2, 3)
    
    # Block 4: n=1, m=3, p=3 (Term 4: A12 B22)
    # A12 is 1x3, B22 is 3x3.
    U_b4, V_b4, W_b4 = get_schoolbook(1, 3, 3)
    
    # Block 5: n=2, m=2, p=3 (Term 5: A21 B11)
    # A21 is 2x2, B11 is 2x3.
    U_b5, V_b5, W_b5 = get_m223_strassen()
    
    # Block 6: n=2, m=3, p=3 (Term 6: A22 B21)
    # A22 is 2x3, B21 is 3x3.
    U_b6, V_b6, W_b6 = get_m233_strassen()
    
    # Block 7: n=2, m=2, p=3 (Term 7: A21 B12)
    # A21 is 2x2, B12 is 2x3.
    U_b7, V_b7, W_b7 = get_m223_strassen()
    
    # Block 8: n=2, m=3, p=3 (Term 8: A22 B22)
    # A22 is 2x3, B22 is 3x3.
    U_b8, V_b8, W_b8 = get_m233_strassen()
    
    # Combine into global U, V, W
    # Offsets:
    # i: 0 (for n=1), 1 (for n=2)
    # j: 0 (for m=2), 2 (for m=3) -- Wait, m split is 2+3. So j_off 0 and 2.
    # k: 0 (for p=3), 3 (for p=3) -- Wait, p split is 3+3. So k_off 0 and 3.
    
    # Let's map the blocks to offsets:
    # Block 1: i=0, j=0, k=0
    # Block 2: i=0, j=2, k=0
    # Block 3: i=0, j=0, k=3
    # Block 4: i=0, j=2, k=3
    # Block 5: i=1, j=0, k=0
    # Block 6: i=1, j=2, k=0
    # Block 7: i=1, j=0, k=3
    # Block 8: i=1, j=2, k=3
    
    blocks = [
        (U_b1, V_b1, W_b1, 1, 2, 3, 0, 0, 0),
        (U_b2, V_b2, W_b2, 1, 3, 3, 0, 2, 0),
        (U_b3, V_b3, W_b3, 1, 2, 3, 0, 0, 3),
        (U_b4, V_b4, W_b4, 1, 3, 3, 0, 2, 3),
        (U_b5, V_b5, W_b5, 2, 2, 3, 1, 0, 0),
        (U_b6, V_b6, W_b6, 2, 3, 3, 1, 2, 0),
        (U_b7, V_b7, W_b7, 2, 2, 3, 1, 0, 3),
        (U_b8, V_b8, W_b8, 2, 3, 3, 1, 2, 3)
    ]
    
    total_rank = sum(U_b.shape[1] for U_b, _, _, _, _, _, _, _, _ in blocks)
    U_global = np.zeros((n * m, total_rank), dtype=int)
    V_global = np.zeros((m * p, total_rank), dtype=int)
    W_global = np.zeros((p * n, total_rank), dtype=int)
    
    col_offset = 0
    for U_b, V_b, W_b, n_b, m_b, p_b, i_off, j_off, k_off in blocks:
        # Embed U_b
        for i in range(n_b):
            for j in range(m_b):
                u_idx = i * m + j
                u_b_idx = i * m_b + j
                U_global[u_idx, col_offset:col_offset+U_b.shape[1]] = U_b[u_b_idx, :]
        
        # Embed V_b
        for j in range(m_b):
            for k in range(p_b):
                v_idx = j * p + k
                v_b_idx = j * p_b + k
                V_global[v_idx, col_offset:col_offset+V_b.shape[1]] = V_b[v_b_idx, :]
        
        # Embed W_b
        for k in range(p_b):
            for i in range(n_b):
                w_idx = k * n + i
                w_b_idx = k * n_b + i
                W_global[w_idx, col_offset:col_offset+W_b.shape[1]] = W_b[w_b_idx, :]
        
        col_offset += U_b.shape[1]
    
    # Convert to lists of lists for return
    U_list = U_global.tolist()
    V_list = V_global.tolist()
    W_list = W_global.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
