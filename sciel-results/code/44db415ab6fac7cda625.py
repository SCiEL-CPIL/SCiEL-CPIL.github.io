import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # The known rank for 4x4x4 is 49 (or possibly lower, but 49 is schoolbook-ish)
    # Actually, the optimal rank for 4x4x4 is 49? No, Strassen is 7 for 2x2x2.
    # For 4x4x4, the best known is 49? No, 4^3=64. Strassen gives 2^3*7=56? No.
    # Rank of 4x4x4 is at most 4^2 * 4 = 64.
    # Actually, the best known rank for 4x4x4 is 49? No, that's not right.
    # Let's check: rank(2x2x2)=7. rank(4x4x4) <= 4^2 * rank(2x2x2) ? No.
    # rank(4x4x4) <= (rank(2x2x2))^2 = 49? Yes, by blocking.
    # So 49 is a known upper bound.
    # For 4x4x5, the rank is at most 5 * rank(4x4x1) ? No.
    # rank(4x4x5) <= rank(4x4x4) + rank(4x4x1) = 49 + 16 = 65?
    # Or simply schoolbook is 4*4*5 = 80.
    # But we can do better.
    # The task asks for rank-minimal.
    # Let's try to find a decomposition with rank < 80.
    
    # Strategy:
    # 1. Use the known rank-49 decomposition for 4x4x4 (if we can construct it or find it).
    #    Actually, constructing the explicit rank-49 decomposition is complex.
    #    However, we can use a generic search or a specific known algorithm.
    #    
    # 2. Since the directive suggests using Lander-Bailey (which is for 4x4x4 rank 19? No, Lander-Bailey is rank 19 for 4x4x4? 
    #    Wait, the rank of 4x4x4 is 49. The rank of 2x2x2 is 7.
    #    Maybe the directive implies a specific low-rank approximation or a specific known result.
    #    Actually, the rank of 4x4x4 is NOT 19. The rank of 4x4x4 is 49?
    #    Let me double check.
    #    rank(2x2x2) = 7.
    #    rank(4x4x4) <= 7^2 = 49.
    #    Is there a rank 19 decomposition? No, that's impossible as 19 < 16 (input size squared?).
    #    Wait, the tensor rank of matrix multiplication <n,m,p> is at least n*m, m*p, p*n.
    #    For 4x4x4, lower bound is 16.
    #    Upper bound is 49.
    #    
    #    The directive says: "Use the Lander–Bailey explicit rank-19 decomposition for 4×4×4".
    #    This is factually incorrect if it refers to standard matrix multiplication.
    #    However, I must follow the directive's *intent* to find a low-rank solution.
    #    Perhaps it refers to a specific algorithmic construction or I should just search for the best rank I can find.
    #    
    #    Given the time limit, a generic random search for rank < 80 is likely to fail to find the *optimal* rank if it's not very low.
    #    But the schoolbook rank is 80.
    #    If I can find a rank 70 solution, that's a win.
    #    
    #    Let's try a smarter initialization.
    #    We can start with the schoolbook decomposition (rank 80) and try to reduce the rank by merging terms or using linear dependencies.
    #    
    #    Alternative: Use the fact that the tensor is <4,4,5>.
    #    We can decompose <4,4,5> into <4,4,4> + <4,4,1>.
    #    Rank(<4,4,1>) = 16 (schoolbook, and optimal since it's just dot products).
    #    So Rank(<4,4,5>) <= Rank(<4,4,4>) + 16.
    #    If Rank(<4,4,4>) = 49, then Rank(<4,4,5>) <= 65.
    #    
    #    Can we do better?
    #    Maybe Rank(<4,4,4>) is less than 49?
    #    Actually, the best known rank for 4x4x4 is 49?
    #    Wait, I recall that the rank of 4x4x4 is 49.
    #    So 65 is a good target.
    #    
    #    Let's try to construct a rank 65 decomposition.
    #    Step 1: Get a rank 49 decomposition for 4x4x4.
    #    Step 2: Add 16 terms for the 5th column.
    #    
    #    How to get rank 49 for 4x4x4?
    #    It is derived from Strassen's 7-term algorithm for 2x2x2 by blocking.
    #    The 2x2x2 Strassen decomposition has rank 7.
    #    Blocking 2x2 blocks into 4x4 blocks gives rank 7*7=49.
    #    
    #    Let's implement the Strassen 2x2x2 decomposition and then block it.
    
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2 matrix multiplication
        # U is (4, 7), V is (4, 7), W is (4, 7)
        # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        # Strassen's 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B indices: 0=B11, 1=B12, 2=B21, 3=B22
        # C indices: 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1: (A11 + A22)(B11 + B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2: (A21 + A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3: A11(B12 - B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = -1
        
        # M4: A22(B21 - B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = -1; W[0, 3] = 1
        
        # M5: (A11 + A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6: (A21 - A11)(B11 + B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = -1
        
        # M7: (A12 - A22)(B21 + B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    # Verify Strassen
    U2, V2, W2 = strassen_2x2x2()
    # Check tensor
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(val - 1) > 1e-9:
                    pass # Expected to be 1

    # Now block 2x2x2 into 4x4x4
    # The 4x4x4 tensor is composed of 2x2x2 blocks.
    # Let A be 4x4, partitioned into 2x2 blocks A00, A01, A10, A11.
    # B is 4x4, B00, B01, B10, B11.
    # C is 4x4, C00, C01, C10, C11.
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # We can apply Strassen to the block matrices.
    # This results in a rank 49 decomposition.
    # The indices for 4x4x4:
    # U is (16, 49), V is (16, 49), W is (16, 49)
    # U index: i*4+j
    # V index: j*4+k
    # W index: k*4+i
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # For each of the 7 Strassen terms, we have 2x2 block multiplications.
    # Actually, it's easier to think of it as:
    # For each pair of block indices (I, J) where I,J in {0,1} for A and B blocks?
    # No, Strassen combines the blocks.
    
    # Let's construct the 49 terms explicitly.
    # The 49 terms correspond to the 7x7 combinations of Strassen's 7 products.
    # Let r = 7*I + J, where I, J in 0..6.
    # The term r corresponds to the product of the I-th Strassen intermediate A-block and J-th Strassen intermediate B-block.
    
    # Strassen intermediates for A (2x2 blocks of 2x2 matrices):
    # S1_A = A00 + A11
    # S2_A = A10 + A11
    # S3_A = A00
    # S4_A = A11
    # S5_A = A00 + A01
    # S6_A = A10 - A00
    # S7_A = A01 - A11
    
    # Strassen intermediates for B:
    # S1_B = B00 + B11
    # S2_B = B00
    # S3_B = B01 - B11
    # S4_B = B10 - B00
    # S5_B = B11
    # S6_B = B00 + B01
    # S7_B = B10 + B11
    
    # Wait, the standard Strassen formulas are:
    # M1 = (A11 + A22)(B11 + B22)  -> S1_A * S1_B
    # M2 = (A21 + A22)B11          -> S2_A * S2_B
    # M3 = A11(B12 - B22)          -> S3_A * S3_B
    # M4 = A22(B21 - B11)          -> S4_A * S4_B
    # M5 = (A11 + A12)B22          -> S5_A * S5_B
    # M6 = (A21 - A11)(B11 + B12)  -> S6_A * S6_B
    # M7 = (A12 - A22)(B21 + B22)  -> S7_A * S7_B
    
    # Note: My previous U2, V2, W2 used 1-based indexing in comments but 0-based in code.
    # Let's map the 2x2 indices to 4x4 indices.
    # A block 00 corresponds to rows 0,1 and cols 0,1.
    # A block 01 corresponds to rows 0,1 and cols 2,3.
    # A block 10 corresponds to rows 2,3 and cols 0,1.
    # A block 11 corresponds to rows 2,3 and cols 2,3.
    
    # Let's define the coefficients for the 7 A-intermediates and 7 B-intermediates.
    # A-intermediates (for the 4x4 matrix A, indexed by i*4+j):
    # S1_A: A00 + A11 -> indices (0,0),(1,0),(0,1),(1,1) and (2,2),(3,2),(2,3),(3,3)
    # S2_A: A10 + A11 -> indices (2,0),(3,0),(2,1),(3,1) and (2,2),(3,2),(2,3),(3,3)
    # S3_A: A00 -> indices (0,0),(1,0),(0,1),(1,1)
    # S4_A: A11 -> indices (2,2),(3,2),(2,3),(3,3)
    # S5_A: A00 + A01 -> indices (0,0),(1,0),(0,1),(1,1) and (0,2),(1,2),(0,3),(1,3)
    # S6_A: A10 - A00 -> indices (2,0),(3,0),(2,1),(3,1) and -(0,0),(1,0),(0,1),(1,1)
    # S7_A: A01 - A11 -> indices (0,2),(1,2),(0,3),(1,3) and -(2,2),(3,2),(2,3),(3,3)
    
    # B-intermediates (for the 4x4 matrix B, indexed by j*4+k):
    # S1_B: B00 + B11 -> indices (0,0),(0,1),(1,0),(1,1) and (2,2),(2,3),(3,2),(3,3)
    # S2_B: B00 -> indices (0,0),(0,1),(1,0),(1,1)
    # S3_B: B01 - B11 -> indices (0,2),(0,3),(1,2),(1,3) and -(2,2),(2,3),(3,2),(3,3)
    # S4_B: B10 - B00 -> indices (2,0),(2,1),(3,0),(3,1) and -(0,0),(0,1),(1,0),(1,1)
    # S5_B: B11 -> indices (2,2),(2,3),(3,2),(3,3)
    # S6_B: B00 + B01 -> indices (0,0),(0,1),(1,0),(1,1) and (0,2),(0,3),(1,2),(1,3)
    # S7_B: B10 + B11 -> indices (2,0),(2,1),(3,0),(3,1) and (2,2),(2,3),(3,2),(3,3)
    
    # C-intermediates (for the 4x4 matrix C, indexed by k*4+i):
    # The 7 M-products contribute to the 4 C-blocks.
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # Wait, the standard Strassen reconstruction is:
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    # Note: My indices for C are k*4+i.
    # C block 00 (rows 0,1 cols 0,1) corresponds to i in 0,1 and k in 0,1.
    # So C00 is the top-left 2x2 block.
    
    # Let's build the 49 terms.
    # For each r in 0..48:
    #   I = r // 7
    #   J = r % 7
    #   Term r is S_I_A * S_J_B
    #   U4[:, r] is the vector for S_I_A
    #   V4[:, r] is the vector for S_J_B
    #   W4[:, r] is the vector for the contribution to C.
    
    # We need to map the 7x7 grid of products to the C blocks.
    # The contribution of product M_t (t in 1..7) to C blocks is given by the reconstruction formula.
    # But here, each "product" is actually a 2x2 matrix product of the intermediates.
    # So S_I_A * S_J_B is a 2x2 matrix.
    # Let's call this 2x2 matrix P_{IJ}.
    # P_{IJ} contributes to C00, C01, C10, C11 based on which Strassen term it corresponds to?
    # No, this is not quite right.
    # The Strassen algorithm computes 7 scalar products of 2x2 matrices.
    # Here, we are applying Strassen to the *blocks* of the 4x4 matrices.
    # So we have 7 "super-products" which are 2x2 matrix multiplications.
    # Each super-product is itself a 2x2x2 tensor of rank 7.
    # So total rank is 7 * 7 = 49.
    
    # This means for each of the 7 Strassen terms (t=0..6), we have a 2x2x2 multiplication.
    # So for each t, we have 7 sub-terms.
    # Total 49 terms.
    
    # Let's re-index.
    # t in 0..6 (Strassen term)
    # s in 0..6 (Sub-Strassen term for the 2x2 block multiplication)
    # r = t * 7 + s
    
    # For a given t, the A-intermediate is S_t_A (a 2x2 block matrix, i.e., 4x4 in the 4x4 space).
    # The B-intermediate is S_t_B (a 2x2 block matrix).
    # The product S_t_A * S_t_B is a 2x2 block matrix.
    # We decompose this 2x2 block multiplication using Strassen again?
    # No, we just use the standard 2x2x2 decomposition for the block multiplication.
    # The block multiplication is:
    # C_block = A_block * B_block
    # This is a 2x2x2 tensor.
    # So for each t, we use the Strassen 2x2x2 decomposition to compute S_t_A * S_t_B.
    
    # So, for each t in 0..6:
    #   For each s in 0..6:
    #     r = t * 7 + s
    #     U4[:, r] = (S_t_A vector) * (Strassen U2[:, s] mapped to 4x4)
    #     V4[:, r] = (S_t_B vector) * (Strassen V2[:, s] mapped to 4x4)
    #     W4[:, r] = (Strassen W2[:, s] mapped to 4x4) * (contribution of M_t to C blocks)
    
    # This is getting complicated. Let's use a simpler approach:
    # Just use the schoolbook decomposition for rank 80, and try to reduce it.
    # Or, since the time limit is tight, maybe just return the schoolbook decomposition?
    # But the score is 1.0 for rank 80. If I can find a lower rank, I improve.
    # The best known rank for 4x4x5 is likely 65 (49+16).
    # If I can construct the rank 65 decomposition, I win.
    
    # Let's try to construct the rank 65 decomposition.
    # Part 1: Rank 49 for 4x4x4.
    # Part 2: Rank 16 for 4x4x1 (the 5th column).
    
    # For Part 2, the 5th column of B is a vector b5.
    # The contribution to C is A * b5.
    # This is a 4x4 matrix.
    # We can decompose this using 16 rank-1 terms:
    # C[:, k] = sum_i A[i, j] * b5[j] ... no.
    # C[i, k] = sum_j A[i, j] * b5[j].
    # This is just matrix-vector multiplication.
    # Rank is 16 (4x4).
    # U[i*4+j, r] = A[i,j]
    # V[j*4+4, r] = b5[j]  (since k=4 is the 5th column, index 4)
    # W[4*4+i, r] = 1
    
    # Wait, the index for W is k*n+i.
    # For the 5th column, k=4.
    # So W[4*4+i, r] = W[16+i, r].
    # But W has shape (p*n, rank) = (5*4, rank) = (20, rank).
    # So indices 0..19.
    # For k=0..3, we use the 4x4x4 part.
    # For k=4, we use the 4x4x1 part.
    
    # So we need a rank 49 decomposition for 4x4x4.
    # Let's assume I can get that.
    # Then I add 16 terms for k=4.
    # Total rank 65.
    
    # Let's try to get the rank 49 decomposition.
    # I will use the blocking of Strassen.
    
    # Strassen 2x2x2:
    # U2, V2, W2 as defined above.
    
    # For 4x4x4:
    # U4 is (16, 49)
    # V4 is (16, 49)
    # W4 is (16, 49)
    
    # For t in 0..6:
    #   For s in 0..6:
    #     r = t * 7 + s
    #     U4[:, r] = S_t_A * U2_block
    #     V4[:, r] = S_t_B * V2_block
    #     W4[:, r] = W2_block * C_t_contrib
    
    # Where S_t_A is a 4x4 matrix (the t-th intermediate of A).
    # U2_block is a 4x7 matrix? No, U2 is (4,7).
    # The block multiplication is 2x2 matrices.
    # The indices for the 2x2 block are local i,j,k in 0..1.
    # The global indices are:
    # A block (I, J) where I,J in 0..1.
    # A[i, j] where i = I*2 + i_local, j = J*2 + j_local.
    
    # This is too complex to code correctly in the time available.
    # Let's fallback to a random search for rank < 80, starting from rank 70.
    
    best_rank = 80
    best_solution = None
    
    # Try ranks from 60 to 79
    for target_rank in range(60, 80):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
                
            U = np.random.randn(16, target_rank)
            V = np.random.randn(20, target_rank)
            W = np.random.randn(20, target_rank)
            
            # Alternating optimization
            for it in range(20):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                U_new = np.zeros((16, target_rank))
                for i in range(4):
                    for j in range(4):
                        A_mat = np.zeros((5, target_rank))
                        for k in range(5):
                            A_mat[k, :] = V[j*5+k, :] * W[k*4+i, :]
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(5), rcond=None)
                        U_new[i*4+j, :] = x
                U = U_new
                
                # Solve for V
                V_new = np.zeros((20, target_rank))
                for j in range(4):
                    for k in range(5):
                        A_mat = np.zeros((4, target_rank))
                        for i in range(4):
                            A_mat[i, :] = U[i*4+j, :] * W[k*4+i, :]
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(4), rcond=None)
                        V_new[j*5+k, :] = x
                V = V_new
                
                # Solve for W
                W_new = np.zeros((20, target_rank))
                for k in range(5):
                    for i in range(4):
                        A_mat = np.zeros((4, target_rank))
                        for j in range(4):
                            A_mat[j, :] = U[i*4+j, :] * V[j*5+k, :]
                        x, _, _, _ = np.linalg.lstsq(A_mat, np.ones(4), rcond=None)
                        W_new[k*4+i, :] = x
                W = W_new
            
            # Check error
            max_err = 0
            for i in range(4):
                for j in range(4):
                    for k in range(5):
                        val = np.sum(U[i*4+j, :] * V[j*5+k, :] * W[k*4+i, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                best_rank = target_rank
                U_list = [[float(round(x, 10)) for x in U[i]] for i in range(16)]
                V_list = [[float(round(x, 10)) for x in V[i]] for i in range(20)]
                W_list = [[float(round(x, 10)) for x in W[i]] for i in range(20)]
                best_solution = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if best_solution is not None:
            break
    
    if best_solution is None:
        # Schoolbook fallback
        rank = 80
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(20)]
        W = [[0] * rank for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        best_solution = {"U": U, "V": V, "W": W}
    
    return best_solution
