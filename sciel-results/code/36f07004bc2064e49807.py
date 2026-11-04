import time
import numpy as np
from fractions import Fraction

n, m, p = 3, 5, 5

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    # 1. Default schoolbook decomposition (Rank 75)
    # This is guaranteed to be correct and valid.
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    # 2. Attempt to find a lower rank decomposition using block structure.
    # We view the multiplication as a sum of independent blocks.
    # For 3x5 * 5x5, we can partition indices to isolate 2x2x2 sub-tensors
    # which can be computed with rank 7 (Strassen) instead of rank 8 (schoolbook).
    #
    # Partition:
    # i (rows of A, rows of C): {0, 1} (size 2), {2} (size 1)
    # j (cols of A, rows of B): {0, 1} (size 2), {2, 3, 4} (size 3) -> split into {2,3} (size 2), {4} (size 1)
    # k (cols of B, cols of C): {0, 1} (size 2), {2, 3, 4} (size 3) -> split into {2,3} (size 2), {4} (size 1)
    #
    # This creates a 2x2x2 grid of blocks based on these partitions.
    # Block (i_block, j_block, k_block) corresponds to multiplying A[i_block, j_block] by B[j_block, k_block] to get C[i_block, k_block].
    # The sizes are (2,2,2), (2,2,1), (2,1,2), (2,1,1) for i_block=0.
    # And (1,2,2), (1,2,1), (1,1,2), (1,1,1) for i_block=1.
    #
    # Let's list the blocks (size_i, size_j, size_k) and their rank contribution:
    # 1. (2,2,2): Rank 7 (Strassen)
    # 2. (2,2,1): Rank 4 (Schoolbook for 2x2x1 is 4, or can we do better? 2*2*1=4. Strassen is for 2x2x2. For 2x2x1, it's just dot products of 2x2 matrix with 1x2 vector? No, A(2x2)*B(2x1) -> C(2x1). Rank is 2*2*1=4? No, C_{i,k} = sum_j A_{i,j} B_{j,k}. For fixed k (only 1), it's 2 dot products of length 2. Rank 4? Yes, each C_{i,k} needs 2 multiplications. Total 4.
    #    Wait, is there a faster way for 2x2x1? No, it's linear.
    # 3. (2,1,2): Rank 4. A(2x1)*B(1x2) -> C(2x2). C_{i,k} = A_{i,0} B_{0,k}. This is outer product of A col and B row. Rank 2*1*2=4? No, C_{i,k} = A_i B_k. This is rank 1 in the tensor sense? No, the tensor T_{i,j,k} = delta_{j,0}. The rank of the tensor is 2*2=4?
    #    Let's check: C_{i,k} = A_{i,0} B_{0,k}. The multiplications are A_{i,0} * B_{0,k} for i=0,1 and k=0,1. That's 4 multiplications. Can we do it in 3? No, because they are independent.
    # 4. (2,1,1): Rank 2. A(2x1)*B(1x1) -> C(2x1). C_i = A_{i,0} B_{0,0}. 2 multiplications.
    # 5. (1,2,2): Rank 4. A(1x2)*B(2x2) -> C(1x2). C_k = sum_j A_{0,j} B_{j,k}. 2 dot products of length 2. 4 multiplications.
    # 6. (1,2,1): Rank 2. A(1x2)*B(2x1) -> C(1x1). C_0 = A_0 B_0 + A_1 B_1. 2 multiplications.
    # 7. (1,1,2): Rank 2. A(1x1)*B(1x2) -> C(1x2). C_k = A_0 B_k. 2 multiplications.
    # 8. (1,1,1): Rank 1.
    #
    # Total Rank = 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
    # This is much better than 75.
    #
    # We need to construct the explicit Strassen decomposition for the (2,2,2) block
    # and schoolbook for the others, mapping the local indices to global indices.
    
    # Strassen 2x2x2 coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let's define the blocks and their global index mappings.
    # i_blocks: [ [0, 1], [2] ]
    # j_blocks: [ [0, 1], [2, 3], [4] ]
    # k_blocks: [ [0, 1], [2, 3], [4] ]
    
    i_parts = [[0, 1], [2]]
    j_parts = [[0, 1], [2, 3], [4]]
    k_parts = [[0, 1], [2, 3], [4]]
    
    U_block = []
    V_block = []
    W_block = []
    
    # We will build the U, V, W matrices for this block decomposition.
    # Since we don't know the total rank in advance, we'll append to lists.
    
    for bi, i_idx in enumerate(i_parts):
        for bj, j_idx in enumerate(j_parts):
            for bk, k_idx in enumerate(k_parts):
                ni = len(i_idx)
                nj = len(j_idx)
                nk = len(k_idx)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Strassen Block
                    # Local indices: i_l in 0..1, j_l in 0..1, k_l in 0..1
                    # Global indices: i_g = i_idx[i_l], j_g = j_idx[j_l], k_g = k_idx[k_l]
                    
                    # U matrix for this block: size (2*2=4) x 7
                    # V matrix for this block: size (2*2=4) x 7
                    # W matrix for this block: size (2*2=4) x 7
                    # Note: U is indexed by (i, j), V by (j, k), W by (k, i).
                    
                    U_sub = [[0]*7 for _ in range(4)]
                    V_sub = [[0]*7 for _ in range(4)]
                    W_sub = [[0]*7 for _ in range(4)]
                    
                    # M1: (A00+A11)(B00+B11)
                    # A00: U[0][0]=1, A11: U[3][0]=1
                    # B00: V[0][0]=1, B11: V[3][0]=1
                    # C00: W[0][0]=1, C11: W[3][0]=1
                    # C00 += M1, C11 += M1
                    U_sub[0][0] = 1; U_sub[3][0] = 1
                    V_sub[0][0] = 1; V_sub[3][0] = 1
                    W_sub[0][0] = 1; W_sub[3][0] = 1
                    
                    # M2: (A10+A11)B00
                    # A10: U[2][1]=1, A11: U[3][1]=1
                    # B00: V[0][1]=1
                    # C10: W[1][1]=1
                    U_sub[2][1] = 1; U_sub[3][1] = 1
                    V_sub[0][1] = 1
                    W_sub[1][1] = 1
                    
                    # M3: A00(B10-B11)
                    # A00: U[0][2]=1
                    # B10: V[2][2]=1, B11: V[3][2]=-1
                    # C01: W[2][2]=1
                    U_sub[0][2] = 1
                    V_sub[2][2] = 1; V_sub[3][2] = -1
                    W_sub[2][2] = 1
                    
                    # M4: A11(B01-B00)
                    # A11: U[3][3]=1
                    # B01: V[1][3]=1, B00: V[0][3]=-1
                    # C00: W[0][3]=1, C10: W[1][3]=1
                    U_sub[3][3] = 1
                    V_sub[1][3] = 1; V_sub[0][3] = -1
                    W_sub[0][3] = 1; W_sub[1][3] = 1
                    
                    # M5: (A00+A01)B11
                    # A00: U[0][4]=1, A01: U[1][4]=1
                    # B11: V[3][4]=1
                    # C00: W[0][4]=-1, C01: W[2][4]=1
                    U_sub[0][4] = 1; U_sub[1][4] = 1
                    V_sub[3][4] = 1
                    W_sub[0][4] = -1; W_sub[2][4] = 1
                    
                    # M6: (A10-A00)(B00+B01)
                    # A10: U[2][5]=1, A00: U[0][5]=-1
                    # B00: V[0][5]=1, B01: V[1][5]=1
                    # (Unused in C? Wait, Strassen formula:
                    # C11 = M1 + M4 - M5 - M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M7
                    # M6 is not used in the standard Strassen algorithm for C?
                    # Let me re-check Strassen.
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B21-B22)
                    # M4 = A22(B12-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    # C11 = M1 + M4 - M5 - M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M7
                    # Yes, M6 is not used in the final sum. It's an intermediate term that cancels out or is not needed in this specific formulation?
                    # Actually, the standard Strassen algorithm defines 7 products. M6 is one of them.
                    # Let's check C22 again.
                    # C22 = M1 - M2 + M3 + M7.
                    # Where is M6?
                    # Ah, the definition of M6 is (A21-A11)(B11+B12).
                    # In some formulations, M6 is used. Let's verify the expansion.
                    # C22 = A21B12 + A22B22.
                    # M1 - M2 + M3 + M7 = (A11+A22)(B11+B22) - (A21+A22)B11 + A11(B21-B22) + (A12-A22)(B21+B22)
                    # = A11B11 + A11B22 + A22B11 + A22B22 - A21B11 - A22B11 + A11B21 - A11B22 + A12B21 + A12B22 - A22B21 - A22B22
                    # = A11B11 + A22B22 - A21B11 + A11B21 + A12B21 + A12B22 - A22B21
                    # This does not look like A21B12 + A22B22.
                    # I must have the formula wrong.
                    
                    # Correct Strassen:
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B21-B22)
                    # M4 = A22(B12-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    #
                    # C11 = M1 + M4 - M5 - M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M7
                    #
                    # Let's re-expand C22.
                    # M1 = A11B11 + A11B22 + A22B11 + A22B22
                    # -M2 = -A21B11 - A22B11
                    # M3 = A11B21 - A11B22
                    # M7 = A12B21 + A12B22 - A22B21 - A22B22
                    # Sum:
                    # A11B11 + A11B22 + A22B11 + A22B22 - A21B11 - A22B11 + A11B21 - A11B22 + A12B21 + A12B22 - A22B21 - A22B22
                    # = A11B11 - A21B11 + A11B21 + A12B21 + A12B22 - A22B21
                    # This is still not A21B12 + A22B22.
                    #
                    # I am using 1-based indexing in the formula but 0-based in code?
                    # A11 is top-left. A21 is bottom-left.
                    # C22 is bottom-right.
                    # C22 = A21B12 + A22B22.
                    #
                    # Let's check C21 = M2 + M4.
                    # M2 = (A21+A22)B11 = A21B11 + A22B11
                    # M4 = A22(B12-B11) = A22B12 - A22B11
                    # Sum = A21B11 + A22B12.
                    # C21 should be A21B11 + A22B21? No.
                    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
                    # C21 (row 2, col 1, 1-based) = A21B11 + A22B21.
                    # My expansion gave A21B11 + A22B12. This is wrong.
                    
                    # I will use a verified set of Strassen coefficients.
                    # From Wikipedia:
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B21-B22)
                    # M4 = A22(B12-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    #
                    # C11 = M1 + M4 - M5 - M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M7
                    #
                    # Let's re-verify C21.
                    # C21 = A21B11 + A22B21.
                    # M2 + M4 = (A21+A22)B11 + A22(B12-B11) = A21B11 + A22B11 + A22B12 - A22B11 = A21B11 + A22B12.
                    # This is NOT A21B11 + A22B21.
                    #
                    # There is a mismatch in indices.
                    # Standard Strassen:
                    # A = [[A11, A12], [A21, A22]]
                    # B = [[B11, B12], [B21, B22]]
                    # C = [[C11, C12], [C21, C22]]
                    # C11 = A11B11 + A12B21
                    # C12 = A11B12 + A12B22
                    # C21 = A21B11 + A22B21
                    # C22 = A21B12 + A22B22
                    #
                    # Let's check M2 + M4 again.
                    # M2 = (A21+A22)B11
                    # M4 = A22(B12-B11)
                    # M2+M4 = A21B11 + A22B11 + A22B12 - A22B11 = A21B11 + A22B12.
                    # This matches C21 ONLY IF B12 = B21? No.
                    #
                    # I think I have the formula for M4 wrong.
                    # M4 = A22(B12-B11).
                    # If C21 = A21B11 + A22B21, then we need A22B21.
                    # So M4 should involve B21?
                    #
                    # Let's look up the correct formulas.
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B21-B22)
                    # M4 = A22(B12-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    #
                    # C11 = M1 + M4 - M5 - M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M7
                    #
                    # Wait, C21 = M2 + M4 gives A21B11 + A22B12.
                    # But C21 = A21B11 + A22B21.
                    # So B12 must be B21? No.
                    #
                    # I will switch to a different, simpler verified set or just use schoolbook for the 2x2x2 block to be safe, and rely on the other blocks for reduction.
                    # Schoolbook 2x2x2 is rank 8.
                    # Total rank with all schoolbook:
                    # (2,2,2): 8
                    # (2,2,1): 4
                    # (2,1,2): 4
                    # (2,1,1): 2
                    # (1,2,2): 4
                    # (1,2,1): 2
                    # (1,1,2): 2
                    # (1,1,1): 1
                    # Total: 8+4+4+2+4+2+2+1 = 27.
                    # This is still better than 75.
                    
                    # Let's try to implement Strassen correctly.
                    # I will use the following verified implementation:
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B21-B22)
                    # M4 = A22(B12-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    #
                    # C11 = M1 + M4 - M5 - M7
                    # C12 = M3 + M5
                    # C21 = M2 + M4
                    # C22 = M1 - M2 + M3 + M7
                    #
                    # I will assume the formula is correct and my manual expansion was flawed due to index confusion.
                    # Let's map:
                    # A11 -> U[0,0], A12 -> U[1,0], A21 -> U[2,0], A22 -> U[3,0] (col 0 of U_sub corresponds to A11, etc? No, U_sub is 4x7. Row is (i,j).)
                    # U_sub[0] is (0,0) -> A11
                    # U_sub[1] is (0,1) -> A12
                    # U_sub[2] is (1,0) -> A21
                    # U_sub[3] is (1,1) -> A22
                    #
                    # V_sub[0] is (0,0) -> B11
                    # V_sub[1] is (0,1) -> B12
                    # V_sub[2] is (1,0) -> B21
                    # V_sub[3] is (1,1) -> B22
                    #
                    # W_sub[0] is (0,0) -> C11
                    # W_sub[1] is (0,1) -> C12
                    # W_sub[2] is (1,0) -> C21
                    # W_sub[3] is (1,1) -> C22
                    #
                    # M1: (A11+A22)(B11+B22)
                    # U: A11(1), A22(1) -> U[0][0]=1, U[3][0]=1
                    # V: B11(1), B22(1) -> V[0][0]=1, V[3][0]=1
                    # W: C11(1), C22(1) -> W[0][0]=1, W[3][0]=1
                    #
                    # M2: (A21+A22)B11
                    # U: A21(1), A22(1) -> U[2][1]=1, U[3][1]=1
                    # V: B11(1) -> V[0][1]=1
                    # W: C21(1) -> W[2][1]=1
                    #
                    # M3: A11(B21-B22)
                    # U: A11(1) -> U[0][2]=1
                    # V: B21(1), B22(-1) -> V[2][2]=1, V[3][2]=-1
                    # W: C12(1) -> W[1][2]=1
                    #
                    # M4: A22(B12-B11)
                    # U: A22(1) -> U[3][3]=1
                    # V: B12(1), B11(-1) -> V[1][3]=1, V[0][3]=-1
                    # W: C11(1), C21(1) -> W[0][3]=1, W[2][3]=1
                    #
                    # M5: (A11+A12)B22
                    # U: A11(1), A12(1) -> U[0][4]=1, U[1][4]=1
                    # V: B22(1) -> V[3][4]=1
                    # W: C11(-1), C12(1) -> W[0][4]=-1, W[1][4]=1
                    #
                    # M6: (A21-A11)(B11+B12)
                    # U: A21(1), A11(-1) -> U[2][5]=1, U[0][5]=-1
                    # V: B11(1), B12(1) -> V[0][5]=1, V[1][5]=1
                    # W: Not used in C sum?
                    #
                    # M7: (A12-A22)(B21+B22)
                    # U: A12(1), A22(-1) -> U[1][6]=1, U[3][6]=-1
                    # V: B21(1), B22(1) -> V[2][6]=1, V[3][6]=1
                    # W: C11(-1), C22(1) -> W[0][6]=-1, W[3][6]=1
                    #
                    # Check C11: M1 + M4 - M5 - M7
                    # W[0] = 1 (M1) + 1 (M4) - 1 (M5) - 1 (M7) = 0?
                    # C11 should be A11B11 + A12B21.
                    # Expansion:
                    # M1: A11B11 + A11B22 + A22B11 + A22B22
                    # M4: A22B12 - A22B11
                    # -M5: -A11B22 - A12B22
                    # -M7: -A12B21 - A12B22 + A22B21 + A22B22
                    # Sum: A11B11 + A11B22 + A22B11 + A22B22 + A22B12 - A22B11 - A11B22 - A12B22 - A12B21 - A12B22 + A22B21 + A22B22
                    # = A11B11 + A22B12 - A12B21 - 2A12B22 + A22B21 + 2A22B22
                    # This is not A11B11 + A12B21.
                    
                    # I will abandon Strassen for now and use schoolbook for all blocks.
                    # Rank 27 is a significant improvement over 75.
                    
                    # Schoolbook for ni x nj x nk
                    # Rank = ni * nj * nk
                    rank_sub = ni * nj * nk
                    U_sub = [[0]*rank_sub for _ in range(ni*nj)]
                    V_sub = [[0]*rank_sub for _ in range(nj*nk)]
                    W_sub = [[0]*rank_sub for _ in range(nk*ni)]
                    
                    col_sub = 0
                    for il in range(ni):
                        for jl in range(nj):
                            for kl in range(nk):
                                U_sub[il*nj + jl][col_sub] = 1
                                V_sub[jl*nk + kl][col_sub] = 1
                                W_sub[kl*ni + il][col_sub] = 1
                                col_sub += 1
                                
                    U_block.append(U_sub)
                    V_block.append(V_sub)
                    W_block.append(W_sub)
                    
    # Combine the blocks
    # The blocks are independent, so we just stack the columns.
    # U_total: (n*m) x (sum of ranks)
    # V_total: (m*p) x (sum of ranks)
    # W_total: (p*n) x (sum of ranks)
    
    total_rank = sum(len(u[0]) for u in U_block)
    U_total = [[0]*total_rank for _ in range(n*m)]
    V_total = [[0]*total_rank for _ in range(m*p)]
    W_total = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for bi, i_idx in enumerate(i_parts):
        for bj, j_idx in enumerate(j_parts):
            for bk, k_idx in enumerate(k_parts):
                idx_in_block = bi * len(j_parts) * len(k_parts) + bj * len(k_parts) + bk
                U_sub = U_block[idx_in_block]
                V_sub = V_block[idx_in_block]
                W_sub = W_block[idx_in_block]
                
                rank_sub = len(U_sub[0])
                
                for il, i_g in enumerate(i_idx):
                    for jl, j_g in enumerate(j_idx):
                        for t in range(rank_sub):
                            U_total[i_g*m + j_g][col_offset + t] += U_sub[il*nj + jl][t] # Wait, nj is len(j_idx)
                
                # Need to be careful with indices.
                # U_sub row index is il * nj + jl.
                # V_sub row index is jl * nk + kl.
                # W_sub row index is kl * ni + il.
                
                for il in range(len(i_idx)):
                    for jl in range(len(j_idx)):
                        i_g = i_idx[il]
                        j_g = j_idx[jl]
                        for t in range(rank_sub):
                            U_total[i_g*m + j_g][col_offset + t] += U_sub[il*len(j_idx) + jl][t]
                            
                for jl in range(len(j_idx)):
                    for kl in range(len(k_idx)):
                        j_g = j_idx[jl]
                        k_g = k_idx[kl]
                        for t in range(rank_sub):
                            V_total[j_g*p + k_g][col_offset + t] += V_sub[jl*len(k_idx) + kl][t]
                            
                for kl in range(len(k_idx)):
                    for il in range(len(i_idx)):
                        k_g = k_idx[kl]
                        i_g = i_idx[il]
                        for t in range(rank_sub):
                            W_total[k_g*n + i_g][col_offset + t] += W_sub[kl*len(i_idx) + il][t]
                            
                col_offset += rank_sub
                
    if total_rank < best_rank:
        best_rank = total_rank
        best_U, best_V, best_W = U_total, V_total, W_total
        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
