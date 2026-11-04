"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    # Rank 49 decomposition via Strassen on 4x4x4 padded to 4x4x8
    # We compute C = A * B where A is 3x4, B is 4x6
    # Pad A to 4x4 (A4) and B to 4x8 (B4)
    # C4 = A4 * B4 is 4x8, we only need the 3x6 part
    
    # Strassen 4x4x4 uses 8 multiplications of 2x2x2
    # But we need 4x4x8, so we split B into two 4x4 blocks
    # C4 = A4 * B4 = [A4*B4_left, A4*B4_right]
    # Each 4x4x4 uses 8 multiplications of 2x2x2
    # Total: 16 multiplications of 2x2x2
    # Each 2x2x2 uses 7 multiplications
    # Total: 16 * 7 = 112? That's worse than 72.
    
    # Let me reconsider. Strassen 4x4x4:
    # Split 4x4 into 2x2 blocks.
    # M4 = A * B where A, B are 4x4
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = (A11+A22)*(B11+B22) - (A21+A22)*B22 + A11*B11
    # Wait, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2x2 multiplication, rank 7.
    # So 4x4x4 has rank 8*7 = 56? No, that's not right.
    # The rank of 4x4x4 via Strassen is 8 * r(2,2,2) = 8 * 7 = 56.
    # But schoolbook is 64. So 56 < 64. Good.
    
    # For 4x4x8: split B into B_left (4x4) and B_right (4x4)
    # C = [A*B_left, A*B_right]
    # Each A*B is 4x4x4 with rank 56
    # Total rank: 56 + 56 = 112. Worse than 72.
    
    # I need a better approach. Let me use the fact that A is 3x4, not 4x4.
    # Actually, the padding wastes rank.
    
    # Let me try a different approach: use rank 23 Winograd for 3x3x3.
    # Split B into two 4x3 blocks: B1, B2
    # C1 = A * B1 (3x3), C2 = A * B2 (3x3)
    # Each is 3x4x3. What's the rank of 3x4x3?
    # It's at most 3*4*3 = 36, but can be lower.
    # Actually, 3x4x3 can be seen as 3 independent 1x4x3 multiplications.
    # 1x4x3 has rank 4 (schoolbook: c_k = sum_j a_j b_jk, needs 4 mults? No, 3 mults if we factor).
    # Actually 1x4x3: c_k = a_0 b_0k + a_1 b_1k + a_2 b_2k + a_3 b_3k
    # This needs 4 multiplications: a_0 b_0k, a_1 b_1k, a_2 b_2k, a_3 b_3k for each k.
    # But we can do: m_0 = a_0, m_1 = a_1, m_2 = a_2, m_3 = a_3, then c_k = m_0 b_0k + m_1 b_1k + m_2 b_2k + m_3 b_3k
    # That's 4 multiplications per k, total 12. But the m's are shared.
    # Actually the rank of 1x4x3 is 4 (since we can factor out the a's).
    # So 3x4x3 has rank 3*4 = 12? No, the a's are shared across k.
    # C_ik = sum_j A_ij B_jk
    # For fixed i: c_k = sum_j A_ij B_jk, this is a 1x4x3 tensor with rank 4.
    # For 3 different i's, if the A_ij are independent, the total rank is 3*4 = 12.
    # So 3x4x3 has rank at most 12.
    # Then 3x4x6 = 3x4x3 + 3x4x3, total rank 12 + 12 = 24.
    # That's much better than 72!
    
    # Let me verify: C_ik = sum_j A_ij B_jk
    # For i=0: c_0k = A_00 B_0k + A_01 B_1k + A_02 B_2k + A_03 B_3k
    # For i=1: c_1k = A_10 B_0k + A_11 B_1k + A_12 B_2k + A_13 B_3k
    # For i=2: c_2k = A_20 B_0k + A_21 B_1k + A_22 B_2k + A_23 B_3k
    
    # The multiplications needed:
    # For each j and k: A_ij B_jk
    # Total: 3 * 4 * 6 = 72. That's schoolbook.
    
    # But if we factor: for each i, we have 4 terms.
    # The rank of the 3x4x3 tensor (for one 3-column block of B) is:
    # It's the tensor T_ikj = A_ij B_jk for j=0..3, k=0..2, i=0..2
    # This is a 3x3x4 tensor (i, k, j).
    # Its rank is at most 12 (3*4), achieved by:
    # For each j: m_jk = B_jk, then c_ik = sum_j A_ij m_jk
    # But m_jk depends on k, so we need 4*3 = 12 "multiplications" if we count A_ij * B_jk.
    # Actually, the standard rank of 3x4x3 is 12 (since it's a sum of 12 rank-1 tensors).
    # Can it be lower? Probably not, since the slices are independent.
    
    # So 3x4x6 has rank at most 24 (two blocks of 3x4x3, each rank 12).
    # But wait, the two blocks share the same A, so the U factors are shared.
    # Let me construct the rank 24 decomposition.
    
    # For B1 (4x3, cols 0-2):
    # C1_ik = sum_j A_ij B1_jk for k=0,1,2
    # Multiplications: for each j=0..3 and k=0..2: m_jk = A_ij B1_jk? No, A_ij depends on i.
    # Actually, the multiplication is A_ij * B_jk, which depends on both i and k.
    # So we can't factor out A_ij easily.
    
    # The rank-1 decomposition of 3x4x3:
    # T(i,k,j) = A(i,j) * B(j,k)
    # This is not a rank-1 tensor in general.
    # The tensor is T(i,k) = sum_j A(i,j) B(j,k)
    # As a tensor in R^{3 x 3 x 4} (i, k, j), it's T(i,k,j) = A(i,j) B(j,k)
    # This has rank at most 12, with the decomposition:
    # T = sum_j (A(:,j) * B(j,:)) where A(:,j) is a 3x1 vector and B(j,:) is a 1x3 vector
    # So T(i,k,j) = sum_j A(i,j) B(j,k) = sum_j [A(i,j)] * [B(j,k)]
    # Each term is a rank-1 tensor: u_i = A(i,j), v_k = B(j,k), w_j = 1
    # So the rank is at most 4 (one for each j), not 12!
    
    # Wait, that's for a fixed i and k. Let me re-examine.
    # T(i,k) = sum_j A(i,j) B(j,k)
    # The tensor T has indices (i, k) and the "fiber" is over j.
    # As a 3-way tensor T(i,k,j), the rank is the minimum number of rank-1 tensors.
    # T(i,k,j) = A(i,j) B(j,k)
    # This is not separable as u(i) v(k) w(j) in general.
    # The rank of this tensor is at most 3*4 = 12 (by fixing i and k, but that's not right).
    
    # Actually, the standard result: the rank of M_{n,m,p} is at most n*m*p (schoolbook).
    # For 3x4x3, schoolbook is 36.
    # But we can do better: for each j, we have A(i,j) B(j,k) for all i,k.
    # This is a rank-1 tensor in the (i,k) space for each j, but the "weight" is 1.
    # So T = sum_j (A(:,j) outer B(j,:)) * 1
    # Each term is a rank-1 tensor in R^{3x3x1}, but we need R^{3x3x4}.
    # To make it a rank-1 tensor in R^{3x3x4}, we need w(j) to be 1 for the specific j and 0 otherwise.
    # So T(i,k,j) = sum_{j'} [A(i,j') B(j',k)] * [delta_{j,j'}]
    # This is a sum of 4 rank-1 tensors, each with w being a standard basis vector.
    # So the rank is at most 4!
    
    # Wait, that can't be right. Let me check with a simple case.
    # M_{1,2,1}: c = a0*b0 + a1*b1
    # T(i,k,j) with i=0, k=0, j=0,1: T(0,0,0) = a0*b0, T(0,0,1) = a1*b1
    # Rank-1 decomposition: T = a0*b0 * e0 + a1*b1 * e1
    # This is 2 rank-1 tensors. Rank is 2.
    # Schoolbook is 2. So rank is 2.
    
    # For M_{3,4,3}:
    # T(i,k,j) = A(i,j) B(j,k) for i=0..2, k=0..2, j=0..3
    # T = sum_{j=0}^{3} [A(:,j) outer B(j,:)] * e_j
    # Each term is a rank-1 tensor: u_i = A(i,j), v_k = B(j,k), w_j = delta_{j,j'}
    # So the rank is at most 4.
    
    # That seems too good. Let me verify with a specific example.
    # A = [[1,0,0,0],[0,1,0,0],[0,0,1,0]], B = [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
    # C = A*B = [[1,0,0],[0,1,0],[0,0,1]]
    # T(i,k,j) = A(i,j) B(j,k)
    # T(0,0,0) = 1*1 = 1, T(0,0,1) = 0*0 = 0, T(0,0,2) = 0*0 = 0, T(0,0,3) = 0*0 = 0
    # T(0,1,0) = 1*0 = 0, T(0,1,1) = 0*1 = 0, T(0,1,2) = 0*0 = 0, T(0,1,3) = 0*0 = 0
    # T(1,0,0) = 0*1 = 0, T(1,0,1) = 1*0 = 0, T(1,0,2) = 0*0 = 0, T(1,0,3) = 0*0 = 0
    # T(1,1,0) = 0*0 = 0, T(1,1,1) = 1*1 = 1, T(1,1,2) = 0*0 = 0, T(1,1,3) = 0*0 = 0
    # T(2,0,0) = 0*1 = 0, T(2,0,1) = 0*0 = 0, T(2,0,2) = 1*0 = 0, T(2,0,3) = 0*0 = 0
    # T(2,1,0) = 0*0 = 0, T(2,1,1) = 0*1 = 0, T(2,1,2) = 1*0 = 0, T(2,1,3) = 0*0 = 0
    # T(2,2,0) = 0*0 = 0, T(2,2,1) = 0*0 = 0, T(2,2,2) = 1*1 = 1, T(2,2,3) = 0*0 = 0
    
    # The non-zero terms are T(0,0,0)=1, T(1,1,1)=1, T(2,2,2)=1
    # This can be decomposed as:
    # T = 1*e0_i * e0_k * e0_j + 1*e1_i * e1_k * e1_j + 1*e2_i * e2_k * e2_j
    # That's 3 rank-1 tensors, not 4.
    # But in general, it could be up to 4.
    
    # So for 3x4x3, the rank is at most 4.
    # For 3x4x6, split into two 3x4x3 blocks: rank at most 4 + 4 = 8.
    # That's way better than 72!
    
    # Let me construct the rank 8 decomposition.
    # B1 = B[:, 0:3], B2 = B[:, 3:6]
    # C1 = A * B1, C2 = A * B2
    # For C1: T1(i,k,j) = A(i,j) B1(j,k) for j=0..3, k=0..2
    # Decomposition: sum_{j=0}^{3} A(i,j) * B1(j,k) * delta_j
    # For C2: T2(i,k,j) = A(i,j) B2(j,k) for j=0..3, k=0..2
    # Decomposition: sum_{j=0}^{3} A(i,j) * B2(j,k) * delta_j
    
    # Total 8 terms.
    # U has 12 rows (3*4), V has 24 rows (4*6), W has 18 rows (6*3)
    # Wait, W is indexed by k*n+i = k*3+i for k=0..5, i=0..2
    
    # For term (j, block):
    # block=0 (B1): k in 0..2, so W index = k*3+i
    # block=1 (B2): k in 3..5, so W index = k*3+i
    
    # U[i*4+j] = 1 for the term (j, block)
    # V[j*6+k] = 1 for the term (j, block) where k is the column in the full B
    # W[k*3+i] = 1 for the term (j, block)
    
    rank = 8
    U = [[0] * rank for _ in range(12)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(18)]
    
    col = 0
    for block in range(2):
        for j in range(4):
            for i in range(3):
                for k in range(3):
                    k_full = block * 3 + k
                    U[i*4+j][col] = 1
                    V[j*6+k_full][col] = 1
                    W[k_full*3+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
