import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # The goal is to find a rank-minimal exact tensor decomposition.
    # Strategy: Block diagonal composition.
    # We can decompose the multiplication C = A @ B.
    # If we partition A into blocks A11, A12; A21, A22 (2x2 blocks)
    # and B into blocks B11, B21; B12, B22 (2x3 blocks? No, m=4, p=6)
    
    # Let's use the known result for 4x4x6.
    # Schoolbook rank: 4*4*6 = 96.
    # Using Strassen for 2x2x2 blocks:
    # If we treat the problem as a 2x2 block matrix multiplication where each block is 2x2, 
    # but the output is 2x3? No, that doesn't align perfectly.
    
    # Let's try a simpler, robust structural approach:
    # The tensor for 4x4x6 can be viewed as two 4x4x3 tensors side-by-side.
    # The rank of 4x4x3 multiplication is at most 2 * rank(2x2x3) ?
    # Actually, rank(4x4x3) is known to be 12? Or 15?
    # Strassen for 2x2x2 is 7.
    # 4x4x4 is 21 (Strassen).
    # 4x4x3 can be done with rank 12? 
    # If rank(4x4x3) = 12, then rank(4x4x6) = 24.
    # But wait, the lower bound is max(nm, mp, pn) = max(16, 24, 24) = 24.
    # If we can achieve 24, that is optimal.
    # However, it is generally believed that rank(4x4x3) > 12.
    # Actually, rank(2x2x2)=7.
    # rank(4x4x4) <= 21.
    # rank(4x4x3) <= ?
    
    # Let's use a concrete construction that is guaranteed to work and is better than 96.
    # We will use the block structure with Strassen's algorithm on 2x2x2 blocks.
    # Partition n=4 into two 2s. m=4 into two 2s. p=6 into three 2s.
    # This gives a 2x2x3 grid of 2x2x2 subproblems.
    # Total subproblems: 2 * 2 * 3 = 12.
    # If we use Strassen (rank 7) for each, total rank = 12 * 7 = 84.
    # This is a known valid construction (as noted in the "Knowledge" section: "Block-decomposing 4x4x6 into 12 sub-problems of 2x2x2 using Strassen's rank-7 decomposition for each, achieving total rank 84").
    
    # Can we do better?
    # Perhaps by sharing Strassen terms between adjacent blocks?
    # For 4x4x4, Strassen uses 21 terms (not 8*7=56) by sharing.
    # For 4x4x6, we can view it as 4x4x4 + 4x4x2.
    # Rank(4x4x4) = 21.
    # Rank(4x4x2) ? 
    # 4x4x2 is like two 4x4x1.
    # Rank(4x4x1) = 16? (Schoolbook)
    # Can we do better for 4x4x2?
    # 2x2x1 rank is 4? No, 2x2x1 is just dot products, rank 4.
    # 4x4x2 can be decomposed into 2x2 blocks.
    # It is known that rank(4x4x2) = 16? Or less?
    # Actually, rank(2x2x2)=7.
    # rank(4x4x2) <= 2 * rank(2x2x2) = 14?
    # If we partition n=4, m=4 into 2x2 blocks, and p=2 is a single column.
    # This is not a standard cubic block.
    
    # Let's stick to the robust 84 construction first, as it is guaranteed to be valid and integer-based.
    # Then, we will attempt to verify if a lower rank is easily constructible via simple partitioning.
    # If we can find a rank 24 decomposition, that would be a huge jump.
    # But rank 24 is the theoretical minimum. It is unlikely to be achievable for 4x4x6.
    # The best known for 4x4x4 is 21.
    # For 4x4x6, a reasonable target is 42 (21 * 6/4? No).
    # 21 (for 4x4x4) + 12 (for 4x4x2) = 33?
    # If rank(4x4x2) = 12, then total 33.
    # Is rank(4x4x2) = 12?
    # 4x4x2 tensor.
    # Using Strassen on 2x2x2 blocks?
    # Partition n=4, m=4 into 2x2. p=2 is not 2.
    # We can view 4x4x2 as 2x (4x4x1).
    # Or 4x4x2 = [2x2x2, 2x2x2; 2x2x2, 2x2x2] but p is only 2.
    # So we have 4 blocks of 2x2x1?
    # Rank(2x2x1) = 4.
    # Total 16.
    # So 21 + 16 = 37.
    
    # Let's implement the 84 Strassen block construction. It is safe and valid.
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
        # A (2x2), B (2x2), C (2x2).
        # C = A @ B
        # Strassen formulas:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map this to U, V, W matrices.
        # U: (n*m, rank) = (4, 7)
        # V: (m*p, rank) = (4, 7)
        # W: (p*n, rank) = (4, 7)
        
        # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        # A: (1,1) and (2,2) -> indices 0 and 3
        # B: (1,1) and (2,2) -> indices 0 and 3
        # C: (1,1) -> index 0, (2,2) -> index 3
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        # A: (2,1) and (2,2) -> indices 2 and 3
        # B: (1,1) -> index 0
        # C: (2,1) -> index 2, (2,2) -> index 3
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1 # Wait, C21 = M2 + M4. C22 = M1 - M2 ...
        # Let's check Strassen formulas carefully.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # Let's re-assign W for M2.
        # C21 = M2 + M4. So W[2, 1] = 1.
        # C22 = M1 - M2 + M3 + M6. So W[3, 1] = -1.
        # C11, C12 do not involve M2. So W[0, 1]=0, W[1, 1]=0.
        
        # M3: A11(B12-B22)
        # A11 -> index 0.
        # B12 -> index 1, B22 -> index 3.
        # C12 = M3 + M5. So W[1, 2] = 1.
        # C22 = ... + M3 ... So W[3, 2] = 1.
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: A22(B21-B11)
        # A22 -> index 3.
        # B21 -> index 2, B11 -> index 0.
        # C11 = ... + M4 ... So W[0, 3] = 1.
        # C21 = M2 + M4. So W[2, 3] = 1.
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5: (A11+A12)B22
        # A11 -> 0, A12 -> 1.
        # B22 -> 3.
        # C11 = ... - M5 ... So W[0, 4] = -1.
        # C12 = M3 + M5. So W[1, 4] = 1.
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6: (A21-A11)(B11+B12)
        # A21 -> 2, A11 -> 0.
        # B11 -> 0, B12 -> 1.
        # C22 = ... + M6 ... So W[3, 5] = 1.
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7: (A12-A22)(B21+B22)
        # A12 -> 1, A22 -> 3.
        # B21 -> 2, B22 -> 3.
        # C11 = ... + M7 ... So W[0, 6] = 1.
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def verify_decomp(U, V, W, n, m, p):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # W index: k*n + i
        R = U.shape[1]
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # Target: C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor T[i, j, k] = 1 if valid? No.
        # The tensor decomposition represents the bilinear map.
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # This should equal 1 if the term A[i, j] B[j, k] contributes to C[i, k].
        # In standard matrix multiplication, each A[i, j] B[j, k] contributes exactly 1 to C[i, k].
        # So the target tensor is all ones.
        
        computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        return np.allclose(computed, target, atol=1e-6)

    # Construct 4x4x6 using 12 blocks of 2x2x2.
    # n=4, m=4, p=6.
    # Blocks:
    # n blocks: 2 (size 2)
    # m blocks: 2 (size 2)
    # p blocks: 3 (size 2)
    
    n_blocks = 2
    m_blocks = 2
    p_blocks = 3
    total_blocks = n_blocks * m_blocks * p_blocks
    block_rank = 7
    total_rank = total_blocks * block_rank
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    U_block, V_block, W_block = get_strassen_2x2x2()
    
    rank_offset = 0
    for ib in range(n_blocks):
        for jb in range(m_blocks):
            for kb in range(p_blocks):
                # Map block indices to global indices
                # U: (n*m, R). Index is i*m + j.
                # Block ib covers rows [ib*2, ib*2+1] of A.
                # Block jb covers cols [jb*2, jb*2+1] of A.
                # So U indices are (ib*2 + i_local)*m + (jb*2 + j_local)
                
                # V: (m*p, R). Index is j*p + k.
                # Block jb covers rows [jb*2, jb*2+1] of B.
                # Block kb covers cols [kb*2, kb*2+1] of B.
                # So V indices are (jb*2 + j_local)*p + (kb*2 + k_local)
                
                # W: (p*n, R). Index is k*n + i.
                # Block kb covers rows [kb*2, kb*2+1] of C (p dim).
                # Block ib covers cols [ib*2, ib*2+1] of C (n dim).
                # So W indices are (kb*2 + k_local)*n + (ib*2 + i_local)
                
                for r in range(block_rank):
                    # U
                    for i_loc in range(2):
                        for j_loc in range(2):
                            val = U_block[i_loc * 2 + j_loc, r]
                            if val != 0:
                                g_i = ib * 2 + i_loc
                                g_j = jb * 2 + j_loc
                                U_full[g_i * m + g_j, rank_offset + r] += val
                                
                    # V
                    for j_loc in range(2):
                        for k_loc in range(2):
                            val = V_block[j_loc * 2 + k_loc, r]
                            if val != 0:
                                g_j = jb * 2 + j_loc
                                g_k = kb * 2 + k_loc
                                V_full[g_j * p + g_k, rank_offset + r] += val
                                
                    # W
                    for k_loc in range(2):
                        for i_loc in range(2):
                            val = W_block[k_loc * 2 + i_loc, r]
                            if val != 0:
                                g_k = kb * 2 + k_loc
                                g_i = ib * 2 + i_loc
                                W_full[g_k * n + g_i, rank_offset + r] += val
                                
                rank_offset += block_rank

    # Verify
    if verify_decomp(U_full, V_full, W_full, n, m, p):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
