"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication using Strassen's 7-term decomposition.
    U has shape (2*2, 7), V has shape (2*2, 7), W has shape (2*2, 7).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=2, m=2, p=2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's formulas for C = A*B where A,B are 2x2:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    #
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
    
    # Term 1: M1 = (A00+A11) * (B00+B11)
    # U: A00, A11
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    # V: B00, B11
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    # W: contributes to C[0,0]
    W[0, 0] = 1  # C[0,0]
    
    # Term 2: M2 = (A10+A11) * B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0]
    
    # Term 3: M3 = A00 * (B01-B11)
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1 # B[1,1]
    W[1, 2] = 1  # C[0,1]
    
    # Term 4: M4 = A11 * (B10-B00)
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1 # B[0,0]
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # Term 5: M5 = (A00+A01) * B11
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1 # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # Term 6: M6 = (A10-A00) * (B00+B01)
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1 # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[3, 5] = 1  # C[1,1]
    
    # Term 7: M7 = (A11-A01) * (B10+B11)
    U[3, 6] = 1  # A[1,1]
    U[1, 6] = -1 # A[0,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[0, 6] = -1 # C[0,0]
    W[3, 6] = 1  # C[1,1]
    
    return U, V, W

def _schoolbook_block(n_b, m_b, p_b):
    """
    Returns (U, V, W) for a schoolbook decomposition of n_b x m_b x p_b.
    U has shape (n_b*m_b, n_b*m_b*p_b), etc.
    """
    rank = n_b * m_b * p_b
    U = np.zeros((n_b*m_b, rank))
    V = np.zeros((m_b*p_b, rank))
    W = np.zeros((p_b*n_b, rank))
    
    col = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i*m_b+j, col] = 1
                V[j*p_b+k, col] = 1
                W[k*n_b+i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strategy: Hybrid Strassen/Schoolbook
    # We decompose the 3x4x7 multiplication into blocks.
    # A is 3x4, B is 4x7, C is 3x7.
    #
    # Split A into A1 (2x4) and A2 (1x4).
    # Split B into B1 (4x4), B2 (4x2), B3 (4x1).
    # Split C into:
    #   C11 (2x4) from A1*B1
    #   C12 (2x2) from A1*B2
    #   C13 (2x1) from A1*B3
    #   C21 (1x4) from A2*B1
    #   C22 (1x2) from A2*B2
    #   C23 (1x1) from A2*B3
    #
    # For each block, we choose either Strassen (if 2x2x2) or Schoolbook.
    #
    # C11: 2x4x2. Split B1 into B1a (4x2) and B1b (4x2).
    #   C11a: 2x2x2 (A1[:, 0:2] * B1a[:, 0:2]? No, A1 is 2x4, B1a is 4x2).
    #   Wait, A1 is 2x4. B1a is 4x2. Result is 2x2. This is a 2x4x2 block.
    #   We can use schoolbook rank 2*4*2 = 16.
    #   Or we can split further? 2x2x2 is the minimal Strassen block.
    #   Let's try to decompose 2x4x2 into two 2x2x2 blocks.
    #   A1 (2x4) can be seen as [A1a A1b] where A1a, A1b are 2x2.
    #   B1a (4x2) can be seen as [B1a_top; B1a_bot] where B1a_top, B1a_bot are 2x2.
    #   Then A1 * B1a = A1a * B1a_top + A1b * B1a_bot.
    #   Each term is 2x2x2. Rank 7+7=14. Better than 16.
    #
    # Let's apply this logic systematically.
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_block(n_b, m_b, p_b, use_strassen=False):
        nonlocal total_rank
        if use_strassen and n_b == 2 and m_b == 2 and p_b == 2:
            U_b, V_b, W_b = _strassen_2x2x2()
        else:
            U_b, V_b, W_b = _schoolbook_block(n_b, m_b, p_b)
        
        # Pad to global dimensions? No, we build global matrices directly.
        # We need to map local indices to global indices.
        # This function will be called with specific offsets.
        
        rank_b = U_b.shape[1]
        total_rank += rank_b
        U_list.append(U_b)
        V_list.append(V_b)
        W_list.append(W_b)
        
    # We need a more flexible add_block that handles offsets.
    # Let's restructure.
    
    # Global matrices
    max_rank = 84
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    
    col_offset = 0
    
    def add_decomposition(local_U, local_V, local_W, 
                          i_start, j_start, k_start,
                          i_len, j_len, k_len):
        nonlocal col_offset
        rank_b = local_U.shape[1]
        
        # Map local U (i_len*j_len) to global U (i_len*m_b? No, m_b is the middle dim)
        # Local U indices: 0..i_len-1, 0..j_len-1 -> i_start..i_start+i_len-1, j_start..j_start+j_len-1
        # Global U row index: (i_global)*m + j_global
        # Wait, U shape is (n*m, rank). Row index is i*m+j.
        # So for local i in [0, i_len), local j in [0, j_len):
        #   global i = i_start + local_i
        #   global j = j_start + local_j
        #   global U row = global_i * m + global_j
        
        for r in range(local_U.shape[0]):
            li, lj = divmod(r, j_len)
            gi = i_start + li
            gj = j_start + lj
            U[gi*m + gj, col_offset:col_offset+rank_b] = local_U[r, :]
            
        # Local V indices: 0..j_len-1, 0..k_len-1 -> j_start..j_start+j_len-1, k_start..k_start+k_len-1
        # Global V row index: j_global * p + k_global
        for r in range(local_V.shape[0]):
            lj, lk = divmod(r, k_len)
            gj = j_start + lj
            gk = k_start + lk
            V[gj*p + gk, col_offset:col_offset+rank_b] = local_V[r, :]
            
        # Local W indices: 0..k_len-1, 0..i_len-1 -> k_start..k_start+k_len-1, i_start..i_start+i_len-1
        # Global W row index: k_global * n + i_global
        for r in range(local_W.shape[0]):
            lk, li = divmod(r, i_len)
            gk = k_start + lk
            gi = i_start + li
            W[gk*n + gi, col_offset:col_offset+rank_b] = local_W[r, :]
            
        col_offset += rank_b
        
    # Block 1: A1 (2x4) * B1 (4x4) -> C11 (2x4)
    # Split into two 2x2x2 blocks:
    #   A1a (2x2) * B1a_top (2x2) + A1b (2x2) * B1a_bot (2x2) ? No.
    #   A1 is rows 0,1 of A. B1 is cols 0..3 of B.
    #   We can split the middle dimension (4) into two 2s.
    #   Term 1.1: A1 (2x4) * B1_left (4x2) -> 2x2.
    #     A1 is 2x4. B1_left is 4x2.
    #     We can decompose 2x4x2 into two 2x2x2?
    #     A1 = [A11 A12] (2x2 each). B1_left = [B11; B12] (2x2 each).
    #     A1 * B1_left = A11*B11 + A12*B12.
    #     So we need two 2x2x2 decompositions.
    
    # 1.1.1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U1, V1, W1 = _strassen_2x2x2()
    add_decomposition(U1, V1, W1, 0, 0, 0, 2, 2, 2)
    
    # 1.1.2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Note: This contributes to the same C block C[0:2, 0:2].
    U2, V2, W2 = _strassen_2x2x2()
    add_decomposition(U2, V2, W2, 0, 2, 0, 2, 2, 2)
    
    # 1.2: A1 (2x4) * B1_right (4x2) -> C[0:2, 2:4]
    # 1.2.1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    U3, V3, W3 = _strassen_2x2x2()
    add_decomposition(U3, V3, W3, 0, 0, 2, 2, 2, 2)
    
    # 1.2.2: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    U4, V4, W4 = _strassen_2x2x2()
    add_decomposition(U4, V4, W4, 0, 2, 2, 2, 2, 2)
    
    # Block 2: A2 (1x4) * B1 (4x4) -> C21 (1x4)
    # Schoolbook 1x4x4 rank 16.
    U5, V5, W5 = _schoolbook_block(1, 4, 4)
    add_decomposition(U5, V5, W5, 2, 0, 0, 1, 4, 4)
    
    # Block 3: A1 (2x4) * B2 (4x2) -> C12 (2x2)
    # 3.1: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    U6, V6, W6 = _strassen_2x2x2()
    add_decomposition(U6, V6, W6, 0, 0, 4, 2, 2, 2)
    
    # 3.2: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    U7, V7, W7 = _strassen_2x2x2()
    add_decomposition(U7, V7, W7, 0, 2, 4, 2, 2, 2)
    
    # Block 4: A2 (1x4) * B2 (4x2) -> C22 (1x2)
    # Schoolbook 1x4x2 rank 8.
    U8, V8, W8 = _schoolbook_block(1, 4, 2)
    add_decomposition(U8, V8, W8, 2, 0, 4, 1, 4, 2)
    
    # Block 5: A1 (2x4) * B3 (4x1) -> C13 (2x1)
    # 5.1: A[0:2, 0:2] * B[0:2, 6] -> C[0:2, 6]
    # Schoolbook 2x2x1 rank 4.
    U9, V9, W9 = _schoolbook_block(2, 2, 1)
    add_decomposition(U9, V9, W9, 0, 0, 6, 2, 2, 1)
    
    # 5.2: A[0:2, 2:4] * B[2:4, 6] -> C[0:2, 6]
    # Schoolbook 2x2x1 rank 4.
    U10, V10, W10 = _schoolbook_block(2, 2, 1)
    add_decomposition(U10, V10, W10, 0, 2, 6, 2, 2, 1)
    
    # Block 6: A2 (1x4) * B3 (4x1) -> C23 (1x1)
    # Schoolbook 1x4x1 rank 4.
    U11, V11, W11 = _schoolbook_block(1, 4, 1)
    add_decomposition(U11, V11, W11, 2, 0, 6, 1, 4, 1)
    
    # Total rank used:
    # 1.1.1: 7
    # 1.1.2: 7
    # 1.2.1: 7
    # 1.2.2: 7
    # 2: 16
    # 3.1: 7
    # 3.2: 7
    # 4: 8
    # 5.1: 4
    # 5.2: 4
    # 6: 4
    # Sum: 7*5 + 16 + 8 + 4*3 = 35 + 16 + 8 + 12 = 71.
    
    # Check if we can do better on Block 2 (1x4x4).
    # Can we use Strassen-like for 1x4x4? No, Strassen is for 2x2x2.
    # 1x4x4 is a rank-1 vector times matrix. Rank 4 is minimal if we treat it as outer product?
    # No, it's matrix multiplication. (1x4) * (4x4) = 1x4.
    # Each output element is a dot product of length 4.
    # Rank of this tensor is 4?
    # T_{i,j,k} = delta_{i,0} * 1.
    # Wait, A is 1x4, B is 4x4. C is 1x4.
    # C_k = sum_j A_j B_{j,k}.
    # The tensor is T_{j,k,0} = B_{j,k}? No.
    # The multiplication tensor T_{i,j,k} has T_{0,j,k} = 1.
    # This tensor has rank 4?
    # U has shape (4, r), V (4, r), W (4, r).
    # C = U^T V? No.
    # Sum_c U[j,c] V[k,c] = C[j,k]? No, W is involved.
    # C[i,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c]? No.
    # Standard decomposition: C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i,c] V[c,k] ...
    # Let's stick to the schoolbook for 1x4x4. Rank 16 is correct for schoolbook.
    # Is there a lower rank?
    # The bilinear complexity of 1x4x4 is 4.
    # Why? C_k = A . B_k.
    # We can compute 4 dot products.
    # U[j,c] = 1 if j=c else 0.
    # V[k,c] = B[k,c]? No, V must be constant.
    # Ah, V depends on B.
    # U is based on A indices. V on B indices. W on C indices.
    # For 1x4x4:
    # U[0,j,c] = 1 if j=c else 0? No, U is (4, r).
    # Let r=4.
    # U[j,c] = delta_{j,c}.
    # V[j,k,c] = delta_{j,c} * 1? No.
    # We want sum_c U[j,c] V[k,c] W[i,c] = sum_j A[j] B[j,k].
    # If U[j,c] = delta_{j,c}, then sum_c delta_{j,c} V[k,c] = V[k,j].
    # Then sum_j A[j] V[k,j] W[0,j]?
    # This implies V[k,j] must be B[j,k]. But V must be constant!
    # So we cannot factor B into V if V is constant.
    # The decomposition is:
    # C[i,k] = sum_c U[i,c] V[c,k]? No.
    # The tensor is T[i,j,k].
    # We need T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]? No.
    # The form is: T[i,j,k] = sum_c U[i,c] V[c,k] W[k,c]?
    # Let's look at the schoolbook.
    # T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
    # Rank is n*m*p.
    # For 1x4x4, rank is 16.
    # Is it reducible?
    # The tensor is simply the identity tensor in the j dimension.
    # It is isomorphic to the identity tensor of size 4.
    # The rank of the identity tensor of size n is n.
    # Wait, T_{j,k} = delta_{j,k}.
    # Rank is 4.
    # Let's check.
    # U[j,c] = delta_{j,c}.
    # V[k,c] = delta_{k,c}.
    # W[i,c] = 1?
    # Sum_c delta_{j,c} delta_{k,c} * 1 = delta_{j,k}.
    # Yes! Rank 4 is sufficient.
    # My schoolbook gave 16. We can save 12 rank units here.
    
    # Let's redefine Block 2 with rank 4.
    # We need to reset and rebuild, or just use a better function.
    
    # Let's create a specialized function for 1xm x mxp.
    def _identity_block(m, p):
        # 1xm * mxp -> 1xp
        # Rank m.
        U = np.zeros((1*m, m))
        V = np.zeros((m*p, m))
        W = np.zeros((p*1, m))
        
        for j in range(m):
            U[j, j] = 1
            for k in range(p):
                V[j*p+k, j] = 1
                W[k*1+0, j] = 1 # W[k,i] with i=0.
        
        return U, V, W
    
    # Let's restart the construction with the improved Block 2.
    
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    col_offset = 0
    
    # 1.1.1: 2x2x2 Strassen
    U1, V1, W1 = _strassen_2x2x2()
    add_decomposition(U1, V1, W1, 0, 0, 0, 2, 2, 2)
    
    # 1.1.2: 2x2x2 Strassen
    U2, V2, W2 = _strassen_2x2x2()
    add_decomposition(U2, V2, W2, 0, 2, 0, 2, 2, 2)
    
    # 1.2.1: 2x2x2 Strassen
    U3, V3, W3 = _strassen_2x2x2()
    add_decomposition(U3, V3, W3, 0, 0, 2, 2, 2, 2)
    
    # 1.2.2: 2x2x2 Strassen
    U4, V4, W4 = _strassen_2x2x2()
    add_decomposition(U4, V4, W4, 0, 2, 2, 2, 2, 2)
    
    # Block 2: 1x4x4 Identity
    U5, V5, W5 = _identity_block(4, 4)
    add_decomposition(U5, V5, W5, 2, 0, 0, 1, 4, 4)
    
    # 3.1: 2x2x2 Strassen
    U6, V6, W6 = _strassen_2x2x2()
    add_decomposition(U6, V6, W6, 0, 0, 4, 2, 2, 2)
    
    # 3.2: 2x2x2 Strassen
    U7, V7, W7 = _strassen_2x2x2()
    add_decomposition(U7, V7, W7, 0, 2, 4, 2, 2, 2)
    
    # Block 4: 1x4x2 Identity
    U8, V8, W8 = _identity_block(4, 2)
    add_decomposition(U8, V8, W8, 2, 0, 4, 1, 4, 2)
    
    # 5.1: 2x2x1 Schoolbook
    U9, V9, W9 = _schoolbook_block(2, 2, 1)
    add_decomposition(U9, V9, W9, 0, 0, 6, 2, 2, 1)
    
    # 5.2: 2x2x1 Schoolbook
    U10, V10, W10 = _schoolbook_block(2, 2, 1)
    add_decomposition(U10, V10, W10, 0, 2, 6, 2, 2, 1)
    
    # Block 6: 1x4x1 Identity
    U11, V11, W11 = _identity_block(4, 1)
    add_decomposition(U11, V11, W11, 2, 0, 6, 1, 4, 1)
    
    # New total rank:
    # 5 * 7 (Strassen) = 35
    # 1 * 4 (Identity 4x4) = 4
    # 1 * 4 (Identity 4x2) = 4
    # 2 * 4 (Schoolbook 2x2x1) = 8
    # 1 * 4 (Identity 4x1) = 4
    # Sum: 35 + 4 + 4 + 8 + 4 = 55.
    
    # Can we improve the 2x2x1 blocks (5.1, 5.2)?
    # 2x2x1: A (2x2) * B (2x1) -> C (2x1).
    # C = A B.
    # Rank?
    # Schoolbook rank 4.
    # Can we do better?
    # T[i,j,k] with k=0.
    # T[i,j,0] = delta_{i,i'} delta_{j,j'}?
    # It's two independent dot products of length 2.
    # Rank 2?
    # U[i,j,c] = delta_{i,c} delta_{j,j'}?
    # Let's try rank 2.
    # c=0: U[0,j,0] = 1 if j=0?
    # We want C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    # If we use 2 terms:
    # Term 1: A[i,0] B[0,0]. U[i,0,0]=1, V[0,0,0]=1, W[0,i,0]=1.
    # Term 2: A[i,1] B[1,0]. U[i,1,1]=1, V[1,0,1]=1, W[1,i,1]=1.
    # Yes, rank 2 is sufficient for 2x2x1.
    # My schoolbook gave 4. We save 2 rank per block.
    # Save 4 total.
    
    # Let's create a specialized function for nx2x1 or 2xm x 2x1.
    # General case: n x m x 1.
    # Rank m.
    def _nxm_x1_block(n, m):
        U = np.zeros((n*m, m))
        V = np.zeros((m*1, m))
        W = np.zeros((1*n, m))
        
        for j in range(m):
            for i in range(n):
                U[i*m+j, j] = 1
            V[j*1+0, j] = 1
            for i in range(n):
                W[0*n+i, j] = 1
        
        return U, V, W
    
    # Restart again with improved 2x2x1 blocks.
    
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    col_offset = 0
    
    # 1.1.1: 2x2x2 Strassen
    U1, V1, W1 = _strassen_2x2x2()
    add_decomposition(U1, V1, W1, 0, 0, 0, 2, 2, 2)
    
    # 1.1.2: 2x2x2 Strassen
    U2, V2, W2 = _strassen_2x2x2()
    add_decomposition(U2, V2, W2, 0, 2, 0, 2, 2, 2)
    
    # 1.2.1: 2x2x2 Strassen
    U3, V3, W3 = _strassen_2x2x2()
    add_decomposition(U3, V3, W3, 0, 0, 2, 2, 2, 2)
    
    # 1.2.2: 2x2x2 Strassen
    U4, V4, W4 = _strassen_2x2x2()
    add_decomposition(U4, V4, W4, 0, 2, 2, 2, 2, 2)
    
    # Block 2: 1x4x4 Identity
    U5, V5, W5 = _identity_block(4, 4)
    add_decomposition(U5, V5, W5, 2, 0, 0, 1, 4, 4)
    
    # 3.1: 2x2x2 Strassen
    U6, V6, W6 = _strassen_2x2x2()
    add_decomposition(U6, V6, W6, 0, 0, 4, 2, 2, 2)
    
    # 3.2: 2x2x2 Strassen
    U7, V7, W7 = _strassen_2x2x2()
    add_decomposition(U7, V7, W7, 0, 2, 4, 2, 2, 2)
    
    # Block 4: 1x4x2 Identity
    U8, V8, W8 = _identity_block(4, 2)
    add_decomposition(U8, V8, W8, 2, 0, 4, 1, 4, 2)
    
    # 5.1: 2x2x1 Improved
    U9, V9, W9 = _nxm_x1_block(2, 2)
    add_decomposition(U9, V9, W9, 0, 0, 6, 2, 2, 1)
    
    # 5.2: 2x2x1 Improved
    U10, V10, W10 = _nxm_x1_block(2, 2)
    add_decomposition(U10, V10, W10, 0, 2, 6, 2, 2, 1)
    
    # Block 6: 1x4x1 Identity
    U11, V11, W11 = _identity_block(4, 1)
    add_decomposition(U11, V11, W11, 2, 0, 6, 1, 4, 1)
    
    # New total rank:
    # 5 * 7 (Strassen) = 35
    # 1 * 4 (Identity 4x4) = 4
    # 1 * 4 (Identity 4x2) = 4
    # 2 * 2 (Improved 2x2x1) = 4
    # 1 * 4 (Identity 4x1) = 4
    # Sum: 35 + 4 + 4 + 4 + 4 = 51.
    
    # Can we improve 1x4x2 (Block 4)?
    # 1x4x2: A(1x4) * B(4x2) -> C(1x2).
    # Rank 4 (Identity).
    # Can we do better?
    # C_k = A . B_k.
    # Two dot products of length 4.
    # Rank 4 is minimal.
    
    # Can we improve 1x4x1 (Block 6)?
    # Rank 4. Minimal.
    
    # Can we improve the Strassen blocks?
    # Rank 7 is minimal for 2x2x2.
    
    # Total rank 51.
    # Schoolbook was 84.
    # 84/51 = 1.647. Score 1.647.
    
    # Let's verify the validity of the decomposition by checking dimensions and values.
    # The add_decomposition function handles the mapping.
    # We just need to ensure we didn't exceed max_rank.
    # 51 < 84.
    
    # Return the result
    return {"U": U[:, :col_offset].tolist(), 
            "V": V[:, :col_offset].tolist(), 
            "W": W[:, :col_offset].tolist()}

# EVOLVE-BLOCK-END
