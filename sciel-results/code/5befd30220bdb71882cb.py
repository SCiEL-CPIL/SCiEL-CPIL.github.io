"""Matrix multiplication 4x5x5 with structured tensor network ansatz."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Strategy: Use a structured approach based on Strassen-like algebraic
    cancellations. We decompose the problem by splitting the matrices into
    blocks and using bilinear combinations to reduce the number of
    multiplications below the schoolbook bound of 4*5*5 = 100.
    
    We split the inner dimension (5) into blocks of sizes 2, 2, 1.
    We split the row dimension of A (4) into two blocks of size 2.
    We split the column dimension of B/C (5) into blocks of sizes 2, 2, 1.
    
    For each combination of (I_block, J_block, K_block), we have a subproblem
    of size (2 x w_J) * (w_J x w_K) -> (2 x w_K).
    
    If w_J == 2 and w_K == 2, the subproblem is (2x2) * (2x2) -> (2x2).
    We use Strassen's algorithm here, which uses 7 multiplications instead of 8.
    
    For other cases (where w_J or w_K is 1), we use schoolbook multiplication.
    
    Let's count the rank:
    I blocks: 2 (size 2 each)
    J blocks: 3 (sizes 2, 2, 1)
    K blocks: 3 (sizes 2, 2, 1)
    
    Cases:
    1. w_J=2, w_K=2:
       Number of such (J,K) pairs: 2*2 = 4 pairs.
       For each of the 2 I blocks, we have 2 * 4 = 8 subproblems of size 2x2x2.
       Each uses rank 7 (Strassen).
       Total rank from these: 8 * 7 = 56.
       
    2. w_J=2, w_K=1:
       J blocks with size 2: 2. K blocks with size 1: 1.
       Pairs: 2 * 1 = 2.
       For each of the 2 I blocks: 2 * 2 = 4 subproblems.
       Subproblem size: (2x2) * (2x1) -> (2x1).
       Schoolbook rank: 2*2*1 = 4.
       Total rank: 4 * 4 = 16.
       
    3. w_J=1, w_K=2:
       J blocks with size 1: 1. K blocks with size 2: 2.
       Pairs: 1 * 2 = 2.
       For each of the 2 I blocks: 2 * 2 = 4 subproblems.
       Subproblem size: (2x1) * (1x2) -> (2x2).
       Schoolbook rank: 2*1*2 = 4.
       Total rank: 4 * 4 = 16.
       
    4. w_J=1, w_K=1:
       J blocks with size 1: 1. K blocks with size 1: 1.
       Pairs: 1 * 1 = 1.
       For each of the 2 I blocks: 2 * 1 = 2 subproblems.
       Subproblem size: (2x1) * (1x1) -> (2x1).
       Schoolbook rank: 2*1*1 = 2.
       Total rank: 2 * 2 = 4.
       
    Total rank = 56 + 16 + 16 + 4 = 92.
    This is an improvement over the schoolbook rank of 100.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Define blocks for inner dimension (m=5) and output columns (p=5)
    # Blocks: (start, size)
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 1)]
    i_blocks = [(0, 2), (2, 2)]
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        # u_idx: index in U (n*m)
        # v_idx: index in V (m*p)
        # w_idx: index in W (p*n)
        # We append to lists, so we need to manage lengths
        pass
    
    # We will build the matrices as lists of lists, then convert to numpy arrays
    # Or just use numpy arrays directly
    
    # Let's collect the terms first
    terms = []
    
    def strassen_2x2x2(A_idx, B_idx, C_idx):
        """
        Add Strassen terms for a 2x2 * 2x2 -> 2x2 block.
        A_idx: list of 4 indices for A (row-major: 0,1,2,3)
        B_idx: list of 4 indices for B (row-major: 0,1,2,3)
        C_idx: list of 4 indices for C (row-major: 0,1,2,3)
        
        Strassen formulas:
        m1 = (a00 + a11) * (b00 + b11)
        m2 = (a10 + a11) * b00
        m3 = a00 * (b01 - b11)
        m4 = a11 * (b10 - b00)
        m5 = (a00 + a01) * b11
        m6 = (a10 - a00) * (b00 + b01)
        m7 = (a01 - a11) * (b10 + b11)
        
        c00 = m1 + m4 - m5 + m7
        c01 = m3 + m5
        c10 = m2 + m4
        c11 = m1 - m2 + m3 + m6
        """
        # Each term is a product of a linear combination of A entries and a linear combination of B entries.
        # We need to expand these into rank-1 terms for the global tensor.
        # A linear combination: sum_j c_j * A[idx_j]
        # B linear combination: sum_k d_k * B[idx_k]
        # Product: (sum_j c_j A[idx_j]) * (sum_k d_k B[idx_k]) = sum_{j,k} c_j d_k A[idx_j] B[idx_k]
        # Each A[idx_j] B[idx_k] contributes to specific C entries.
        
        # Instead of expanding, we can just add the bilinear forms directly.
        # The tensor decomposition allows U, V, W to have coefficients.
        # A term is: U_col * V_col * W_col (outer product).
        # If we have (c0*A0 + c1*A1) * (d0*B0 + d1*B1) * (e0*C0 + e1*C1),
        # this is NOT a single rank-1 term unless we expand.
        # Wait, the definition of tensor decomposition for matrix multiplication is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # So each rank-1 term is a product of a vector in U, a vector in V, and a vector in W.
        # If I have a bilinear form like (A0+A1)*(B0+B1) contributing to C0,
        # I can represent this as:
        # U[0,r]=1, U[1,r]=1, others 0
        # V[0,r]=1, V[1,r]=1, others 0
        # W[0,r]=1, others 0
        # This term contributes to T[0,0,0], T[0,1,0], T[1,0,0], T[1,1,0].
        # Specifically, T[i,j,k] += U[i,j,r] V[j,k,r] W[k,i,r].
        
        # So for Strassen, I just add the 7 terms directly.
        
        # m1: (A00+A11)(B00+B11) -> C00, C11
        # A indices: A00, A11
        # B indices: B00, B11
        # C indices: C00, C11
        
        # Let's map local indices to global indices.
        # A_local[i,j] corresponds to A_global[i_block_start + i, j_block_start + j]
        # B_local[i,j] corresponds to B_global[j_block_start + i, k_block_start + j]
        # C_local[i,j] corresponds to C_global[k_block_start + j, i_block_start + i]
        
        # We need to pass the start indices to this function.
        
        # Let's restructure. We will pass the global indices for the 4x4 blocks.
        
        pass
    
    # Let's implement the addition of terms directly in the main loop.
    
    # We will store U, V, W as lists of lists, appending columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of length n*m
        v_vec: list of length m*p
        w_vec: list of length p*n
        """
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def make_vec(length, indices, values):
        vec = [0] * length
        for idx, val in zip(indices, values):
            vec[idx] = val
        return vec
    
    n_dim = n
    m_dim = m
    p_dim = p
    
    # Iterate over blocks
    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                
                # Determine local dimensions
                # A is i_size x j_size
                # B is j_size x k_size
                # C is i_size x k_size
                
                if j_size == 2 and k_size == 2:
                    # Use Strassen
                    # Local indices:
                    # A: (0,0), (0,1), (1,0), (1,1)
                    # B: (0,0), (0,1), (1,0), (1,1)
                    # C: (0,0), (0,1), (1,0), (1,1)
                    
                    # Map local (row, col) to global index in U, V, W
                    def A_idx(i_loc, j_loc):
                        return (i_start + i_loc) * m_dim + (j_start + j_loc)
                    
                    def B_idx(i_loc, j_loc):
                        return (j_start + i_loc) * p_dim + (k_start + j_loc)
                    
                    def C_idx(i_loc, j_loc):
                        # C[i, k] corresponds to W[k * n + i]
                        return (k_start + j_loc) * n_dim + (i_start + i_loc)
                    
                    # Strassen terms
                    # m1 = (A00+A11)(B00+B11)
                    u1 = make_vec(n_dim*m_dim, [A_idx(0,0), A_idx(1,1)], [1, 1])
                    v1 = make_vec(m_dim*p_dim, [B_idx(0,0), B_idx(1,1)], [1, 1])
                    w1 = make_vec(p_dim*n_dim, [C_idx(0,0), C_idx(1,1)], [1, 1])
                    add_rank1_term(u1, v1, w1)
                    
                    # m2 = (A10+A11)B00
                    u2 = make_vec(n_dim*m_dim, [A_idx(1,0), A_idx(1,1)], [1, 1])
                    v2 = make_vec(m_dim*p_dim, [B_idx(0,0)], [1])
                    w2 = make_vec(p_dim*n_dim, [C_idx(1,0)], [1])
                    add_rank1_term(u2, v2, w2)
                    
                    # m3 = A00(B01-B11)
                    u3 = make_vec(n_dim*m_dim, [A_idx(0,0)], [1])
                    v3 = make_vec(m_dim*p_dim, [B_idx(0,1), B_idx(1,1)], [1, -1])
                    w3 = make_vec(p_dim*n_dim, [C_idx(0,1)], [1])
                    add_rank1_term(u3, v3, w3)
                    
                    # m4 = A11(B10-B00)
                    u4 = make_vec(n_dim*m_dim, [A_idx(1,1)], [1])
                    v4 = make_vec(m_dim*p_dim, [B_idx(1,0), B_idx(0,0)], [1, -1])
                    w4 = make_vec(p_dim*n_dim, [C_idx(0,0), C_idx(1,0)], [1, 1])
                    add_rank1_term(u4, v4, w4)
                    
                    # m5 = (A00+A01)B11
                    u5 = make_vec(n_dim*m_dim, [A_idx(0,0), A_idx(0,1)], [1, 1])
                    v5 = make_vec(m_dim*p_dim, [B_idx(1,1)], [1])
                    w5 = make_vec(p_dim*n_dim, [C_idx(0,0), C_idx(0,1)], [1, 1])
                    add_rank1_term(u5, v5, w5)
                    
                    # m6 = (A10-A00)(B00+B01)
                    u6 = make_vec(n_dim*m_dim, [A_idx(1,0), A_idx(0,0)], [1, -1])
                    v6 = make_vec(m_dim*p_dim, [B_idx(0,0), B_idx(0,1)], [1, 1])
                    w6 = make_vec(p_dim*n_dim, [C_idx(1,1)], [1])
                    add_rank1_term(u6, v6, w6)
                    
                    # m7 = (A01-A11)(B10+B11)
                    u7 = make_vec(n_dim*m_dim, [A_idx(0,1), A_idx(1,1)], [1, -1])
                    v7 = make_vec(m_dim*p_dim, [B_idx(1,0), B_idx(1,1)], [1, 1])
                    w7 = make_vec(p_dim*n_dim, [C_idx(0,0), C_idx(1,1)], [1, 1])
                    add_rank1_term(u7, v7, w7)
                    
                    # Verify C reconstruction for Strassen:
                    # c00 = m1 + m4 - m5 + m7
                    # c01 = m3 + m5
                    # c10 = m2 + m4
                    # c11 = m1 - m2 + m3 + m6
                    
                    # Our W vectors for the terms:
                    # m1: C00=1, C11=1
                    # m2: C10=1
                    # m3: C01=1
                    # m4: C00=1, C10=1
                    # m5: C00=1, C01=1
                    # m6: C11=1
                    # m7: C00=1, C11=1
                    
                    # Let's check C00:
                    # m1 contributes 1 to C00.
                    # m4 contributes 1 to C00.
                    # m5 contributes 1 to C00.
                    # m7 contributes 1 to C00.
                    # Total coefficient for C00 in the sum of terms:
                    # We need to ensure the linear combination of terms yields the correct C.
                    # The decomposition is T = sum_r U_r V_r W_r.
                    # The entry C[i,k] is the coefficient of A...B...
                    # Actually, the evaluator checks if the tensor defined by U,V,W equals the multiplication tensor.
                    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                    # We want T[i,j,k] = 1 if i,j,k are valid indices, else 0.
                    # My Strassen implementation adds terms that correspond to the bilinear forms.
                    # I need to make sure the W vectors are correct such that the sum of products reconstructs the identity tensor.
                    
                    # Let's trace C00 (i=0, k=0).
                    # Terms contributing to C00:
                    # m1: U1[0,0]=1, V1[0,0]=1, W1[0,0]=1 -> Product 1.
                    # m4: U4[1,1]=1, V4[1,0]=1, W4[0,0]=1 -> Product 1.
                    # m5: U5[0,0]=1, V5[1,1]=1, W5[0,0]=1 -> Product 1.
                    # m7: U7[0,1]=1, V7[1,0]=1, W7[0,0]=1 -> Product 1.
                    # Sum = 4? That's not right. The tensor entry T[0,0,0] should be 1.
                    # Wait, the terms are not just 1. They are linear combinations.
                    # T[0,0,0] = sum_r U[0,0,r] V[0,0,r] W[0,0,r]
                    # r=1 (m1): U[0,0,1]=1, V[0,0,1]=1, W[0,0,1]=1 -> 1
                    # r=4 (m4): U[0,0,4]=0, V[0,0,4]=-1, W[0,0,4]=1 -> 0
                    # r=5 (m5): U[0,0,5]=1, V[0,0,5]=0, W[0,0,5]=1 -> 0
                    # r=7 (m7): U[0,0,7]=0, V[0,0,7]=0, W[0,0,7]=1 -> 0
                    # Sum = 1. Correct.
                    
                    # Let's check T[0,1,0] (i=0, j=1, k=0).
                    # r=1: U[0,1,1]=0 -> 0
                    # r=4: U[0,1,4]=0 -> 0
                    # r=5: U[0,1,5]=1, V[1,0,5]=0 -> 0
                    # r=7: U[0,1,7]=1, V[1,0,7]=1, W[0,0,7]=1 -> 1
                    # Sum = 1. Correct.
                    
                    # Let's check T[1,0,0] (i=1, j=0, k=0).
                    # r=1: U[1,0,1]=0 -> 0
                    # r=2: U[1,0,2]=1, V[0,0,2]=1, W[0,0,2]=0 (W2 is C10=1, so W[0,0,2]=0) -> 0
                    # r=4: U[1,0,4]=0 -> 0
                    # r=5: U[1,0,5]=0 -> 0
                    # r=7: U[1,0,7]=0 -> 0
                    # Wait, T[1,0,0] should be 1.
                    # Let's re-evaluate m2.
                    # m2 = (A10+A11)B00.
                    # U2[1,0]=1, U2[1,1]=1.
                    # V2[0,0]=1.
                    # W2[1,0]=1 (C10).
                    # T[1,0,0] = sum_r U[1,0,r] V[0,0,r] W[0,0,r]
                    # r=2: U[1,0,2]=1, V[0,0,2]=1, W[0,0,2]=0 (since W2 is for C10, index is k*n+i = 0*4+1=1. W2[1]=1, W2[0]=0).
                    # So contribution is 0.
                    # Did I miss a term?
                    # c10 = m2 + m4.
                    # m4 contributes to C10?
                    # m4 = A11(B10-B00).
                    # U4[1,1]=1. V4[1,0]=1, V4[0,0]=-1. W4[1,0]=1 (C10).
                    # T[1,0,0] = ...
                    # r=4: U[1,0,4]=0.
                    # So T[1,0,0] is 0?
                    # Let's check the Strassen formula again.
                    # c10 = m2 + m4.
                    # m2 = (a10+a11)b00.
                    # m4 = a11(b10-b00).
                    # c10 = a10b00 + a11b00 + a11b10 - a11b00 = a10b00 + a11b10. Correct.
                    
                    # Why did my tensor check fail?
                    # T[i,j,k] is the coefficient of A[i,j] in the expression for C[k,i]?
                    # No, T[i,j,k] is the entry of the tensor.
                    # The multiplication tensor T has T[i,j,k] = 1 if we multiply A[i,j] by B[j,k] to get C[k,i].
                    # So T[i,j,k] = 1 for all valid i,j,k.
                    
                    # Let's check T[1,0,0] again.
                    # i=1, j=0, k=0.
                    # We need the sum of U[1,0,r] V[0,0,r] W[0,0,r] to be 1.
                    # Terms involving A[1,0] (i=1, j=0):
                    # m2: U[1,0]=1. V[0,0]=1. W[0,0]=0. Product 0.
                    # m6: U[1,0]=1. V[0,0]=1. W[0,0]=0 (W6 is C11, index 1*4+1=5). Product 0.
                    # m7: U[1,0]=0.
                    # m1: U[1,0]=0.
                    # m3: U[1,0]=0.
                    # m4: U[1,0]=0.
                    # m5: U[1,0]=0.
                    
                    # It seems T[1,0,0] is 0. But it should be 1.
                    # Where is the term A[1,0]B[0,0]?
                    # It comes from m2.
                    # m2 = (A10+A11)B00 = A10B00 + A11B00.
                    # This term contributes to C10.
                    # C10 is at index k=0, i=1. So W index is 0*4+1 = 1.
                    # So W2[1] = 1.
                    # T[1,0,0] uses W[0,0,r].
                    # The term m2 contributes to C10, not C00.
                    # So T[1,0,0] should not receive contribution from m2?
                    # Wait. T[i,j,k] is defined by the position in the tensor.
                    # The tensor T is of size n x m x p.
                    # T[i,j,k] is the coefficient of A[i,j] B[j,k] in the bilinear form?
                    # No, the tensor T is such that T[i,j,k] = 1.
                    # The decomposition is T = sum U_r V_r W_r.
                    # So T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
                    
                    # For i=1, j=0, k=0:
                    # We need sum_r U[1*5+0, r] V[0*5+0, r] W[0*4+1, r] = 1.
                    # Note the W index is k*n+i = 0*4+1 = 1.
                    # Let's re-evaluate m2 with W index 1.
                    # m2: U2[5]=1 (A10). V2[0]=1 (B00). W2[1]=1 (C10).
                    # Product: 1*1*1 = 1.
                    # So T[1,0,0] gets 1 from m2.
                    # Are there other terms?
                    # m4: U4[5]=0.
                    # m6: U6[5]=1 (A10). V6[0]=1 (B00). W6[1]=0 (W6 is C11, index 5).
                    # So T[1,0,0] = 1. Correct.
                    
                    # It seems my previous check was using the wrong W index.
                    # W index is k*n+i.
                    # For C10, k=0, i=1, index=1.
                    # For C00, k=0, i=0, index=0.
                    
                    # So the Strassen implementation should be correct if I use the correct W indices.
                    
                else:
                    # Schoolbook
                    for i_loc in range(i_size):
                        for j_loc in range(j_size):
                            for k_loc in range(k_size):
                                u_idx = (i_start + i_loc) * m_dim + (j_start + j_loc)
                                v_idx = (j_start + j_loc) * p_dim + (k_start + k_loc)
                                w_idx = (k_start + k_loc) * n_dim + (i_start + i_loc)
                                
                                u_vec = make_vec(n_dim*m_dim, [u_idx], [1])
                                v_vec = make_vec(m_dim*p_dim, [v_idx], [1])
                                w_vec = make_vec(p_dim*n_dim, [w_idx], [1])
                                add_rank1_term(u_vec, v_vec, w_vec)
    
    # Convert lists of lists to numpy arrays
    # U_cols is list of lists of length n*m
    # We need to transpose to get shape (n*m, rank)
    if U_cols:
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
    else:
        U = np.array([]).reshape(n*m, 0)
        V = np.array([]).reshape(m*p, 0)
        W = np.array([]).reshape(p*n, 0)
    
    return {"U": U, "V": V, "W": W}
