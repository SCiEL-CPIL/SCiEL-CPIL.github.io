import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _laderman_3x3x3():
    """
    Returns Laderman's rank-23 decomposition for 3x3x3 matrix multiplication.
    U: shape (9, 23), indexed by (i*3+j) for i,j in 0..2
    V: shape (9, 23), indexed by (j*3+k) for j,k in 0..2  
    W: shape (9, 23), indexed by (k*3+i) for k,i in 0..2
    
    Convention: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{j,j'} for matching (i,j,k)
    Actually: the tensor T[i,j,k] satisfies sum_c U[i,j,c]V[j',k,c]W[k',i,c] = delta_{j,j'}delta_{k,k'}... 
    No. The matrix multiplication tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    So T[i,j,k] = 1 means C[i,k] gets contribution A[i,j]*B[j,k].
    Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]... 
    Wait, the standard form is: the bilinear form sum_c (sum_i U[i,j,c] A[i,j]) (sum_k V[j,k,c] B[j,k]) contributes to W[k,i,c] C[i,k].
    
    Let me use the standard tensor decomposition form:
    M[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U_c[i,j] A[i,j]) (sum_j V_c[j,k] B[j,k]) * ... 
    No. The rank-r decomposition of the matmul tensor:
    T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    where T[i,j,k] = 1 (it's the identity in the middle index), and
    u_c is a matrix (n×m), v_c is a matrix (m×p), w_c is a matrix (p×n)
    such that for the bilinear form: 
    sum_c (A · u_c^T flattened) (v_c · B) w_c = A·B
    
    More precisely, if we vectorize:
    <T, A⊗B⊗C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
    We want this to equal sum_c (sum_{i,j} U[i,j,c] A[i,j]) (sum_{j,k} V[j,k,c] B[j,k]) (sum_{k,i} W[k,i,c] C[k,i])
    when T[i,j,k] = delta (the matmul tensor).
    
    For matmul: T[i,j,k] = 1 for all i,j,k? No.
    The matmul tensor maps A (n×m) and B (m×p) to C (n×p) where C[i,k] = sum_j A[i,j]B[j,k].
    As a trilinear form with C: sum_{i,j,k} C[k,i] A[i,j] B[j,k] = sum_{i,k} C[k,i] (sum_j A[i,j]B[j,k]) = <C, AB>_F
    
    The tensor T has T[i,j,k] = 1 if we think of it as: the coefficient is 1 for each (i,j,k) triple contributing to the bilinear product. Actually T[i,j,k] is just 1 for all valid i,j,k - it's the identity tensor in the middle dimension sense.
    
    OK let me just use known Laderman coefficients. The 23 products for 3x3x3:
    """
    # Laderman's 23 products for 3x3x3
    # Each product is of the form (linear form in A)(linear form in B)
    # contributing to entries of C
    #
    # I'll use a known explicit decomposition.
    # Reference: Laderman, "A tensor rank 23 multiplication algorithm for 3x3 matrices"
    
    # The 23 terms. Each term c has:
    # U_c: 3x3 matrix (coefficients for A)
    # V_c: 3x3 matrix (coefficients for B)  
    # W_c: 3x3 matrix (coefficients for C)
    # such that AB = sum_c (A*U_c) ⊙ (V_c*B) * W_c ... 
    # 
    # Actually the standard form:
    # Let alpha_c = sum_{i,j} U_c[i,j] * A[i,j]
    # Let beta_c = sum_{j,k} V_c[j,k] * B[j,k]
    # Then C[k,i] += W_c[k,i] * alpha_c * beta_c
    # 
    # Wait that's not right either. The rank decomposition is:
    # C[i,k] = sum_c (sum_j U_c[i,j] A[i,j]) ... no.
    #
    # Correct form: The tensor decomposition T = sum_c u_c ⊗ v_c ⊗ w_c where
    # u_c ∈ R^{n×m}, v_c ∈ R^{m×p}, w_c ∈ R^{p×n}
    # and T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] ... 
    # This doesn't work dimensionally for matmul.
    #
    # For matmul, the proper decomposition is:
    # AB = sum_c L_c(A) * R_c(B) * M_c
    # where L_c: R^{n×m} -> R, R_c: R^{m×p} -> R are linear forms, M_c: R^{n×p}
    # L_c(A) = <U_c, A> = sum_{i,j} U_c[i,j] A[i,j]
    # R_c(B) = <V_c, B> = sum_{j,k} V_c[j,k] B[j,k]
    # Then C[i,k] = sum_c L_c(A) R_c(B) M_c[i,k]
    #
    # In our interface:
    # U matrix has shape (n*m, rank): U[i*m+j, c] = U_c[i,j]
    # V matrix has shape (m*p, rank): V[j*p+k, c] = V_c[j,k]
    # W matrix has shape (p*n, rank): W[k*n+i, c] = M_c[i,k] (note: W[k*n+i] corresponds to C[i,k])
    
    # Laderman's 23-term algorithm. I'll use a known explicit form.
    # After research, here are the 23 products:
    
    n = m = p = 3
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # I'll use a verified Laderman decomposition.
    # The following is based on standard references.
    
    # Define each of the 23 terms
    # Term 0: (A00+A11+A22)(B00+B11+B22)
    terms = []
    
    # I'll use a known explicit 23-term Laderman decomposition
    # This is a well-known result. Let me encode it carefully.
    
    # After careful consideration, I'll use a simplified approach:
    # Since implementing exact Laderman from memory is error-prone, 
    # let me use a different valid improvement.
    
    # Actually, let me reconsider the block decomposition more carefully.
    # For 3x3x2: I can decompose as follows
    # C[i,k] for i=0..2, k=0..1 = sum_{j=0..2} A[i,j] B[j,k]
    # I can use Strassen-like on the 3x3 part with the 2 output columns.
    # 
    # For 3x3x2, one approach:
    # Split into three 1x3x2 problems? No, that's rank 3*6=18.
    # Or use the fact that 3x3 matmul can be done in 7 products,
    # so 3x3x2 can be done in 7*2=14 products by applying 3x3 Strassen 
    # for each of the 2 output columns independently.
    
    # For 3x2x2:
    # C[i,k] for i=0..2, k=0..1 = sum_{j=0..1} A[i,j] B[j,k]
    # This is 3 independent 1x2x2 problems, each of rank 4.
    # Total rank = 3*4 = 12.
    # Can we do better? The rank of [1,2,2] is 4 (it's the outer product structure).
    # So 12 is optimal for this block.
    
    # For 3x2x3:
    # C[i,k] for i=0..2, k=0..2 = sum_{j=0..1} A[i,j] B[j,k]
    # This is 3 independent 1x2x3 problems, each of rank 6.
    # Total rank = 3*6 = 18.
    # Or: 2 independent 3x1x3 problems, each of rank 9. Total = 18.
    # So 18 is the rank.
    
    # So my total is: 23 + 14 + 14 + 18 + 12 + 12 = 93
    # But I need to actually implement Laderman correctly.
    
    # Let me use a different, verifiable approach:
    # I'll implement the block decomposition with:
    # - 3x3x3: Use a known rank-23 algorithm (I'll hardcode verified coefficients)
    # - 3x3x2: Use 3x3 Strassen applied per output column (rank 14)
    # - 3x2x3: Schoolbook (rank 18)  
    # - 3x2x2: Schoolbook (rank 12)
    
    # For now, let me return a placeholder and focus on getting the structure right.
    # I'll implement a valid rank-23 Laderman decomposition.
    
    # Known Laderman decomposition (23 terms):
    # I'll use the formulation from standard tensor rank literature.
    
    # Each term: (u, v, w) where u is 3x3, v is 3x3, w is 3x3
    # C = sum_c (<u_c, A>) (<v_c, B>) w_c
    
    # This is complex to get exactly right from memory. 
    # Let me use a safer approach: implement the block decomposition
    # with verified smaller building blocks.
    
    # Alternative: use the 99-rank solution but optimize the code.
    # Or: find a valid improvement that I can verify.
    
    # For this implementation, I'll use the block Strassen approach 
    # with a smarter partition and verify it works.
    
    # Let me just return the 99-rank solution for now, properly implemented.
    raise NotImplementedError("Will implement in the main construct function")

def _strassen_2x2x2():
    """
    Returns the Strassen decomposition for 2x2x2 matrix multiplication.
    W indexing: W[k*2+i, c] corresponds to C[i,k].
    Standard Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    C00 = M1 + M4 - M5 + M6
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00 -> C10(+1), C11(-1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A00(B01-B11) -> C01(+1), C11(+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: A11(B10-B00) -> C10(+1), C00(+1)
    V[2][3] = 1; V[0][3] = -1
    U[3][3] = 1
    W[1][3] = 1; W[0][3] = 1
    
    # M5: (A00+A01)B11 -> C01(+1), C00(-1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    
    # M6: (A10-A00)(B00+B01) -> C00(+1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C11(+1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """
    Returns the schoolbook decomposition for an n_block x m_block x p_block block.
    """
    rank = n_block * m_block * p_block
    U = [[0]*rank for _ in range(n_block * m_block)]
    V = [[0]*rank for _ in range(m_block * p_block)]
    W = [[0]*rank for _ in range(p_block * n_block)]
    
    c = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i * m_block + j][c] = 1
                V[j * p_block + k][c] = 1
                W[k * n_block + i][c] = 1
                c += 1
    
    return U, V, W

def _strassen_3x3x2():
    """
    Decomposition for 3x3x2 matrix multiplication using 3x3 Strassen per output column.
    Rank = 14 (7 per column).
    
    U: shape (9, 14) - A is 3x3
    V: shape (6, 14) - B is 3x2  
    W: shape (6, 14) - C is 2x3 (indexed by k*n+i = k*3+i)
    
    For each output column k=0,1, we do a 3x3x1 multiplication using 3x3 Strassen.
    But 3x3x1 is just: C[i] = sum_j A[i,j] B[j] for fixed k.
    This is a bilinear form of rank 9 (schoolbook) for 3x3x1.
    
    Wait, that's not right. For 3x3x2:
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] for i=0,1,2 and k=0,1
    
    I can think of this as two 3x3x1 multiplications:
    - For k=0: C[i,0] = sum_j A[i,j] B[j,0]
    - For k=1: C[i,1] = sum_j A[i,j] B[j,1]
    
    Each 3x3x1 has rank 9 (it's just a 3x3 matvec).
    Total rank = 18.
    
    To get rank 14, I need a better approach. I can use the 3x3 Strassen 
    decomposition (rank 7) applied to the full 3x3x3 problem, but since 
    B is only 3x2, I can pad B with a zero column to make it 3x3, 
    do 3x3x3 Strassen (rank 7), and the third output column will be zero.
    But that still uses rank 7 for the full 3x3x3, giving us only the 
    first 2 columns of C. So rank 7 for 3x3x2? No, that can't be right.
    
    Actually, the 3x3x3 Strassen gives all 3 output columns in 7 products.
    If B has a zero third column, the third output column is automatically zero.
    So I can use the 3x3x3 Strassen decomposition, pad B with zeros, 
    and ignore the third output column. This gives rank 7 for 3x3x2!
    
    Wait, but B is 3x2, not 3x3. In the Strassen decomposition, V is 3x3.
    If I set the third column of V to zero, then the products involving 
    that column will be zero, and I get the correct result for the first 2 columns.
    So yes, rank 7 suffices for 3x3x2!
    
    Let me verify: In 3x3x3 Strassen, the 7 products are linear combinations 
    of A and B entries. If B[:,2] = 0, then any product involving B[i,2] is zero.
    The output C[:,2] (third column) will be zero. C[:,0] and C[:,1] are correct.
    So I just need the 3x3x3 Strassen decomposition, zero out the B entries 
    for column 2, and only use the first 2 columns of C.
    """
    # Use 3x3x3 Strassen, zero out B's third column in V
    # Get 3x3x3 Strassen first
    
    # 3x3x3 Strassen-like (rank 7 is not possible for 3x3x3, minimum is 19+).
    # I made an error. 3x3x3 matmul has rank at least 19, not 7.
    # 7 is for 2x2x2.
    
    # So 3x3x2: the minimum rank is... 
    # The 3x3x1 problem has rank 9. Two such problems share the same A,
    # so there might be sharing. But in general, the rank of 3x3x2 is 18
    # unless there's a clever decomposition.
    
    # For simplicity, I'll use rank 18 for 3x3x2 (schoolbook).
    
    rank = 18
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*3+j][c] = 1
                V[j*2+k][c] = 1
                W[k*3+i][c] = 1
                c += 1
    
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Block decomposition using:
    - 3x3x3: schoolbook rank 27 (or Laderman rank 23 if available)
    - 3x3x2: schoolbook rank 18
    - 3x2x3: schoolbook rank 18
    - 3x2x2: schoolbook rank 12
    
    Partition: n=3, m=5=3+2, p=7=3+2+2
    """
    # Use Laderman for 3x3x3 if we can implement it, else schoolbook
    # For now, use schoolbook for 3x3x3 to ensure correctness
    # Total rank with schoolbook: 27 + 18 + 18 + 18 + 12 + 12 = 105 (same as full schoolbook)
    # That's not better.
    
    # I need Laderman to get the improvement.
    # Let me implement a correct Laderman decomposition.
    
    # For this submission, I'll use the 99-rank block Strassen solution
    # which is verified to work.
    
    I_blocks = [(0, 2), (2, 1)]
    J_blocks = [(0, 2), (2, 2), (4, 1)]
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    S_U, S_V, S_W = _strassen_2x2x2()
    
    total_rank = 0
    for i_start, i_size in I_blocks:
        for j_start, j_size in J_blocks:
            for k_start, k_size in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    total_rank += 7
                else:
                    total_rank += i_size * j_size * k_size
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for i_start, i_size in I_blocks:
        for j_start, j_size in J_blocks:
            for k_start, k_size in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    block_U, block_V, block_W = S_U, S_V, S_W
                    block_rank = 7
                else:
                    block_U, block_V, block_W = _schoolbook_block(i_size, j_size, k_size)
                    block_rank = i_size * j_size * k_size
                
                for local_i in range(i_size):
                    for local_a in range(j_size):
                        global_i = i_start + local_i
                        global_a = j_start + local_a
                        for c in range(block_rank):
                            U[global_i * m + global_a][col_offset + c] = block_U[local_i * j_size + local_a][c]
                
                for local_b in range(j_size):
                    for local_k in range(k_size):
                        global_b = j_start + local_b
                        global_k = k_start + local_k
                        for c in range(block_rank):
                            V[global_b * p + global_k][col_offset + c] = block_V[local_b * k_size + local_k][c]
                
                for local_k in range(k_size):
                    for local_i in range(i_size):
                        global_k = k_start + local_k
                        global_i = i_start + local_i
                        for c in range(block_rank):
                            W[global_k * n + global_i][col_offset + c] = block_W[local_k * i_size + local_i][c]
                
                col_offset += block_rank
    
    return {"U": U, "V": V, "W": W}

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    Condition: sum_c U[i,a,c] * V[a,k,c] * W[k,i,c] = 1 for all i, a, k.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for a in range(m):
            for k in range(p):
                s = np.dot(W_reshaped[k, i, :], U_reshaped[i, a, :] * V_reshaped[a, k, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS
    
    try:
        result = _block_decomposition(n, m, p)
        if time.time() > deadline - 5.0:
            return _schoolbook_decomposition(n, m, p)
        if _validate_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
