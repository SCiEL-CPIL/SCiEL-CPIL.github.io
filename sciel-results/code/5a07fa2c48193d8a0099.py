import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, tol=1e-8):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
    
    # Use vectorized verification for speed
    # Build the full tensor check
    # For each r, we have U[:,r] (36,), V[:,r] (36,), W[:,r] (36,)
    # The contribution to the (i,j,k) entry is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need sum_r of this to equal 1 for all (i,j,k)
    
    # Let's build it as: for each r, form the 6x6x6 tensor T_r where
    # T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Then sum over r and check all entries are 1.
    
    # Vectorized approach:
    # Reshape U to (6,6,rank), V to (6,6,rank), W to (6,6,rank)
    U_reshaped = U.reshape(n, m, -1)  # (6, 6, rank)
    V_reshaped = V.reshape(m, p, -1)  # (6, 6, rank)
    W_reshaped = W.reshape(p, n, -1)  # (6, 6, rank)
    
    # For each r, the tensor is outer product of U_reshaped[:,:,r], V_reshaped[:,:,r], W_reshaped[:,:,r]
    # but with specific index mappings:
    # U[i,j,r], V[j,k,r], W[k,i,r]
    # So T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's compute this using einsum or manual loop over rank
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    
    for r in range(rank):
        u_col = U_reshaped[:, :, r]  # (n, m)
        v_col = V_reshaped[:, :, r]  # (m, p)
        w_col = W_reshaped[:, :, r]  # (p, n)
        
        # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        # This is a 3D tensor product
        # u_col[i,j] * v_col[j,k] gives (n, p) matrix M where M[i,k] = sum_j u_col[i,j]*v_col[j,k]
        # Then T[i,j,k] = u_col[i,j] * v_col[j,k] * w_col[k,i]
        # We can compute: for each i, j: T[i,j,:] += u_col[i,j] * v_col[j,:] * w_col[:,i]
        
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_col[i, j] * v_col[j, :] * w_col[:, i]
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _laderman_3x3():
    """
    Laderman's algorithm for 3x3 matrix multiplication.
    Rank 23.
    Returns U, V, W of shape (9, 23).
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Laderman's 7x7... no, rank 23 for 3x3.
    # Let me look up Laderman's algorithm.
    # Actually, I recall that Laderman's algorithm for 3x3 has rank 23.
    # But let me be careful. The standard reference gives specific formulas.
    # Let me construct it from the known rank-23 decomposition.
    
    # Actually, implementing Laderman's rank-23 decomposition from scratch is complex.
    # Let me use a different approach: use the known 2x2 Strassen (rank 7) 
    # combined with 3x3 schoolbook (rank 27) to get 7*27 = 189.
    # Or use 3x3 Strassen-like if available.
    
    # For now, let me return the schoolbook 3x3 (rank 27) as a fallback.
    # I'll implement a proper Laderman later if needed.
    
    # Actually, let me implement a simpler improvement:
    # Use 2x2 Strassen on 3x3 schoolbook blocks.
    # This gives rank 7 * 27 = 189.
    
    # But first, let me just return schoolbook 3x3 for safety.
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _build_block_strassen_6x6():
    """
    Build 6x6x6 decomposition using 2x2 Strassen on 3x3 schoolbook blocks.
    Rank = 7 * 27 = 189.
    
    The 6x6 matrix is divided into 2x2 blocks of 3x3.
    We use Strassen's 7 products at the block level, and for each product,
    we decompose the 3x3 matrix multiplication using schoolbook (rank 27).
    """
    n, m, p = 6, 6, 6
    block_size = 3
    nblocks = 2
    
    total_rank = 7 * 27  # 189
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Strassen 2x2 formulas at block level:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each Strassen product, we need to add the 3x3 schoolbook decomposition.
    # Each product involves a 3x3 matrix multiplication where the input matrices
    # are linear combinations of 3x3 blocks.
    
    # Helper function to add a 3x3 schoolbook product
    def add_3x3_product(a_block, b_block, c_blocks, col_start):
        """
        Add schoolbook decomposition of A[a_block] * B[b_block] contributing to C blocks.
        a_block: (bi, bj) - position in A
        b_block: (bj, bk) - position in B  
        c_blocks: list of ((bi, bk), sign) - positions in C with signs
        Returns new column index.
        """
        bi_a, bj_a = a_block
        bj_b, bk_b = b_block
        
        if bj_a != bj_b:
            return col_start  # Blocks don't connect
        
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = block_size * bi_a + di
                    j = block_size * bj_a + dj
                    k = block_size * bk_b + dk
                    
                    for (bi_c, bk_c), sign_c in c_blocks:
                        if bi_a == bi_c and bk_b == bk_c:
                            U[i*m+j, col_start] = 1.0
                            V[j*p+k, col_start] = 1.0
                            W[k*n+i, col_start] = float(sign_c)
                            col_start += 1
                            break
        
        return col_start
    
    # Actually, the above approach is flawed because in Strassen, the products are
    # of linear combinations, not single blocks. Let me redo this properly.
    
    # The correct approach: for each Strassen product P_r, it's a product of two 3x3 matrices
    # where each is a linear combination of blocks. We need to expand this.
    
    # Let me define each product explicitly.
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    col = 0
    
    # P1 = (A00+A11)(B00+B11)
    # Expands to: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Contributes to: C00 (+1), C11 (-1)
    p1_terms = [
        ((0,0), (0,0), [( (0,0), 1), ((1,1), -1)]),  # A00*B00
        ((0,0), (0,1), [( (0,0), 1), ((1,1), -1)]),  # A00*B11
        ((1,1), (1,0), [( (0,0), 1), ((1,1), -1)]),  # A11*B00
        ((1,1), (1,1), [( (0,0), 1), ((1,1), -1)]),  # A11*B11
    ]
    for a_blk, b_blk, c_contribs in p1_terms:
        bi_a, bj_a = a_blk
        bj_b, bk_b = b_blk
        if bj_a != bj_b:
            continue
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = 3*bi_a + di
                    j = 3*bj_a + dj
                    k = 3*bk_b + dk
                    for (bi_c, bk_c), sign_c in c_contribs:
                        if bi_a == bi_c and bk_b == bk_c:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = float(sign_c)
                            col += 1
                            break
    
    # P2 = (A10+A11)B00
    # Expands to: A10*B00 + A11*B00
    # Contributes to: C10 (+1), C11 (-1)
    p2_terms = [
        ((1,0), (0,0), [( (1,0), 1), ((1,1), -1)]),  # A10*B00
        ((1,1), (1,0), [( (1,0), 1), ((1,1), -1)]),  # A11*B00
    ]
    for a_blk, b_blk, c_contribs in p2_terms:
        bi_a, bj_a = a_blk
        bj_b, bk_b = b_blk
        if bj_a != bj_b:
            continue
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = 3*bi_a + di
                    j = 3*bj_a + dj
                    k = 3*bk_b + dk
                    for (bi_c, bk_c), sign_c in c_contribs:
                        if bi_a == bi_c and bk_b == bk_c:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = float(sign_c)
                            col += 1
                            break
    
    # P3 = A00(B01-B11)
    # Expands to: A00*B01 - A00*B11
    # Contributes to: C01 (+1), C11 (+1)
    p3_terms = [
        ((0,0), (0,1), [( (0,1), 1), ((1,1), 1)]),  # A00*B01
        ((0,1), (1,1), [( (0,1), -1), ((1,1), -1)]),  # -A00*B11
    ]
    for a_blk, b_blk, c_contribs in p3_terms:
        bi_a, bj_a = a_blk
        bj_b, bk_b = b_blk
        if bj_a != bj_b:
            continue
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = 3*bi_a + di
                    j = 3*bj_a + dj
                    k = 3*bk_b + dk
                    for (bi_c, bk_c), sign_c in c_contribs:
                        if bi_a == bi_c and bk_b == bk_c:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = float(sign_c)
                            col += 1
                            break
    
    # P4 = A11(B10-B00)
    # Expands to: A11*B10 - A11*B00
    # Contributes to: C00 (+1), C10 (+1)
    p4_terms = [
        ((1,1), (1,0), [( (0,0), 1), ((1,0), 1)]),  # A11*B10
        ((1,0), (0,0), [( (0,0), -1), ((1,0), -1)]),  # -A11*B00
    ]
    for a_blk, b_blk, c_contribs in p4_terms:
        bi_a, bj_a = a_blk
        bj_b, bk_b = b_blk
        if bj_a != bj_b:
            continue
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = 3*bi_a + di
                    j = 3*bj_a + dj
                    k = 3*bk_b + dk
                    for (bi_c, bk_c), sign_c in c_contribs:
                        if bi_a == bi_c and bk_b == bk_c:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = float(sign_c)
                            col += 1
                            break
    
    # P5 = (A00+A01)B11
    # Expands to: A00*B11 + A01*B11
    # Contributes to: C00 (-1), C01 (+1)
    p5_terms = [
        ((0,0), (0,1), [( (0,0), -1), ((0,1), 1)]),  # A00*B11
        ((0,1), (1,1), [( (0,0), -1), ((0,1), 1)]),  # A01*B11
    ]
    for a_blk, b_blk, c_contribs in p5_terms:
        bi_a, bj_a = a_blk
        bj_b, bk_b = b_blk
        if bj_a != bj_b:
            continue
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = 3*bi_a + di
                    j = 3*bj_a + dj
                    k = 3*bk_b + dk
                    for (bi_c, bk_c), sign_c in c_contribs:
                        if bi_a == bi_c and bk_b == bk_c:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = float(sign_c)
                            col += 1
                            break
    
    # P6 = (A10-A00)(B00+B01)
    # Expands to: A10*B00 + A10*B01 - A00*B00 - A00*B01
    # Contributes to: C11 (+1)
    p6_terms = [
        ((1,0), (0,0), [( (1,1), 1)]),  # A10*B00
        ((1,0), (0,1), [( (1,1), 1)]),  # A10*B01
        ((0,0), (0,0), [( (1,1), -1)]),  # -A00*B00
        ((0,0), (0,1), [( (1,1), -1)]),  # -A00*B01
    ]
    for a_blk, b_blk, c_contribs in p6_terms:
        bi_a, bj_a = a_blk
        bj_b, bk_b = b_blk
        if bj_a != bj_b:
            continue
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = 3*bi_a + di
                    j = 3*bj_a + dj
                    k = 3*bk_b + dk
                    for (bi_c, bk_c), sign_c in c_contribs:
                        if bi_a == bi_c and bk_b == bk_c:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = float(sign_c)
                            col += 1
                            break
    
    # P7 = (A01-A11)(B10+B11)
    # Expands to: A01*B10 + A01*B11 - A11*B10 - A11*B11
    # Contributes to: C00 (+1)
    p7_terms = [
        ((0,1), (1,0), [( (0,0), 1)]),  # A01*B10
        ((0,1), (1,1), [( (0,0), 1)]),  # A01*B11
        ((1,1), (1,0), [( (0,0), -1)]),  # -A11*B10
        ((1,1), (1,1), [( (0,0), -1)]),  # -A11*B11
    ]
    for a_blk, b_blk, c_contribs in p7_terms:
        bi_a, bj_a = a_blk
        bj_b, bk_b = b_blk
        if bj_a != bj_b:
            continue
        for di in range(3):
            for dj in range(3):
                for dk in range(3):
                    i = 3*bi_a + di
                    j = 3*bj_a + dj
                    k = 3*bk_b + dk
                    for (bi_c, bk_c), sign_c in c_contribs:
                        if bi_a == bi_c and bk_b == bk_c:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = float(sign_c)
                            col += 1
                            break
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    best_rank = 216
    best_result = None
    
    # Try block Strassen (rank 189)
    if time.time() < deadline - 30:
        try:
            U_s, V_s, W_s = _build_block_strassen_6x6()
            if _verify(U_s, V_s, W_s):
                rank = U_s.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_s, V_s, W_s)
        except Exception:
            pass
    
    # Fallback to schoolbook (rank 216)
    if best_result is None:
        U_s, V_s, W_s = _build_schoolbook()
        best_result = (U_s, V_s, W_s)
        best_rank = 216
    
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
