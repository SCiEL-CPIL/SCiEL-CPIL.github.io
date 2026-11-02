import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _strassen_222():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) such that
    sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # Index mapping: A[i*2+j], B[j*2+k], C[k*2+i]
    # A[0]=a, A[1]=b, A[2]=c, A[3]=d
    # B[0]=e, B[1]=f, B[2]=g, B[3]=h
    # C[0]=C[0,0], C[1]=C[0,1], C[2]=C[1,0], C[3]=C[1,1]
    
    # U[i*2+j, r] for r=0..6
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h)
    # U: a=1, d=1; V: e=1, h=1; W: contributes to C[0,0] and C[1,1]
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C[0,0] (k=0,i=0)
    W[3, 0] = 1  # C[1,1] (k=1,i=1)
    
    # M2 = (c+d)e
    # U: c=1, d=1; V: e=1; W: contributes to C[1,0]
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C[1,0] (k=0,i=1)
    
    # M3 = a(f-h)
    # U: a=1; V: f=1, h=-1; W: contributes to C[0,1]
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1  # h
    W[1, 2] = 1  # C[0,1] (k=1,i=0)
    
    # M4 = d(g-e)
    # U: d=1; V: g=1, e=-1; W: contributes to C[0,0] and C[1,0]
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1  # e
    W[0, 3] = 1  # C[0,0] (k=0,i=0)
    W[2, 3] = 1  # C[1,0] (k=0,i=1)
    
    # M5 = (a+b)h
    # U: a=1, b=1; V: h=1; W: contributes to C[0,0] (negative) and C[0,1]
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    V[3, 4] = 1  # h
    W[0, 4] = -1  # C[0,0] (k=0,i=0)
    W[1, 4] = 1  # C[0,1] (k=1,i=0)
    
    # M6 = (c-a)(e+f)
    # U: c=1, a=-1; V: e=1, f=1; W: contributes to C[1,1]
    U[2, 5] = 1  # c
    U[0, 5] = -1  # a
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[3, 5] = 1  # C[1,1] (k=1,i=1)
    
    # M7 = (b-d)(g+h)
    # U: b=1, d=-1; V: g=1, h=1; W: contributes to C[0,0]
    U[1, 6] = 1  # b
    U[3, 6] = -1  # d
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = 1  # C[0,0] (k=0,i=0)
    
    return U, V, W

def _schoolbook_block(n, m, p, U_offset, V_offset, W_offset, U, V, W, start_col):
    """Add schoolbook decomposition for n x m x p block at given offsets.
    Returns new start_col.
    """
    col = start_col
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[U_offset + i * m + j, col] = 1
                V[V_offset + j * p + k, col] = 1
                W[W_offset + k * n + i, col] = 1
                col += 1
    return col

def _combine_decompositions(decomps, n_offsets, m_offsets, p_offsets, n, m, p):
    """Combine multiple decompositions for blocks.
    decomps: list of (U_block, V_block, W_block) for each block
    n_offsets, m_offsets, p_offsets: row/col offsets for each block
    Returns full U (n*m, total_rank), V (m*p, total_rank), W (p*n, total_rank)
    """
    total_rank = sum(U_block.shape[1] for U_block, _, _ in decomps)
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for idx, (U_block, V_block, W_block) in enumerate(decomps):
        n_off = n_offsets[idx]
        m_off = m_offsets[idx]
        p_off = p_offsets[idx]
        rank = U_block.shape[1]
        
        U_full[n_off:(n_off + U_block.shape[0]), col:col+rank] = U_block
        V_full[m_off:(m_off + V_block.shape[0]), col:col+rank] = V_block
        W_full[p_off:(p_off + W_block.shape[0]), col:col+rank] = W_block
        col += rank
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct.
    Check: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for cross terms: the bilinear form correctly computes C = A*B.
    """
    rank = U.shape[1]
    # Check the tensor condition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Also verify with random matrices
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # L_r = sum_{i,j} U[i*m+j, r] * A[i,j]
            L_r = 0.0
            for i in range(n):
                for j in range(m):
                    L_r += U[i*m+j, r] * A[i, j]
            # M_r = sum_{j,k} V[j*p+k, r] * B[j,k]
            M_r = 0.0
            for j in range(m):
                for k in range(p):
                    M_r += V[j*p+k, r] * B[j, k]
            # C[i,k] += W[k*n+i, r] * L_r * M_r
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k*n+i, r] * L_r * M_r
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 120.0
    
    # Schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Try block decomposition using Strassen for 2x2x2 blocks
    # Strategy: Split the 4-dim middle into two 2-dim parts
    # C = A[:, :2] @ B[:2, :] + A[:, 2:] @ B[2:, :]
    # Each term is 2x2 @ 2x7
    
    # For 2x2 @ 2x7: split 7 into 2+2+3
    # 2x2 @ 2x2: use Strassen (rank 7)
    # 2x2 @ 2x2: use Strassen (rank 7)
    # 2x2 @ 2x3: use schoolbook (rank 12) or try to optimize
    
    # Build 2x2x2x7 decomposition
    def build_2x2x2x7():
        """Build decomposition for 2x2 @ 2x7 using Strassen for 2x2x2 parts."""
        # Split 7 into 2+2+3
        # Block 1: 2x2 @ 2x2 (cols 0-1 of B)
        # Block 2: 2x2 @ 2x2 (cols 2-3 of B)
        # Block 3: 2x2 @ 2x3 (cols 4-6 of B)
        
        U1, V1, W1 = _strassen_222()  # rank 7
        U2, V2, W2 = _strassen_222()  # rank 7
        # For 2x2 @ 2x3, use schoolbook
        n3, m3, p3 = 2, 2, 3
        U3 = np.zeros((n3*m3, n3*m3*p3), dtype=float)
        V3 = np.zeros((m3*p3, n3*m3*p3), dtype=float)
        W3 = np.zeros((p3*n3, n3*m3*p3), dtype=float)
        col3 = 0
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    U3[i*m3+j, col3] = 1
                    V3[j*p3+k, col3] = 1
                    W3[k*n3+i, col3] = 1
                    col3 += 1
        
        # Combine: n=2, m=2, p=7
        # Block 1: A rows 0-1, B cols 0-1, C cols 0-1
        # Block 2: A rows 0-1, B cols 2-3, C cols 2-3
        # Block 3: A rows 0-1, B cols 4-6, C cols 4-6
        
        n_off = [0, 0, 0]
        m_off = [0, 0, 0]  # A is always 2x2
        p_off = [0, 2, 4]  # B column offsets
        
        decomps = [(U1, V1, W1), (U2, V2, W2), (U3, V3, W3)]
        U_full, V_full, W_full = _combine_decompositions(
            decomps, n_off, m_off, p_off, 2, 2, 7
        )
        return U_full, V_full, W_full
    
    # Build full 2x4x7 decomposition
    def build_2x4x7():
        """Build 2x4 @ 2x7 by splitting the 4-dim middle into two 2-dim parts."""
        # C = A[:, :2] @ B[:2, :] + A[:, 2:] @ B[2:, :]
        # Term 1: 2x2 @ 2x7 (A cols 0-1, B rows 0-1)
        # Term 2: 2x2 @ 2x7 (A cols 2-3, B rows 2-3)
        
        U1, V1, W1 = build_2x2x2x7()
        U2, V2, W2 = build_2x2x2x7()
        
        # Combine: n=2, m=4, p=7
        # Block 1: A cols 0-1 (m_off=0), B rows 0-1 (m_off=0 for B)
        # Block 2: A cols 2-3 (m_off=2), B rows 2-3 (m_off=2 for B)
        
        n_off = [0, 0]
        m_off_A = [0, 2]  # A column offsets
        m_off_B = [0, 2]  # B row offsets
        p_off = [0, 0]
        
        # We need to handle the fact that U has m=4 but each block uses m=2
        # U1 is (2*2, rank1) = (4, rank1), but it only uses the first 2 rows of A
        # Actually, U1 has shape (n*m_block, rank) = (2*2, rank) = (4, rank)
        # But we need to place it at the correct rows in the full U (n*m, rank) = (2*4, rank) = (8, rank)
        
        total_rank = U1.shape[1] + U2.shape[1]
        U_full = np.zeros((2*4, total_rank), dtype=float)
        V_full = np.zeros((4*7, total_rank), dtype=float)
        W_full = np.zeros((7*2, total_rank), dtype=float)
        
        col = 0
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        
        # Place block 1: A cols 0-1, B rows 0-1
        # U1[i*2+j, r] for i in 0-1, j in 0-1 -> U_full[i*4+j, col+r]
        for i in range(2):
            for j in range(2):
                U_full[i*4+j, col:col+rank1] = U1[i*2+j, :]
        # V1[j*2+k, r] for j in 0-1, k in 0-6 -> V_full[j*7+k, col+r]
        for j in range(2):
            for k in range(7):
                V_full[j*7+k, col:col+rank1] = V1[j*2+k, :]
        # W1[k*2+i, r] for k in 0-6, i in 0-1 -> W_full[k*2+i, col+r]
        for k in range(7):
            for i in range(2):
                W_full[k*2+i, col:col+rank1] = W1[k*2+i, :]
        col += rank1
        
        # Place block 2: A cols 2-3, B rows 2-3
        # U2[i*2+j, r] for i in 0-1, j in 0-1 -> U_full[i*4+(2+j), col+r]
        for i in range(2):
            for j in range(2):
                U_full[i*4+(2+j), col:col+rank2] = U2[i*2+j, :]
        # V2[j*2+k, r] for j in 0-1, k in 0-6 -> V_full[(2+j)*7+k, col+r]
        for j in range(2):
            for k in range(7):
                V_full[(2+j)*7+k, col:col+rank2] = V2[j*2+k, :]
        # W2[k*2+i, r] for k in 0-6, i in 0-1 -> W_full[k*2+i, col+r]
        for k in range(7):
            for i in range(2):
                W_full[k*2+i, col:col+rank2] = W2[k*2+i, :]
        col += rank2
        
        return U_full, V_full, W_full
    
    # Try the block decomposition
    try:
        U_try, V_try, W_try = build_2x4x7()
        if _verify_decomposition(U_try, V_try, W_try, n, m, p):
            rank_try = U_try.shape[1]
            if rank_try < best_rank:
                best_rank = rank_try
                best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                print(f"Found valid decomposition with rank {rank_try}")
    except Exception as e:
        print(f"Block decomposition failed: {e}")
    
    # Check time
    if time.time() - start_time > time_limit - 10:
        return best_result
    
    # Try alternative: split 4 as 2+2 differently or use other block sizes
    # Try 2x4 @ 4x7 with 4 split as 1+1+1+1 (schoolbook, no improvement)
    # Try using Strassen on 2x2x2 and optimize the 2x2x3 part
    
    # For now, return the best found
    return best_result
