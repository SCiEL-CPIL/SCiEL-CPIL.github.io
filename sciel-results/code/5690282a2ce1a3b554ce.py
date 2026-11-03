import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def _build_schoolbook(n, m, p):
    """Build the rank n*m*p schoolbook decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k
    # (since C_{i,k} = sum_j A_{i,j} B_{j,k}, each term has coefficient 1)
    # T_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    T_correct = np.ones((n, m, p))
    
    # Efficient check: for each r, add the outer product contribution
    max_diff = 0.0
    T_recon = np.zeros((n, m, p))
    
    for r in range(U.shape[1]):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        # Contribution: u_r[i,j] * v_r[j,k] * w_r[k,i]
        T_recon += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
    
    diff = np.max(np.abs(T_recon - T_correct))
    return diff < tol

def _build_strassen_block(bn, bm, bp):
    """
    Build a Strassen decomposition for a bn x bm times bm x bp block.
    Returns U_block (bn*bm, rank), V_block (bm*bp, rank), W_block (bp*bn, rank)
    """
    # Strassen for 2x2 blocks: 7 multiplications
    # We'll use this for any even-sized block, but it's most natural for 2x2
    # For simplicity, let's just do schoolbook for non-2x2 cases
    # Actually, let's implement a general block decomposition using known results
    
    # For now, just return schoolbook for the block
    rank = bn * bm * bp
    U_b = np.zeros((bn * bm, rank))
    V_b = np.zeros((bm * bp, rank))
    W_b = np.zeros((bp * bn, rank))
    col = 0
    for i in range(bn):
        for j in range(bm):
            for k in range(bp):
                U_b[i * bm + j, col] = 1
                V_b[j * bp + k, col] = 1
                W_b[k * bn + i, col] = 1
                col += 1
    return U_b, V_b, W_b

def _build_hybrid_block_decomposition(n, m, p):
    """
    Try to use block structure to reduce rank below n*m*p.
    Partition the dimensions into blocks and use Strassen-like ideas.
    """
    # Try different block sizes
    # For 3x4x7, we can try:
    # - 2x2 blocks where possible
    # - Remaining rows/cols handled separately
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try partitioning n=3 as 2+1, m=4 as 2+2, p=7 as 3+4 or 2+5 etc.
    # This is complex, so let's try a simpler approach:
    # Use the fact that we can decompose the problem into independent blocks
    
    # Strategy: partition into blocks where each block uses schoolbook
    # but we try to find if there's overlap that can be exploited
    
    # For now, let's try a few specific partitionings
    partitionings = [
        # (n_parts, m_parts, p_parts) - each is a list of block sizes
        ([2, 1], [2, 2], [3, 4]),
        ([2, 1], [2, 2], [2, 5]),
        ([1, 2], [2, 2], [3, 4]),
        ([2, 1], [1, 3], [3, 4]),
        ([3], [2, 2], [3, 4]),
        ([2, 1], [4], [3, 4]),
    ]
    
    for n_parts, m_parts, p_parts in partitionings:
        if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
            continue
        
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        total_rank = 0
        for bn, bm, bp in zip(n_parts, m_parts, p_parts):
            # For this block, use schoolbook (or could use Strassen if 2x2x2)
            U_b, V_b, W_b = _build_strassen_block(bn, bm, bp)
            U_blocks.append(U_b)
            V_blocks.append(V_b)
            W_blocks.append(W_b)
            total_rank += U_b.shape[1]
        
        if total_rank < best_rank:
            # Need to embed into full space
            U_full = np.zeros((n * m, total_rank))
            V_full = np.zeros((m * p, total_rank))
            W_full = np.zeros((p * n, total_rank))
            
            col_offset = 0
            n_offset = 0
            m_offset = 0
            p_offset = 0
            
            # This partitioning approach doesn't work directly because
            # the blocks need to be aligned in the full index space
            # Let me try a different approach
            
            pass
        
        n_offset = 0
        for bn in n_parts:
            m_offset = 0
            for bm in m_parts:
                p_offset = 0
                for bp in p_parts:
                    # This block corresponds to rows n_offset:n_offset+bn of A,
                    # cols m_offset:m_offset+bm of A (and B rows),
                    # cols p_offset:p_offset+bp of B (and C cols)
                    # Schoolbook for this block:
                    for i in range(bn):
                        for j in range(bm):
                            for k in range(bp):
                                global_u_idx = (n_offset + i) * m + (m_offset + j)
                                global_v_idx = (m_offset + j) * p + (p_offset + k)
                                global_w_idx = (p_offset + k) * n + (n_offset + i)
                                U_full[global_u_idx, col_offset] = 1
                                V_full[global_v_idx, col_offset] = 1
                                W_full[global_w_idx, col_offset] = 1
                                col_offset += 1
                    p_offset += bp
                m_offset += bm
            n_offset += bn
        
        if col_offset < best_rank:
            U_full = U_full[:, :col_offset]
            V_full = V_full[:, :col_offset]
            W_full = W_full[:, :col_offset]
            if _verify(U_full, V_full, W_full, n, m, p):
                best_U = U_full
                best_V = V_full
                best_W = W_full
                best_rank = col_offset
    
    if best_U is not None:
        return best_U, best_V, best_W
    return None, None, None

def _build_strassen_2x2_block(bn, bm, bp):
    """
    Build Strassen decomposition for a 2x2x2 block.
    Strassen: 7 multiplications instead of 8.
    """
    if bn != 2 or bm != 2 or bp != 2:
        return None, None, None
    
    # Strassen decomposition for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    # 
    # M1 = a11*(b22-b12)
    # M2 = (a11+a12)*b22
    # M3 = (a21+a22)*b11
    # M4 = a22*(b12-b11)
    # M5 = (a11+a21)*(b11+b12)
    # M6 = (a12-a21)*(b21+b22)
    # M7 = (a11-a12)*(b11+b21)
    #
    # C11 = M1+M4+M5+M7
    # C12 = M2+M6
    # C21 = M3+M5
    # C22 = M1+M2+M4-M5
    
    # The tensor decomposition has 7 terms
    # U: (2*2, 7) = (4, 7) - coefficients for A elements
    # V: (2*2, 7) = (4, 7) - coefficients for B elements  
    # W: (2*2, 7) = (4, 7) - coefficients for C elements (output)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: a11*(b22-b12)
    U[0, 0] = 1  # a11
    V[2, 0] = 1  # b22
    V[1, 0] = -1 # b12
    # Output: contributes to C11 and C22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2: (a11+a12)*b22
    U[0, 1] = 1  # a11
    U[1, 1] = 1  # a12
    V[2, 1] = 1  # b22
    W[1, 1] = 1  # C12
    W[3, 1] = 1  # C22
    
    # M3: (a21+a22)*b11
    U[2, 2] = 1  # a21
    U[3, 2] = 1  # a22
    V[0, 2] = 1  # b11
    W[2, 2] = 1  # C21
    
    # M4: a22*(b12-b11)
    U[3, 3] = 1  # a22
    V[1, 3] = 1  # b12
    V[0, 3] = -1 # b11
    W[0, 3] = 1  # C11
    W[3, 3] = 1  # C22
    
    # M5: (a11+a21)*(b11+b12)
    U[0, 4] = 1  # a11
    U[2, 4] = 1  # a21
    V[0, 4] = 1  # b11
    V[1, 4] = 1  # b12
    W[0, 4] = 1  # C11
    W[2, 4] = 1  # C21
    
    # M6: (a12-a21)*(b21+b22)
    U[1, 5] = 1  # a12
    U[2, 5] = -1 # a21
    V[3, 5] = 1  # b21
    V[2, 5] = 1  # b22
    W[1, 5] = 1  # C12
    
    # M7: (a11-a12)*(b11+b21)
    U[0, 6] = 1  # a11
    U[1, 6] = -1 # a12
    V[0, 6] = 1  # b11
    V[3, 6] = 1  # b21
    W[0, 6] = 1  # C11
    W[2, 6] = 1  # C21
    
    return U, V, W

def _build_full_strassen_hybrid(n, m, p):
    """
    Build a decomposition that uses Strassen for 2x2x2 blocks where possible,
    and schoolbook for the rest.
    """
    # For 3x4x7:
    # Partition: n=3 as 2+1, m=4 as 2+2, p=7 as 2+5 or 3+4
    # We need to be careful about how blocks interact
    
    # Let's try: 
    # Block 1: (0:2, 0:2, 0:2) -> Strassen, rank 7
    # Block 2: (0:2, 2:4, 0:7) -> schoolbook, rank 2*2*7=28
    # Block 3: (2:3, 0:4, 0:7) -> schoolbook, rank 1*4*7=28
    # Total: 7+28+28=63 (vs schoolbook 84)
    
    # But we need to verify this is correct! The blocks are not independent
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let me build this and verify
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    
    # Block 1: rows 0-1 of A, cols 0-1 of A/B, cols 0-1 of B/C
    # Use Strassen for this 2x2x2 block
    U1, V1, W1 = _build_strassen_2x2_block(2, 2, 2)
    if U1 is not None:
        U1_full = np.zeros((n * m, U1.shape[1]))
        V1_full = np.zeros((m * p, V1.shape[1]))
        W1_full = np.zeros((p * n, W1.shape[1]))
        
        # Map local indices to global
        # U: A[i,j] -> index i*m+j
        # V: B[j,k] -> index j*p+k
        # W: C[k,i] -> index k*n+i (trace convention W[k*n+i])
        
        for r in range(U1.shape[1]):
            for li in range(2):
                for lj in range(2):
                    local_u_idx = li * 2 + lj
                    global_u_idx = li * m + lj
                    U1_full[global_u_idx, r] = U1[local_u_idx, r]
            
            for lj in range(2):
                for lk in range(2):
                    local_v_idx = lj * 2 + lk
                    global_v_idx = lj * p + lk
                    V1_full[global_v_idx, r] = V1[local_v_idx, r]
            
            for lk in range(2):
                for li in range(2):
                    local_w_idx = lk * 2 + li
                    global_w_idx = lk * n + li
                    W1_full[global_w_idx, r] = W1[local_w_idx, r]
        
        U = np.hstack([U, U1_full])
        V = np.hstack([V, V1_full])
        W = np.hstack([W, W1_full])
    
    # Block 2: rows 0-1 of A, cols 2-3 of A/B, all cols 0-6 of B/C
    # Schoolbook: rank 2*2*7=28
    U2 = np.zeros((n * m, 28))
    V2 = np.zeros((m * p, 28))
    W2 = np.zeros((p * n, 28))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(7):
                global_u_idx = (i) * m + (2 + j)  # rows 0-1, cols 2-3
                global_v_idx = (2 + j) * p + k    # cols 2-3 of B, all k
                global_w_idx = k * n + i          # all k, rows 0-1
                U2[global_u_idx, col] = 1
                V2[global_v_idx, col] = 1
                W2[global_w_idx, col] = 1
                col += 1
    
    U = np.hstack([U, U2])
    V = np.hstack([V, V2])
    W = np.hstack([W, W2])
    
    # Block 3: row 2 of A, all cols 0-3 of A/B, all cols 0-6 of B/C
    # Schoolbook: rank 1*4*7=28
    U3 = np.zeros((n * m, 28))
    V3 = np.zeros((m * p, 28))
    W3 = np.zeros((p * n, 28))
    col = 0
    for j in range(4):
        for k in range(7):
            global_u_idx = 2 * m + j  # row 2, all cols
            global_v_idx = j * p + k  # all rows of B, all cols
            global_w_idx = k * n + 2  # all k, row 2
            U3[global_u_idx, col] = 1
            V3[global_v_idx, col] = 1
            W3[global_w_idx, col] = 1
            col += 1
    
    U = np.hstack([U, U3])
    V = np.hstack([V, V3])
    W = np.hstack([W, W3])
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build schoolbook solution
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Verify schoolbook
    if _verify(U_sb, V_sb, W_sb, n, m, p):
        pass
    else:
        # Should never happen
        pass
    
    # Try hybrid Strassen decomposition
    if time.time() < deadline - 10:
        try:
            U_h, V_h, W_h = _build_full_strassen_hybrid(n, m, p)
            if _verify(U_h, V_h, W_h, n, m, p):
                if U_h.shape[1] < best_rank:
                    best_rank = U_h.shape[1]
                    best_U = U_h
                    best_V = V_h
                    best_W = W_h
        except Exception as e:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
