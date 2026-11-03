import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return (U,V,W) for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7) with W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M5
    
    # U[i*2+j, r] corresponds to A[i,j]
    # V[j*2+k, r] corresponds to B[j,k]
    # W[k*2+i, r] corresponds to C[i,k]
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1  # C00=+1, C11=+1, C01=-1, C10=-1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1  # C10=+1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1; V[3, 2] = -1  # B10, B11
    W[1, 2] = 1; W[3, 2] = -1  # C01=+1, C11=-1
    
    # M4: A11(B01-B00)
    U[3, 3] = 1  # A11
    V[1, 3] = 1; V[0, 3] = -1  # B01, B00
    W[0, 3] = -1; W[1, 3] = 1  # C00=-1, C01=+1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = 1; W[3, 4] = -1  # C00=+1, C11=-1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1  # A10, A00
    V[0, 5] = 1; V[2, 5] = 1  # B00, B10
    W[3, 5] = 1  # C11=+1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, A11
    V[1, 6] = 1; V[3, 6] = 1  # B01, B11
    W[0, 6] = -1; W[2, 6] = 1  # C00=-1, C10=+1
    
    return U, V, W


def _strassen_4x4x4():
    """Return (U,V,W) for rank-49 Strassen decomposition of 4x4x4 matrix multiplication.
    Uses 7 blocks of 2x2x2 Strassen (rank 7 each).
    U: (16,49), V: (16,49), W: (16,49) with W[k*4+i, r].
    """
    s2U, s2V, s2W = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    strassen_ops = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1, 'C01': -1, 'C10': -1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1}),
        ({'A00': 1}, {'B10': 1, 'B11': -1}, {'C01': 1, 'C11': -1}),
        ({'A11': 1}, {'B01': 1, 'B00': -1}, {'C00': -1, 'C01': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': 1, 'C11': -1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B10': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B01': 1, 'B11': 1}, {'C00': -1, 'C10': 1}),
    ]
    
    block_U = {
        'A00': (0, 4),
        'A01': (2, 6),
        'A10': (4, 8),
        'A11': (6, 10),
    }
    block_V = {
        'B00': (0, 4),
        'B01': (2, 6),
        'B10': (4, 8),
        'B11': (6, 10),
    }
    block_W = {
        'C00': (0, 4),
        'C01': (2, 6),
        'C10': (4, 8),
        'C11': (6, 10),
    }
    
    for op_idx, (u_blocks, v_blocks, w_blocks) in enumerate(strassen_ops):
        base_r = op_idx * 7
        for r2 in range(7):
            r = base_r + r2
            for blk, c in u_blocks.items():
                s, e = block_U[blk]
                for i2 in range(2):
                    for j2 in range(2):
                        U[s + i2*2 + j2, r] += c * s2U[i2*2 + j2, r2]
            
            for blk, c in v_blocks.items():
                s, e = block_V[blk]
                for j2 in range(2):
                    for k2 in range(2):
                        V[s + j2*2 + k2, r] += c * s2V[j2*2 + k2, r2]
            
            for blk, c in w_blocks.items():
                s, e = block_W[blk]
                for k2 in range(2):
                    for i2 in range(2):
                        W[s + k2*2 + i2, r] += c * s2W[k2*2 + i2, r2]
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return (U,V,W) for schoolbook decomposition of n x m * m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _combine_decompositions(decomps):
    """Combine a list of (U,V,W) decompositions into one.
    Each (U,V,W) has the same number of rows in U, V, W as the final result.
    Returns combined (U,V,W).
    """
    if not decomps:
        return None
    
    n, m, p = N, M, P
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    
    return U, V, W


def _peel_residual(U, V, W, n, m, p, tol=1e-6):
    """Attempt to reduce rank by peeling linear dependencies.
    Returns reduced (U, V, W) or None if no reduction found."""
    R = U.shape[1]
    if R < 10:
        return None
    
    # Try to find a column r that is a linear combination of others
    # For simplicity, try removing each column and checking if the tensor is still valid
    # This is expensive, so only try a few
    
    best = None
    for r_try in range(R):
        # Build U, V, W without column r_try
        mask = [i for i in range(R) if i != r_try]
        if len(mask) < R - 1:
            continue
        
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        if _validate(U_new, V_new, W_new, tol=1e-5):
            best = (U_new, V_new, W_new)
            break
    
    return best


def _try_peeling_sequence(U, V, W, n, m, p, max_rounds=50, time_limit=30.0):
    """Iteratively peel columns to reduce rank."""
    t0 = time.time()
    U_cur, V_cur, W_cur = U, V, W
    rank_cur = U_cur.shape[1]
    
    for _ in range(max_rounds):
        if time.time() - t0 > time_limit:
            break
        
        reduced = _peel_residual(U_cur, V_cur, W_cur, n, m, p, tol=1e-5)
        if reduced is None:
            break
        
        U_cur, V_cur, W_cur = reduced
        rank_cur = U_cur.shape[1]
    
    return U_cur, V_cur, W_cur


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    # Deadline guard
    if time.time() - t0 > DEADLINE_S:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy: Split the inner dimension (5) into 1+4.
    # A is 4x5, B is 5x5, C is 4x5.
    # A = [A1 (4x1) | A2 (4x4)], B = [[B11 (1x1), B12 (1x4)], [B21 (4x1), B22 (4x4)]]
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    #
    # A1*B11: 4x1 * 1x1, schoolbook rank 4
    # A1*B12: 4x1 * 1x4, schoolbook rank 16
    # A2*B21: 4x4 * 4x1, schoolbook rank 16
    # A2*B22: 4x4 * 4x4, use Strassen rank 49
    #
    # Total: 4 + 16 + 16 + 49 = 85
    
    decomps = []
    
    # Helper to create a schoolbook decomposition embedded in the full space
    def make_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        """Create schoolbook for A[i_off:i_off+n_b, j_off:j_off+m_b] * B[j_off:j_off+m_b, k_off:k_off+p_b]
        contributing to C[i_off:i_off+n_b, k_off:k_off+p_b]
        """
        rank = n_b * m_b * p_b
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[(i_off+i)*m + (j_off+j), c] = 1.0
                    V[(j_off+j)*p + (k_off+k), c] = 1.0
                    W[(k_off+k)*n + (i_off+i), c] = 1.0
                    c += 1
        return U, V, W
    
    # A1*B11: A[0:4,0:1] * B[0:1,0:1] -> C[0:4,0:1]
    decomps.append(make_schoolbook(4, 1, 1, 0, 0, 0))
    
    # A1*B12: A[0:4,0:1] * B[0:1,1:5] -> C[0:4,1:5]
    decomps.append(make_schoolbook(4, 1, 4, 0, 0, 1))
    
    # A2*B21: A[0:4,1:5] * B[1:5,0:1] -> C[0:4,0:1]
    decomps.append(make_schoolbook(4, 4, 1, 0, 1, 0))
    
    # A2*B22: A[0:4,1:5] * B[1:5,1:5] -> C[0:4,1:5]
    # Use Strassen rank 49 for 4x4x4
    s4U, s4V, s4W = _strassen_4x4x4()
    # Embed into full space:
    # s4U has shape (16, 49) with s4U[i*4+j, r] for i,j in 0..3
    # This corresponds to A[0:4, 1:5] -> U[(0+i)*5 + (1+j), r]
    # s4V has shape (16, 49) with s4V[j*4+k, r]
    # This corresponds to B[1:5, 1:5] -> V[(1+j)*5 + (1+k), r]
    # s4W has shape (16, 49) with s4W[k*4+i, r]
    # This corresponds to C[0:4, 1:5] -> W[(1+k)*4 + (0+i), r]
    
    U_embed = np.zeros((n*m, 49))
    V_embed = np.zeros((m*p, 49))
    W_embed = np.zeros((p*n, 49))
    
    for i in range(4):
        for j in range(4):
            U_embed[i*5 + (1+j), :] = s4U[i*4+j, :]
    for j in range(4):
        for k in range(4):
            V_embed[(1+j)*5 + (1+k), :] = s4V[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_embed[(1+k)*4 + i, :] = s4W[k*4+i, :]
    
    decomps.append((U_embed, V_embed, W_embed))
    
    # Combine all decompositions
    U, V, W = _combine_decompositions(decomps)
    
    # Deadline guard: if we're running out of time, return what we have
    if time.time() - t0 > DEADLINE_S - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Validate
    if not _validate(U, V, W):
        # Fallback to schoolbook rank 100
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank via peeling
    if time.time() - t0 < DEADLINE_S - 15:
        U_red, V_red, W_red = _try_peeling_sequence(U, V, W, n, m, p, max_rounds=50, time_limit=DEADLINE_S - 15 - (time.time() - t0))
        if U_red.shape[1] < U.shape[1]:
            U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
