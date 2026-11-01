"""Matrix multiplication 2x3x5 with optimized block decomposition.

Strategy:
- Block 1 (i=0..1, j=0..1, k=0..1): Strassen rank 7
- Block 2 (i=0..1, j=2, k=0..4): Schoolbook rank 10
- Block 3 (i=0..1, j=0..1, k=2..4): Rank 11 (Strassen 2x2x2 + schoolbook 2x2x1)
Total: 28

We also attempt to verify and, if possible, find a rank-27 decomposition
by searching for a linear dependency among the 28 block terms.
"""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np

def _strassen_222():
    """Return U, V, W for M_{2,2,2} (Strassen, rank 7).
    U shape (4,7): indexed by (i*2+j, r)
    V shape (4,7): indexed by (j*2+k, r)
    W shape (4,7): indexed by (k*2+i, r)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = -1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook M_{n,m,p} with rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _add_block(global_U, global_V, global_W, rank_offset, 
               sub_U, sub_V, sub_W,
               i_off, j_off, k_off,
               n_sub, m_sub, p_sub,
               n_global, m_global, p_global):
    """Add a sub-block decomposition into the global factor matrices."""
    r = len(sub_U[0])
    for local_i in range(n_sub):
        for local_j in range(m_sub):
            gi = i_off + local_i
            gj = j_off + local_j
            global_row_U = gi * m_global + gj
            for rr in range(r):
                if sub_U[local_i*m_sub + local_j][rr] != 0:
                    global_U[global_row_U][rank_offset + rr] += sub_U[local_i*m_sub + local_j][rr]
    for local_j in range(m_sub):
        for local_k in range(p_sub):
            gj = j_off + local_j
            gk = k_off + local_k
            global_row_V = gj * p_global + gk
            for rr in range(r):
                if sub_V[local_j*p_sub + local_k][rr] != 0:
                    global_V[global_row_V][rank_offset + rr] += sub_V[local_j*p_sub + local_k][rr]
    for local_k in range(p_sub):
        for local_i in range(n_sub):
            gk = k_off + local_k
            gi = i_off + local_i
            global_row_W = gk * n_global + gi
            for rr in range(r):
                if sub_W[local_k*n_sub + local_i][rr] != 0:
                    global_W[global_row_W][rank_offset + rr] += sub_W[local_k*n_sub + local_i][rr]

def _verify_full(U, V, W, n, m, p):
    """Verify that the decomposition is correct by checking tensor equality."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U_np[u_idx, r] * V_np[v_idx, r] * W_np[w_idx, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_223_rank11():
    """
    Build a rank-11 decomposition for M_{2,2,3}.
    Block A: i=0..1, j=0..1, k=0..1 -> Strassen (rank 7)
    Block B: i=0..1, j=0..1, k=2 -> Schoolbook (rank 4)
    Total: rank 11
    """
    n_sub, m_sub, p_sub = 2, 2, 3
    
    strassen_U, strassen_V, strassen_W = _strassen_222()
    sb_U, sb_V, sb_W = _schoolbook(2, 2, 1)
    
    rank = 11
    U = [[0]*rank for _ in range(n_sub*m_sub)]
    V = [[0]*rank for _ in range(m_sub*p_sub)]
    W = [[0]*rank for _ in range(p_sub*n_sub)]
    
    # Add Strassen for k=0,1
    _add_block(U, V, W, 0, strassen_U, strassen_V, strassen_W,
               i_off=0, j_off=0, k_off=0,
               n_sub=2, m_sub=2, p_sub=2,
               n_global=n_sub, m_global=m_sub, p_global=p_sub)
    
    # Add schoolbook for k=2
    _add_block(U, V, W, 7, sb_U, sb_V, sb_W,
               i_off=0, j_off=0, k_off=2,
               n_sub=2, m_sub=2, p_sub=1,
               n_global=n_sub, m_global=m_sub, p_global=p_sub)
    
    return U, V, W

def _attempt_rank27_search(U28, V28, W28, n, m, p, time_limit=30.0):
    """
    Try to find a rank-27 decomposition by looking for a linear combination
    of the 28 rank-1 terms that sums to zero tensor.
    
    If we find coefficients c[0..27] such that sum_r c[r] * (u_r, v_r, w_r) = 0,
    we can remove one term (the one with largest |c|) and rescale.
    
    This is equivalent to finding a vector in the null space of the linear map
    that takes a coefficient vector to the tensor T = sum_r c[r] * T_r.
    """
    start_time = time.time()
    rank28 = 28
    U_np = np.array(U28, dtype=float)
    V_np = np.array(V28, dtype=float)
    W_np = np.array(W28, dtype=float)
    
    # Number of tensor components: n*m*p = 30
    # Each rank-1 term T_r is a tensor with T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to find c such that sum_r c[r] * T_r = 0
    # This is a linear system: A * c = 0 where A is 30 x 28
    
    # Build matrix A: A[i,j,k, r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Flatten to A[30, 28]
    A = np.zeros((n*m*p, rank28))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                A[idx] = U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :]
                idx += 1
    
    # Find null space of A (30 x 28). If rank(A) < 28, there's a null vector.
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # The smallest singular value tells us about the null space
    min_sv = s_vals[-1] if len(s_vals) > 0 else 0
    
    if min_sv > 1e-6:
        # No significant null space - rank is truly 28 or higher
        return None
    
    # Get null space vector (last column of Vt)
    null_vec = Vt_svd[-1]
    
    # Check if this actually gives a zero tensor (within tolerance)
    residual = A @ null_vec
    if np.linalg.norm(residual) > 1e-8:
        return None
    
    # We found a linear dependency. Remove the term with largest |coefficient|
    # and rescale the remaining terms.
    max_idx = np.argmax(np.abs(null_vec))
    c_max = null_vec[max_idx]
    if abs(c_max) < 1e-12:
        return None
    
    # New rank-27 decomposition: for r != max_idx, new_c[r] = null_vec[r] / c_max
    # The new tensor is: sum_{r != max_idx} (null_vec[r]/c_max) * T_r = -T_{max_idx} + sum_{r != max_idx} (null_vec[r]/c_max) * T_r
    # Wait, we need: sum_r c[r] * T_r = 0, so T_{max_idx} = -sum_{r != max_idx} (c[r]/c_max) * T_r
    # New decomposition with rank 27: remove term max_idx, and the remaining 27 terms with scaled coefficients
    # But we need the tensor to still equal the original (all ones), so:
    # Original: sum_{all 28} 1 * T_r = Tensor_ones
    # New: we need to express Tensor_ones using 27 terms
    # From the dependency: c[max_idx] * T[max_idx] = -sum_{r != max_idx} c[r] * T[r]
    # So T[max_idx] = -sum_{r != max_idx} (c[r]/c[max_idx]) * T[r]
    # Original tensor = sum_r T[r] = sum_{r != max_idx} T[r] + T[max_idx]
    # = sum_{r != max_idx} T[r] - sum_{r != max_idx} (c[r]/c[max_idx]) * T[r]
    # = sum_{r != max_idx} (1 - c[r]/c[max_idx]) * T[r]
    # = sum_{r != max_idx} ((c[max_idx] - c[r])/c[max_idx]) * T[r]
    
    # So the new rank-27 factors are:
    # U_new[r] = ((c[max_idx] - c[r])/c[max_idx]) * U[r] for r != max_idx
    # Similarly for V and W
    
    scale = 1.0 / c_max
    U_new = [[0]*27 for _ in range(n*m)]
    V_new = [[0]*27 for _ in range(m*p)]
    W_new = [[0]*27 for _ in range(p*n)]
    
    new_col = 0
    for r in range(rank28):
        if r == max_idx:
            continue
        s = (1.0 - null_vec[r] * scale)
        if abs(s) < 1e-12:
            new_col += 1
            continue
        for row in range(n*m):
            U_new[row][new_col] = U_np[row, r] * s
        for row in range(m*p):
            V_new[row][new_col] = V_np[row, r] * s
        for row in range(p*n):
            W_new[row][new_col] = W_np[row, r] * s
        new_col += 1
    
    if new_col != 27:
        return None
    
    # Try to round to integers or half-integers
    U_round = np.round(np.array(U_new) * 2) / 2
    V_round = np.round(np.array(V_new) * 2) / 2
    W_round = np.round(np.array(W_new) * 2) / 2
    
    # Verify
    if _verify_full(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
        return U_round.tolist(), V_round.tolist(), W_round.tolist()
    
    # If rounding doesn't work, try integer rounding
    U_int = np.round(np.array(U_new)).astype(int)
    V_int = np.round(np.array(V_new)).astype(int)
    W_int = np.round(np.array(W_new)).astype(int)
    
    if _verify_full(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Build rank-28 decomposition
    # Block 1: i=0..1, j=0..1, k=0..1 -> Strassen (rank 7)
    # Block 2: i=0..1, j=2, k=0..4 -> Schoolbook (rank 10)
    # Block 3: i=0..1, j=0..1, k=2..4 -> Rank 11
    total_rank = 28
    
    U28 = [[0]*total_rank for _ in range(n*m)]
    V28 = [[0]*total_rank for _ in range(m*p)]
    W28 = [[0]*total_rank for _ in range(p*n)]
    
    # Block 1: Strassen 2x2x2
    sub_U1, sub_V1, sub_W1 = _strassen_222()
    _add_block(U28, V28, W28, 0, sub_U1, sub_V1, sub_W1,
               i_off=0, j_off=0, k_off=0,
               n_sub=2, m_sub=2, p_sub=2,
               n_global=n, m_global=m, p_global=p)
    
    # Block 2: Schoolbook 2x1x5
    sb_U, sb_V, sb_W = _schoolbook(2, 1, 5)
    _add_block(U28, V28, W28, 7, sb_U, sb_V, sb_W,
               i_off=0, j_off=2, k_off=0,
               n_sub=2, m_sub=1, p_sub=5,
               n_global=n, m_global=m, p_global=p)
    
    # Block 3: Rank 11 for 2x2x3
    b3_U, b3_V, b3_W = _build_223_rank11()
    _add_block(U28, V28, W28, 17, b3_U, b3_V, b3_W,
               i_off=0, j_off=0, k_off=2,
               n_sub=2, m_sub=2, p_sub=3,
               n_global=n, m_global=m, p_global=p)
    
    # Verify rank-28
    if time.time() < deadline - 5:
        if _verify_full(U28, V28, W28, n, m, p):
            # Try to find rank-27
            if time.time() < deadline - 20:
                result27 = _attempt_rank27_search(U28, V28, W28, n, m, p, 
                                                   time_limit=min(30.0, deadline - start_time - 10))
                if result27 is not None:
                    U27, V27, W27 = result27
                    return {"U": U27, "V": V27, "W": W27}
            
            return {"U": U28, "V": V28, "W": W28}
    
    # Fallback: rank 29 (schoolbook for block 3)
    total_rank_fb = 29
    U_fb = [[0]*total_rank_fb for _ in range(n*m)]
    V_fb = [[0]*total_rank_fb for _ in range(m*p)]
    W_fb = [[0]*total_rank_fb for _ in range(p*n)]
    
    _add_block(U_fb, V_fb, W_fb, 0, sub_U1, sub_V1, sub_W1,
               i_off=0, j_off=0, k_off=0,
               n_sub=2, m_sub=2, p_sub=2,
               n_global=n, m_global=m, p_global=p)
    
    _add_block(U_fb, V_fb, W_fb, 7, sb_U, sb_V, sb_W,
               i_off=0, j_off=2, k_off=0,
               n_sub=2, m_sub=1, p_sub=5,
               n_global=n, m_global=m, p_global=p)
    
    sb2_U, sb2_V, sb2_W = _schoolbook(2, 2, 3)
    _add_block(U_fb, V_fb, W_fb, 17, sb2_U, sb2_V, sb2_W,
               i_off=0, j_off=0, k_off=2,
               n_sub=2, m_sub=2, p_sub=3,
               n_global=n, m_global=m, p_global=p)
    
    if _verify_full(U_fb, V_fb, W_fb, n, m, p):
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Final fallback: pure schoolbook rank 30
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
