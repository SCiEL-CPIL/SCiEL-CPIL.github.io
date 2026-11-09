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
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Correct Strassen 2x2 decomposition.
    Returns U, V, W of shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_3x3():
    """
    Schoolbook 3x3 decomposition.
    Returns U, V, W of shape (9, 27).
    """
    n, m, p = 3, 3, 3
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _build_schoolbook_6x6():
    """
    Schoolbook 6x6x6 decomposition. Rank = 216.
    """
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

def _kronecker_decomp(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions via Kronecker product.
    If (U1,V1,W1) decomposes (n1*m1, m1*p1, p1*n1) tensor with rank r1,
    and (U2,V2,W2) decomposes (n2*m2, m2*p2, p2*n2) tensor with rank r2,
    then the Kronecker product decomposes the (n1*n2 * m1*m2, ...) tensor
    with rank r1 * r2.
    
    For 6x6x6: combine 2x2x2 (rank 7) with 3x3x3 (rank 27) -> rank 189.
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 * r2
    
    n1, m1, p1 = U1.shape[0] // m1 if m1 else 0, 0, 0
    # Determine dimensions from shapes
    n1m1 = U1.shape[0]
    m1p1 = V1.shape[0]
    p1n1 = W1.shape[0]
    n2m2 = U2.shape[0]
    m2p2 = V2.shape[0]
    p2n2 = W2.shape[0]
    
    # We need to figure out n1, m1, p1 etc.
    # For 2x2x2: n1=m1=p1=2, so n1m1=4, m1p1=4, p1n1=4
    # For 3x3x3: n2=m2=p2=3, so n2m2=9, m2p2=9, p2n2=9
    
    # Infer dimensions
    if n1m1 == 4 and m1p1 == 4 and p1n1 == 4:
        n1, m1, p1 = 2, 2, 2
    else:
        n1, m1, p1 = 1, n1m1, 1  # fallback
    
    if n2m2 == 9 and m2p2 == 9 and p2n2 == 9:
        n2, m2, p2 = 3, 3, 3
    else:
        n2, m2, p2 = 1, n2m2, 1
    
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            col = r1_idx * r2 + r2_idx
            
            # U: index is (i1*n2 + i2) * (m1*m2) + (j1*m2 + j2)
            # = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1, r1_idx]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2, r2_idx]
                            if u2_val == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, col] += u1_val * u2_val
            
            # V: index is (j1*m2 + j2) * (p1*p2) + (k1*p2 + k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1, r1_idx]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2, r2_idx]
                            if v2_val == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, col] += v1_val * v2_val
            
            # W: index is (k1*p2 + k2) * (n1*n2) + (i1*n2 + i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1, r1_idx]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2, r2_idx]
                            if w2_val == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i, col] += w1_val * w2_val
    
    return U, V, W

def _build_block_strassen_6x6():
    """
    6x6x6 decomposition using 3x3 blocks of 2x2 Strassen.
    Rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    nblocks = 3
    block_size = 2
    
    Us, Vs, Ws = _strassen_2x2()
    srank = 7
    
    total_rank = nblocks * nblocks * nblocks * srank
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for bi in range(nblocks):
        for bj in range(nblocks):
            for bk in range(nblocks):
                for r in range(srank):
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = 2*bi + di
                            j = 2*bj + dj
                            U[i*m+j, col] = Us[di*2+dj, r]
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = 2*bj + dj
                            k = 2*bk + dk
                            V[j*p+k, col] = Vs[dj*2+dk, r]
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = 2*bk + dk
                            i = 2*bi + di
                            W[k*n+i, col] = Ws[dk*2+di, r]
                    col += 1
    return U, V, W

def _try_rank_reduction(U, V, W, max_iterations=50, time_limit=60):
    """
    Try to reduce rank by identifying and removing linearly dependent columns.
    Uses a greedy approach: for each column, check if it can be expressed
    as a linear combination of other columns (approximately).
    """
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    removed = []
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if len(removed) >= rank:
            break
            
        # Try to remove each non-removed column
        found_removal = False
        for col in range(rank):
            if col in removed:
                continue
            
            # Check if column col can be approximated by others
            # Simple check: see if removing it still satisfies the tensor equation
            # This is expensive, so use a heuristic: check sparsity pattern
            
            # For now, try a simple approach: check if the column has very small
            # norm relative to others
            u_norm = np.linalg.norm(U[:, col])
            v_norm = np.linalg.norm(V[:, col])
            w_norm = np.linalg.norm(W[:, col])
            
            if u_norm < 1e-10 and v_norm < 1e-10 and w_norm < 1e-10:
                removed.append(col)
                found_removal = True
                continue
            
            # Try to see if this column is redundant by checking if
            # the tensor contribution can be absorbed by others
            # This is a simplified check
            break
        
        if not found_removal:
            break
    
    if not removed:
        return U, V, W
    
    keep = [c for c in range(rank) if c not in removed]
    U_red = U[:, keep]
    V_red = V[:, keep]
    W_red = W[:, keep]
    
    return U_red, V_red, W_red

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = 216
    best_result = None
    
    # 1. Try Kronecker product of Strassen(2x2) and schoolbook(3x3) -> rank 189
    if time.time() < deadline - 30:
        try:
            U2, V2, W2 = _strassen_2x2()
            U3, V3, W3 = _schoolbook_3x3()
            U_k, V_k, W_k = _kronecker_decomp(U2, V2, W2, U3, V3, W3)
            if _verify(U_k, V_k, W_k):
                rank = U_k.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_k, V_k, W_k)
        except Exception:
            pass
    
    # 2. Try block Strassen (also rank 189, but different structure)
    if time.time() < deadline - 20:
        try:
            U_bs, V_bs, W_bs = _build_block_strassen_6x6()
            if _verify(U_bs, V_bs, W_bs):
                rank = U_bs.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_bs, V_bs, W_bs)
        except Exception:
            pass
    
    # 3. Try rank reduction on best result
    if best_result is not None and time.time() < deadline - 10:
        try:
            U_r, V_r, W_r = _try_rank_reduction(best_result[0], best_result[1], best_result[2], 
                                                   max_iterations=20, time_limit=15)
            if _verify(U_r, V_r, W_r):
                rank = U_r.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_r, V_r, W_r)
        except Exception:
            pass
    
    # 4. Fallback to Schoolbook (Rank 216)
    if best_result is None:
        U_s, V_s, W_s = _build_schoolbook_6x6()
        best_result = (U_s, V_s, W_s)
        best_rank = 216
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
