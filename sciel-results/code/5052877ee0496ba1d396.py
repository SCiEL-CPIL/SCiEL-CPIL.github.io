import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _reduce_rank(U, V, W, tol=1e-9):
    """
    Attempts to reduce the rank of the decomposition by finding linear dependencies.
    Returns a new decomposition with potentially lower rank.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = U.shape[0] // 2, V.shape[0] // 7, W.shape[0] // 2 # Approximate, better to pass or infer
    # Infer dimensions from shapes
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    nm, rank = U.shape
    mp, _ = V.shape
    pn, _ = W.shape
    
    # Flatten the tensor space
    tensor_dim = nm * mp * pn
    
    # Construct the matrix A where each column is the flattened rank-1 tensor u_r v_r w_r
    A = np.zeros((tensor_dim, rank))
    for r in range(rank):
        A[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
    
    # Find the null space of A^T? No, we want to find c such that A c = 0.
    # This means the columns of A are linearly dependent.
    # We can use SVD to find the rank.
    try:
        U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
        # Determine numerical rank
        tol_svd = max(A.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
        num_rank = np.sum(s > tol_svd)
        
        if num_rank < rank:
            # There are dependencies.
            # The null space of A^T is spanned by the last (rank - num_rank) rows of Vt_svd.
            # Wait, Vt_svd is (rank, rank). The null space of A (matrix tensor_dim x rank)
            # is the null space of A^T? No.
            # We want c such that A c = 0.
            # The columns of A are linearly dependent.
            # The right null space of A is spanned by the eigenvectors corresponding to zero singular values.
            # In SVD A = U s Vt, the columns of Vt corresponding to small s are the null space.
            
            # Let's identify the indices of the dependent columns.
            # We can simply drop the columns corresponding to the smallest singular values.
            # But we must ensure the remaining columns still span the same space.
            
            # A safer approach: Use QR with column pivoting on A^T?
            # Or just keep the first 'num_rank' columns if the SVD suggests the last ones are dependent?
            # This is heuristic.
            
            # Let's try to project out the null space components.
            # Actually, if we just drop the columns associated with the smallest singular values,
            # the approximation might be good, but we need EXACT decomposition.
            
            # For integer/rational decompositions, dependencies are exact.
            # We can check for integer dependencies.
            
            # Since we are in a time-limited context, let's just return the original if we can't be sure.
            # Or, we can try to solve for the dependencies exactly using fractions?
            # Too slow.
            
            # Let's just return the original for now, as heuristic SVD might break exactness.
            pass
    except Exception:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Strassen-based decomposition for 2x4 by 4x7
    # Rank 50
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M2
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        # M3
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M4
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        # M5
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        # M6
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        # M7
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)
    
    if col != rank:
        return _schoolbook(n, m, p)
    
    # Attempt to reduce rank if possible
    # For now, just verify
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if _verify(U, V, W, n, m, p):
        return result
    
    return _schoolbook(n, m, p)
