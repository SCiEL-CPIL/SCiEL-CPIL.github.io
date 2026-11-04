"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # seconds safety margin before 150s kill

def _build_strassen_2x2():
    """Build Strassen's rank-7 decomposition for 2x2 matrices."""
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0, -1, 0,  1,  0,  1]
    ], dtype=int)
    
    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0]
    ], dtype=int)
    
    return U2, V2, W2

def _verify_decomposition(U, V, W, n=4, m=4, p=4):
    """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == delta(i,k) for all i,j,k."""
    rank = len(U[0])
    # Build the tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                T[i,j,k] = s
    # Check: T[i,j,k] should be 1 if i==k, 0 otherwise
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if i == k else 0
                if T[i,j,k] != expected:
                    return False
    return True

def _build_rank49():
    """Build the rank-49 Kronecker-of-Strassen decomposition."""
    n, m, p = 4, 4, 4
    U2, V2, W2 = _build_strassen_2x2()
    
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Permutations to match required indexing
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        perm_U[S] = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
    
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        perm_V[S] = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
    
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        perm_W[S] = (k_out * 2 + i_out) * 4 + (k_in * 2 + i_in)
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    return U4.tolist(), V4.tolist(), W4.tolist()

def _try_reduce_rank(U, V, W, n=4, m=4, p=4, max_time=130.0):
    """
    Try to reduce rank by finding linearly dependent rank-1 tensor terms.
    Each term r defines a vector v_r in R^(n*m*p) where v_r[i*m+j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    Actually, the tensor is T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
    Term r contributes a rank-1 tensor: t_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
    We can flatten each t_r to a vector of length 64 and check for linear dependence.
    """
    rank = len(U[0])
    
    # Build the 64-dim vector for each rank-1 term
    # Flatten index: idx = (i*m+j)*p + k
    vecs = []
    for r in range(rank):
        v = np.zeros(n*m*p, dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v[(i*m+j)*p + k] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        vecs.append(v)
    
    vecs = np.array(vecs, dtype=float)  # (rank, 64)
    
    # Find the rank of this matrix using SVD
    try:
        U_svd, s, Vt = np.linalg.svd(vecs, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Determine numerical rank with tolerance
    tol = 1e-10
    num_rank = np.sum(s > tol)
    
    if num_rank >= rank:
        return None  # No reduction possible
    
    # Get the column indices of the pivot columns from QR with column pivoting
    # Use QR decomposition with column pivoting
    Q, R, piv = np.linalg.qr(vecs.T, mode='complete', pivoting=True)
    
    # The first num_rank pivot columns are independent
    pivot_cols = piv[:num_rank]
    
    # Verify that using only these columns still gives a valid decomposition
    U_new = [[U[i][r] for r in pivot_cols] for i in range(n*m)]
    V_new = [[V[i][r] for r in pivot_cols] for i in range(m*p)]
    W_new = [[W[i][r] for r in pivot_cols] for i in range(p*n)]
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    
    return None

def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    Starts with Kronecker-of-Strassen rank 49, then attempts rank reduction.
    """
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: schoolbook rank-64
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                c = i * m + j
                U[i*m + j][c] = 1
                for k in range(p):
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    n, m, p = DIMENSIONS
    
    # Build rank-49 decomposition
    U4, V4, W4 = _build_rank49()
    
    # Verify it's correct
    if not _verify_decomposition(U4, V4, W4, n, m, p):
        # Shouldn't happen, but fallback
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                c = i * m + j
                U[i*m + j][c] = 1
                for k in range(p):
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if time.time() - _START_TIME < _DEADLINE - 5:
        reduced = _try_reduce_rank(U4, V4, W4, n, m, p)
        if reduced is not None:
            U4, V4, W4 = reduced
    
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
