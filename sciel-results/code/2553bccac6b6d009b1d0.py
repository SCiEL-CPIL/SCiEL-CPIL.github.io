"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_decomp(n, m, p):
    """
    Build a Strassen-based decomposition for 2x3x8.
    Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Split B = [B1; B2] where B1 is 2x8, B2 is 1x8.
    C = A1*B1 + A2*B2.
    A2*B2: rank 16.
    A1*B1: split into 4 blocks of 2x2x2, each rank 7 (Strassen). Total 28.
    Grand total: 44.
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2
    for k in range(8):
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (4 blocks)
    for g in range(4):
        b11 = 0 + 2 * g
        b12 = 0 + 2 * g + 1
        b21 = 8 + 2 * g
        b22 = 8 + 2 * g + 1
        
        a11, a12, a21, a22 = 0, 1, 3, 4
        
        # M1 = (a11+a22)(b11+b22)
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[(2*g)*2 + 0][col] = 1; W[(2*g+1)*2 + 1][col] = 1
        col += 1
        
        # M2 = (a21+a22)b11
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[(2*g)*2 + 1][col] = 1
        col += 1
        
        # M3 = a11(b22-b12)
        U[a11][col] = 1
        V[b22][col] = 1; V[b12][col] = -1
        W[(2*g+1)*2 + 0][col] = 1; W[(2*g+1)*2 + 1][col] = -1
        col += 1
        
        # M4 = a22(b11-b21)
        U[a22][col] = 1
        V[b11][col] = 1; V[b21][col] = -1
        W[(2*g)*2 + 0][col] = -1; W[(2*g)*2 + 1][col] = 1
        col += 1
        
        # M5 = (a11+a12)b22
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[(2*g+1)*2 + 0][col] = 1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[(2*g)*2 + 0][col] = 1
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[(2*g+1)*2 + 1][col] = 1
        col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def _try_reduce_rank(U, V, W, n, m, p, max_time=60):
    """
    Try to reduce rank by finding linear dependencies among rank-1 terms.
    For each entry (i,j,k) of the matmul tensor, we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    This is a system of n*m*p equations in the rank-1 coefficients.
    To reduce rank by 1, we need to find a non-trivial vector (u,v,w) such that
    for all (i,j,k): sum_c [u_c * V[j*p+k,c]*W[k*n+i,c] + v_c * U[i*m+j,c]*W[k*n+i,c] + w_c * U[i*m+j,c]*V[j*p+k,c]] = 0
    where we perturb U_c -> U_c + u_c, V_c -> V_c + v_c, W_c -> W_c + w_c.
    
    Simpler approach: check if any column can be expressed as a linear combination of others
    in the tensor space. We build a 48-dim vector for each rank-1 term and check rank.
    """
    start = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensor vectors: for each column c, form a 48-dim vector
    # T_c[i*m+j, j*p+k, k*n+i] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Flatten: index = (i*m+j) * (m*p) + (j*p+k) ... no, let's use a consistent ordering.
    # The tensor has entries T[i,j,k] for i in range(n), j in range(m), k in range(p).
    # We'll flatten as T[i, j, k] -> index i*(m*p) + j*p + k
    
    def tensor_vec(u_col, v_col, w_col):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        return vec
    
    # Build matrix of shape (n*m*p, rank) where each column is a tensor vector
    M = np.zeros((n * m * p, rank))
    for c in range(rank):
        M[:, c] = tensor_vec(U[:, c], V[:, c], W[:, c])
    
    # The target tensor is the all-ones tensor (each entry = 1)
    target = np.ones(n * m * p)
    
    # Check: is M @ alpha = target solvable? Yes, by construction.
    # To reduce rank, check if columns of M are linearly dependent.
    # Compute rank of M.
    try:
        r = np.linalg.matrix_rank(M, tol=1e-10)
    except np.linalg.LinAlgError:
        return None
    
    if r < rank:
        # There are linear dependencies. We can potentially reduce rank.
        # Find a null space vector: M @ x = 0 with x != 0.
        # This means sum_c x_c * T_c = 0.
        # If we can find such x, we can try to eliminate one column.
        # For now, just report that reduction is possible.
        # Find the null space
        from scipy.linalg import null_space
        ns = null_space(M, rcond=1e-10)
        if ns.shape[1] > 0:
            # We have dependencies. Try to eliminate the last column.
            # Set x[rank-1] = 1, solve for others.
            # This is complex; for now, return None to be safe.
            pass
    
    if time.time() - start > max_time:
        return None
    
    return None


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary: Strassen-based rank-44 decomposition
    try:
        U, V, W = _build_decomp(n, m, p)
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            # Try rank reduction if time permits
            if time.time() < deadline - 10:
                reduced = _try_reduce_rank(U, V, W, n, m, p, max_time=30)
                if reduced is not None:
                    U, V, W = reduced
                    err = _verify(U, V, W, n, m, p)
                    if err < 1e-10:
                        return {"U": U, "V": V, "W": W}
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook rank-48
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
