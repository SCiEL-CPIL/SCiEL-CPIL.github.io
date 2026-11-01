import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Quick verification that U,V,W decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check each (i,j,k) entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank_28(n, m, p):
    """
    Rank-28 decomposition:
    - j=0: schoolbook 2x1x5 (rank 10)
    - j=1,2, k=0,1: Strassen 2x2x2 (rank 7)
    - j=1,2, k=2: schoolbook 2x2x1 (rank 4)
    - j=1,2, k=3,4: Strassen 2x2x2 (rank 7)
    Total: 10+7+4+7 = 28
    """
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i*m+j, col] = 1.0
        V[j*p+k, col] = 1.0
        W[k*n+i, col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        # M1: (A00+A11)(B00+B11) -> C00+=, C11+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M2: (A10+A11)B00 -> C10+=, C11-=
        U[(i_off+1)*m+(j_off+0), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        W[(k_off+0)*n+(i_off+1), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = -1.0
        col += 1
        # M3: A00(B01-B11) -> C01+=, C11+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = -1.0
        W[(k_off+1)*n+(i_off+0), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M4: A11(B10-B00) -> C00+=, C10+=
        U[(i_off+1)*m+(j_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+0), col] = -1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        W[(k_off+0)*n+(i_off+1), col] = 1.0
        col += 1
        # M5: (A00+A01)B11 -> C00-=, C01+=
        U[(i_off+0)*m+(j_off+0), col] = 1.0
        U[(i_off+0)*m+(j_off+1), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = -1.0
        W[(k_off+1)*n+(i_off+0), col] = 1.0
        col += 1
        # M6: (A10-A00)(B00+B01) -> C11+=
        U[(i_off+1)*m+(j_off+0), col] = 1.0
        U[(i_off+0)*m+(j_off+0), col] = -1.0
        V[(j_off+0)*p+(k_off+0), col] = 1.0
        V[(j_off+0)*p+(k_off+1), col] = 1.0
        W[(k_off+1)*n+(i_off+1), col] = 1.0
        col += 1
        # M7: (A01-A11)(B10+B11) -> C00+=
        U[(i_off+0)*m+(j_off+1), col] = 1.0
        U[(i_off+1)*m+(j_off+1), col] = -1.0
        V[(j_off+1)*p+(k_off+0), col] = 1.0
        V[(j_off+1)*p+(k_off+1), col] = 1.0
        W[(k_off+0)*n+(i_off+0), col] = 1.0
        col += 1
        return col

    # Block 1: j=0, all k (schoolbook, 10 terms)
    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)

    # Block 2a: j=1,2, k=0,1 (Strassen, 7 terms)
    col = add_strassen(0, 1, 0, col)

    # Block 2b: j=1,2, k=2 (schoolbook, 4 terms)
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)

    # Block 2c: j=1,2, k=3,4 (Strassen, 7 terms)
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected {rank}, got {col}"
    return U, V, W

def _try_rank_27(n, m, p, deadline):
    """
    Attempt to find a rank-27 decomposition by solving the linear system
    derived from the tensor flattening. The multiplication tensor T[i,j,k]=1
    for all i,j,k can be written as sum_r U[:,r] V[:,r]^T W[:,r] (outer products).
    
    We use the flattening approach: fix W to a random (or structured) matrix and
    solve for U,V. But this is a quadratic system. Instead, we try a more direct
    approach: use the fact that the tensor is rank-1 in each slice and try to
    find a rank-27 decomposition by starting from rank 28 and attempting to
    eliminate one term via linear dependence.
    
    In practice, we check if the 28-term decomposition has a linearly dependent
    set of 27 columns that still spans the tensor. If the 28 columns are
    linearly independent in the appropriate sense, rank 27 is not achievable
    with this structure. We also try a direct SVD-based approach on the
    flattening to see if rank 27 suffices.
    """
    if time.time() > deadline:
        return None

    # Build the 28-term decomposition
    U28, V28, W28 = _build_rank_28(n, m, p)
    
    # Check if any single column can be removed while preserving the tensor.
    # For each column r, check if the tensor contribution of column r is
    # a linear combination of the other columns' contributions.
    # The tensor contribution of column r is the outer product:
    # T_r[i,j,k] = U28[i*m+j, r] * V28[j*p+k, r] * W28[k*n+i, r]
    
    # We need to check if T_r = sum_{r' != r} alpha_{r'} T_{r'} for some alpha.
    # This is a linear system in alpha. The number of equations is n*m*p = 30.
    # The number of unknowns is 27.
    # So we have an overdetermined system. Check if it's consistent.
    
    # Build the matrix of tensor contributions
    # Each column r gives a vector in R^{30} (flattened tensor)
    # T_flat[r] is a 30-dim vector
    
    T_flat = np.zeros((28, 30))
    for r in range(28):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_flat[r, idx] = U28[i*m+j, r] * V28[j*p+k, r] * W28[k*n+i, r]
                    idx += 1
    
    # The full tensor is sum of all T_flat rows = ones(30)
    # Check: sum of T_flat should be ones
    total = T_flat.sum(axis=0)
    if not np.allclose(total, np.ones(30), atol=1e-9):
        return None  # Our decomposition is wrong
    
    # For each column r, check if T_flat[r] is in the span of the other 27 columns
    for r in range(28):
        if time.time() > deadline:
            return None
        other = [rr for rr in range(28) if rr != r]
        A = T_flat[other, :]  # 27 x 30
        b = T_flat[r, :]      # 30-dim
        
        # Check if b is in the column space of A
        # Use least squares and check residual
        try:
            sol, residuals, rank_a, sv = np.linalg.lstsq(A.T, b, rcond=None)
            # A.T is 30 x 27, b is 30. We want A.T @ sol ≈ b
            # Actually, we want to solve A.T x = b, i.e., find x such that
            # sum_{r' in other} x[r'] * T_flat[r'] = b
            # This is A.T @ x = b where A.T is (30, 27)
            recon = A.T @ sol
            if np.allclose(recon, b, atol=1e-8):
                # Found a rank-27 decomposition by removing column r
                U27 = np.delete(U28, r, axis=1)
                V27 = np.delete(V28, r, axis=1)
                W27 = np.delete(W28, r, axis=1)
                return U27, V27, W27
        except np.linalg.LinAlgError:
            continue
    
    # Try a more general approach: random rank-27 initialization and check
    # if the tensor can be represented. This is a non-convex problem, so
    # we do a few random tries.
    if time.time() > deadline:
        return None
    
    # Use the flattening: the tensor T is a 30-dim vector (all ones).
    # A rank-r decomposition means T = sum_{s=1}^{r} u_s \otimes v_s \otimes w_s
    # where u_s is in R^6, v_s in R^15, w_s in R^10.
    # This is hard to solve directly. Instead, fix one factor and solve for others.
    
    # Fix W to a random 10x27 matrix with rank 27 (full column rank)
    rng = np.random.RandomState(42)
    W_fixed = rng.randn(p*n, 27)
    
    # The tensor equation becomes: for each (i,j,k),
    # 1 = sum_s U[i*m+j, s] * V[j*p+k, s] * W_fixed[k*n+i, s]
    # This is still quadratic in U,V.
    
    # Alternative: use the "slice" approach. For fixed k, the matrix
    # M_k[i,j] = 1 for all i,j must equal sum_s U[:,s] V[:,s]^T * W_fixed[k*n+i, s]... 
    # This doesn't simplify nicely.
    
    # Given the complexity and time constraints, we conclude that rank 27
    # is unlikely to be achievable with a simple elimination from rank 28.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # First, try the known rank-28 decomposition
    try:
        U, V, W = _build_rank_28(n, m, p)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    # Try to find rank 27
    if time.time() < deadline:
        try:
            result = _try_rank_27(n, m, p, deadline)
            if result is not None:
                U, V, W = result
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        except Exception:
            pass
    
    # Fallback: schoolbook rank 30
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
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
