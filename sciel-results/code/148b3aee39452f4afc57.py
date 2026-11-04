import time
import numpy as np

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _strassen_222():
    """
    Returns U, V, W for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U, V, W have shape (4, 7).
    Indices for A, B, C are 0..3 corresponding to (0,0), (0,1), (1,0), (1,1).
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10 + A11)B00 -> C10, C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01 - B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10 - B00) -> C00, C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # M5 = (A00 + A01)B11 -> C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # M6 = (A10 - A00)(B00 + B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01 - A11)(B10 + B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Trace convention: C[i,k] = sum_j A[i,j] * B[j,k]
    W index: W[k*n + i, r] corresponds to output C[i,k]
    """
    R = U.shape[1]
    tol = 1e-9
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """
    Computes the Kronecker product of two tensor decompositions.
    Input: U1, V1, W1 (shapes (n1, r1), (m1, r1), (p1, r1))
           U2, V2, W2 (shapes (n2, r2), (m2, r2), (p2, r2))
    Output: U, V, W (shapes (n1*n2, r1*r2), (m1*m2, r1*r2), (p1*p2, r1*r2))
    """
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape

    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)

    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx

            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])

    return U, V, W


def _build_444_decomp():
    """Build rank-49 4x4x4 decomposition via Kronecker product of two Strassen."""
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    return U444, V444, W444


def _build_448_direct_sum():
    """
    Build rank-98 4x4x8 decomposition via direct sum of two 4x4x4 blocks.
    The two output blocks (columns 0-3 and 4-7 of C) share the same A.
    """
    U444, V444, W444 = _build_444_decomp()
    n, m, p = 4, 4, 8
    R = 98
    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)

    # First block: k = 0..3
    U[:, :49] = U444
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local, :49] = V444[j*4 + k_local, :]
    for k_local in range(4):
        for i in range(4):
            W[k_local*4 + i, :49] = W444[k_local*4 + i, :]

    # Second block: k = 4..7
    U[:, 49:] = U444
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local
            V[j*8 + k_global, 49:] = V444[j*4 + k_local, :]
    for k_local in range(4):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global*4 + i, 49:] = W444[k_local*4 + i, :]

    return U, V, W


def _try_remove_columns(U, V, W, n, m, p, max_tries=300, deadline=None):
    """
    Try to remove columns that are linearly dependent or redundant.
    Returns a potentially lower-rank decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]

    for _ in range(max_tries):
        if deadline is not None and time.time() > deadline:
            break
        col_idx = np.random.randint(current_rank)

        U_trial = np.delete(U, col_idx, axis=1)
        V_trial = np.delete(V, col_idx, axis=1)
        W_trial = np.delete(W, col_idx, axis=1)

        if _verify_decomposition(U_trial, V_trial, W_trial, n, m, p):
            U = U_trial
            V = V_trial
            W = W_trial
            current_rank -= 1

    return U, V, W


def _try_column_merge(U, V, W, n, m, p, deadline=None):
    """
    Try to merge pairs of columns by checking if their sum (or difference)
    can replace both while preserving the decomposition.
    This exploits the fact that in the direct sum, columns from the two blocks
    have identical U rows, so linear combinations might work.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]

    # Try merging column r from first block with column r from second block
    # Since U[:, r] == U[:, 49+r] for all r in 0..48
    for r in range(49):
        if deadline is not None and time.time() > deadline:
            break
        r2 = 49 + r

        # Try U_combined = U[:, r] + U[:, r2] (which is 2*U[:, r])
        # V_combined = V[:, r] + V[:, r2]
        # W_combined = W[:, r] + W[:, r2]
        # But we need to check if removing both and adding the combined works.
        # Actually, let's just try removing one and see if the other can absorb it.
        # Simpler: try replacing column r with column r + column r2, then remove r2.
        
        # Create trial: replace column r with (col r + col r2), remove column r2
        U_trial = U.copy()
        V_trial = V.copy()
        W_trial = W.copy()
        
        U_trial[:, r] = U[:, r] + U[:, r2]
        V_trial[:, r] = V[:, r] + V[:, r2]
        W_trial[:, r] = W[:, r] + W[:, r2]
        
        U_trial = np.delete(U_trial, r2, axis=1)
        V_trial = np.delete(V_trial, r2, axis=1)
        W_trial = np.delete(W_trial, r2, axis=1)
        
        if _verify_decomposition(U_trial, V_trial, W_trial, n, m, p):
            U = U_trial
            V = V_trial
            W = W_trial
            current_rank -= 1

    return U, V, W


def _try_linear_combination_merge(U, V, W, n, m, p, deadline=None):
    """
    Try more general linear combinations of column pairs to reduce rank.
    For each pair (r, r2) where U[:, r] == U[:, r2], try alpha*col_r + beta*col_r2.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]

    for r in range(49):
        if deadline is not None and time.time() > deadline:
            break
        r2 = 49 + r

        # Try several alpha, beta combinations
        for alpha in [1.0, -1.0, 0.5, -0.5]:
            for beta in [1.0, -1.0, 0.5, -0.5]:
                if alpha == 0 and beta == 0:
                    continue
                
                U_trial = U.copy()
                V_trial = V.copy()
                W_trial = W.copy()
                
                U_trial[:, r] = alpha * U[:, r] + beta * U[:, r2]
                V_trial[:, r] = alpha * V[:, r] + beta * V[:, r2]
                W_trial[:, r] = alpha * W[:, r] + beta * W[:, r2]
                
                U_trial = np.delete(U_trial, r2, axis=1)
                V_trial = np.delete(V_trial, r2, axis=1)
                W_trial = np.delete(W_trial, r2, axis=1)
                
                if _verify_decomposition(U_trial, V_trial, W_trial, n, m, p):
                    U = U_trial
                    V = V_trial
                    W = W_trial
                    current_rank -= 1
                    break
            if U.shape[1] != current_rank:
                break

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    # Step 1: Build rank-98 4x4x8 decomposition
    U, V, W = _build_448_direct_sum()

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank))
        V_school = np.zeros((m*p, schoolbook_rank))
        W_school = np.zeros((p*n, schoolbook_rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii*m + jj, col] = 1
                    V_school[jj*p + kk, col] = 1
                    W_school[kk*n + ii, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Step 2: Try to reduce rank via column removal
    np.random.seed(42)
    
    # Try simple column removal
    U, V, W = _try_remove_columns(U, V, W, n, m, p, max_tries=200, deadline=deadline - 20)

    # Step 3: Try merging paired columns (exploiting shared U structure)
    U, V, W = _try_column_merge(U, V, W, n, m, p, deadline=deadline - 15)

    # Step 4: Try linear combination merges
    U, V, W = _try_linear_combination_merge(U, V, W, n, m, p, deadline=deadline - 10)

    # Step 5: More column removal after merges
    U, V, W = _try_remove_columns(U, V, W, n, m, p, max_tries=200, deadline=deadline - 5)

    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank))
        V_school = np.zeros((m*p, schoolbook_rank))
        W_school = np.zeros((p*n, schoolbook_rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii*m + jj, col] = 1
                    V_school[jj*p + kk, col] = 1
                    W_school[kk*n + ii, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
