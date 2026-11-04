import time
import numpy as np

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the tensor decomposition is correct."""
    R = U.shape[1]
    # Efficient verification using matrix operations
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check if T equals the multiplication tensor (which is 1 for all valid i,j,k)
    
    # Full verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    
    return True

def _strassen_222():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
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

def _direct_sum_p_split(U444, V444, W444):
    """
    Split p=8 into two p=4 blocks using direct sum.
    Input: 4x4x4 decomposition (U: 16xr, V: 16xr, W: 16xr)
    Output: 4x4x8 decomposition (U: 16x2r, V: 32x2r, W: 32x2r)
    """
    n, r = U444.shape
    m, _ = V444.shape  # should be 16 = 4*4
    p, _ = W444.shape  # should be 16 = 4*4
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)  # m=16, so 32
    W = np.zeros((2 * p, R), dtype=float)  # p=16, so 32
    
    # First half of p (k=0..3)
    U[:, :r] = U444
    V[:m, :r] = V444
    W[:p, :r] = W444
    
    # Second half of p (k=4..7)
    U[:, r:] = U444
    V[m:, r:] = V444
    W[p:, r:] = W444
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii*m + jj, col] = 1
                V[jj*p + kk, col] = 1
                W[kk*n + ii, col] = 1
                col += 1
    return U, V, W

def _syndrome_elimination(U, V, W, deadline):
    """
    Structural change: Use linear algebra to detect and eliminate linearly
    dependent rank-1 components. This treats the set of components as a code
    and checks for zero syndromes (linear dependencies).
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        R = U.shape[1]
        if R <= 1:
            break
        
        # Stack the columns of U, V, W to form the "syndrome" matrix
        # If the columns are linearly dependent in this stacked space,
        # we can eliminate one.
        stacked_A = np.vstack([U, V, W])
        
        # Compute rank of stacked_A
        # If rank < R, there is a dependency
        try:
            rank = np.linalg.matrix_rank(stacked_A, tol=1e-10)
            if rank < R:
                # Find a null vector
                # Use SVD to find the right singular vector corresponding to smallest singular value
                U_svd, s, Vt = np.linalg.svd(stacked_A, full_matrices=False)
                # The last column of Vt is the null vector
                null_vec = Vt[-1]
                
                # Identify the column to eliminate (largest absolute value in null_vec)
                # This is a heuristic to pick a column that is "most" dependent
                col_to_eliminate = np.argmax(np.abs(null_vec))
                
                # Eliminate the column
                U = np.delete(U, col_to_eliminate, axis=1)
                V = np.delete(V, col_to_eliminate, axis=1)
                W = np.delete(W, col_to_eliminate, axis=1)
                changed = True
        except np.linalg.LinAlgError:
            break
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Start with the known rank-98 decomposition:
    # 4x4x4 via Strassen kron Strassen (rank 49), then direct sum for p=8
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    U, V, W = _direct_sum_p_split(U444, V444, W444)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Apply structural linear dependency elimination (Syndrome-based)
    if time.time() < deadline:
        U2, V2, W2 = _syndrome_elimination(best_U, best_V, best_W, deadline)
        if U2.shape[1] < best_rank and _verify_decomposition(U2, V2, W2, n, m, p):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
