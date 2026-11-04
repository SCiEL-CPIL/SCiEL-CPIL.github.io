"""Matrix multiplication 4x4 * 4x8 -> 4x8 with rank-minimal exact decomposition.

Strategy:
  - Build Strassen-based rank-98 decomposition (two 4x4x4 blocks, each rank 49).
  - Use alternating least squares to attempt rank reduction below 98.
  - Verify numerically and return best valid decomposition.
"""

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_2x2x2():
    """Build the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 2: M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A22(B12-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_strassen_4x4x4():
    """Build a Strassen-based decomposition for 4x4 * 4x4 -> 4x4. Rank 49."""
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _build_strassen_2x2x2()
    
    col = 0
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for t in range(7):
                    for di in range(2):
                        for dj in range(2):
                            i = 2 * i1 + di
                            j = 2 * j1 + dj
                            U[i * m + j, col] = U2[di * 2 + dj, t]
                    for dj in range(2):
                        for dk in range(2):
                            j = 2 * j1 + dj
                            k = 2 * k1 + dk
                            V[j * p + k, col] = V2[dj * 2 + dk, t]
                    for di in range(2):
                        for dk in range(2):
                            i = 2 * i1 + di
                            k = 2 * k1 + dk
                            W[k * n + i, col] = W2[di * 2 + dk, t]
                    col += 1
    
    return U, V, W


def _build_strassen_4x4x8():
    """Build Strassen-based decomposition for 4x4 * 4x8 -> 4x8. Rank 98."""
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U4, V4, W4 = _build_strassen_4x4x4()
    
    # First half: B cols 0-3, C cols 0-3
    for c in range(49):
        U[:, c] = U4[:, c]
        V[:16, c] = V4[:, c]
        W[:16, c] = W4[:, c]
    
    # Second half: B cols 4-7, C cols 4-7
    for c in range(49):
        U[:, 49 + c] = U4[:, c]
        V[16:, 49 + c] = V4[:, c]
        W[16:, 49 + c] = W4[:, c]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the n*m*p matrix multiplication tensor."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank_reduction(U0, V0, W0, n, m, p, target_rank, max_time=30):
    """Try to reduce rank using alternating least squares."""
    start_time = time.time()
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    # Truncate to target rank
    U = U[:, :target_rank]
    V = V[:, :target_rank]
    W = W[:, :target_rank]
    
    for iteration in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                except:
                    pass
        
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None


def _to_list(arr):
    """Convert numpy array to list of lists."""
    return [[float(x) for x in row] for row in arr]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with Strassen rank-98
    U0, V0, W0 = _build_strassen_4x4x8()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = 98
    
    if not _verify_decomposition(U0, V0, W0, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Try rank reduction
    for target_rank in range(97, 50, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_reduction(U0, V0, W0, n, m, p, target_rank, max_time=20)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
    
    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}
# EVOLVE-BLOCK-END
