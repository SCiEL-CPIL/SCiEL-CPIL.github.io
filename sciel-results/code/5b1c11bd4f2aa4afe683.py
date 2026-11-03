import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Verified rank-7 Strassen decomposition of the 2x2x2 multiplication tensor.
    U (4x7) indexed by (i,j) in row-major, V (4x7) by (j,k), W (4x7) by (k,i)."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Return True iff sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _remove_zero_cols(U, V, W):
    """Drop columns that are identically zero in all three matrices."""
    keep = []
    for c in range(U.shape[1]):
        if np.any(U[:, c]) or np.any(V[:, c]) or np.any(W[:, c]):
            keep.append(c)
    return U[:, keep], V[:, keep], W[:, keep]


def _tensor_flatten(U, V, W):
    """Flatten the tensor T = sum_c U_c ⊗ V_c ⊗ W_c into a 3D array."""
    n, m, p = U.shape[0], V.shape[0], W.shape[0]
    # Infer dimensions from the matrices
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U[:, c].reshape(n, m), V[:, c].reshape(m, p), W[:, c].reshape(p, n))
    return T


def _try_reduce_aggressive(U, V, W, n, m, p, deadline, tol=1e-9):
    """Aggressive reduction using tensor-level linear algebra."""
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]
    
    # First, try the simple span-based reduction
    changed = True
    while r > 0 and time.time() < deadline:
        changed = False
        for c in range(r - 1, -1, -1):
            if time.time() > deadline:
                break
            U_o = np.delete(U, c, axis=1)
            V_o = np.delete(V, c, axis=1)
            W_o = np.delete(W, c, axis=1)
            U_c = U[:, c]
            V_c = V[:, c]
            W_c = W[:, c]
            # Quick span test
            if np.linalg.matrix_rank(np.hstack([U_o, U_c.reshape(-1, 1)])) == np.linalg.matrix_rank(U_o):
                continue
            if np.linalg.matrix_rank(np.hstack([V_o, V_c.reshape(-1, 1)])) == np.linalg.matrix_rank(V_o):
                continue
            if np.linalg.matrix_rank(np.hstack([W_o, W_c.reshape(-1, 1)])) == np.linalg.matrix_rank(W_o):
                continue
            # Candidate redundant: confirm with a global verify
            if _verify(U_o, V_o, W_o, n, m, p, tol):
                U, V, W = U_o, V_o, W_o
                r -= 1
                changed = True
                break
        if not changed:
            break
    
    # Now try a more sophisticated approach: look for linear dependencies
    # in the tensor space. If we have a set of rank terms, we can check if
    # any linear combination sums to zero.
    r = U.shape[1]
    if r <= 1 or time.time() > deadline:
        return U, V, W
    
    # Build the tensor matrix: each column c gives a vector in R^(n*m*p)
    # T_c = vec(U_c ⊗ V_c ⊗ W_c)
    # If the columns are linearly dependent, we can remove one.
    # But we need to be careful: the tensor must remain exactly the multiplication tensor.
    
    # Alternative: use the fact that for matrix multiplication, the tensor has
    # a specific structure. We can try to find a lower-rank decomposition by
    # solving a linear system.
    
    # Let's try a different approach: iterate and try to replace pairs of columns
    # with a single column that spans their combined effect.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        r = U.shape[1]
        for c1 in range(r):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, r):
                if time.time() > deadline:
                    break
                # Try to see if columns c1 and c2 can be merged
                # by finding a linear combination that preserves the tensor.
                # This is complex, so skip for now.
                pass
        # Fallback: try removing columns in different orders
        for c in range(r - 1, -1, -1):
            if time.time() > deadline:
                break
            U_o = np.delete(U, c, axis=1)
            V_o = np.delete(V, c, axis=1)
            W_o = np.delete(W, c, axis=1)
            if _verify(U_o, V_o, W_o, n, m, p, tol):
                U, V, W = U_o, V_o, W_o
                improved = True
                break
    
    return U, V, W


def _try_reduce(U, V, W, n, m, p, deadline, tol=1e-9):
    """Iteratively remove redundant columns using the exact span criterion."""
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]
    changed = True
    while r > 0 and time.time() < deadline:
        changed = False
        for c in range(r - 1, -1, -1):
            if time.time() > deadline:
                break
            U_o = np.delete(U, c, axis=1)
            V_o = np.delete(V, c, axis=1)
            W_o = np.delete(W, c, axis=1)
            U_c = U[:, c]
            V_c = V[:, c]
            W_c = W[:, c]
            # span test on U rows
            if np.linalg.matrix_rank(np.hstack([U_o, U_c.reshape(-1, 1)])) == np.linalg.matrix_rank(U_o):
                continue
            if np.linalg.matrix_rank(np.hstack([V_o, V_c.reshape(-1, 1)])) == np.linalg.matrix_rank(V_o):
                continue
            if np.linalg.matrix_rank(np.hstack([W_o, W_c.reshape(-1, 1)])) == np.linalg.matrix_rank(W_o):
                continue
            # candidate redundant: confirm with a global verify
            if _verify(U_o, V_o, W_o, n, m, p, tol):
                U, V, W = U_o, V_o, W_o
                r -= 1
                changed = True
                break
        if not changed:
            break
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 8

    U2, V2, W2 = _strassen_2x2x2()

    # Tiling: two 2x2 Strassen blocks over (i in {0,1}, j in {0,1} and {2,3})
    # with all 8 k's handled as four k-pairs, plus schoolbook for i=2.
    # Strassen blocks: 2 j-blocks * 4 k-blocks * rank 7 = 56
    # Schoolbook i=2: 4 j * 8 k = 32
    # Total = 88
    total_rank = 2 * 4 * 7 + 4 * 8
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col = 0
    for j0 in (0, 2):            # two j-blocks of size 2
        for k0 in range(0, 8, 2):  # four k-blocks of size 2
            for c in range(7):
                # U: local (li,lj) -> global i=li, j=j0+lj
                U[0 * m + j0 + 0, col + c] = U2[0][c]
                U[0 * m + j0 + 1, col + c] = U2[1][c]
                U[1 * m + j0 + 0, col + c] = U2[2][c]
                U[1 * m + j0 + 1, col + c] = U2[3][c]
                # V: local (lj, lk) -> global j=j0+lj, k=k0+lk
                V[(j0 + 0) * p + (k0 + 0), col + c] = V2[0][c]
                V[(j0 + 0) * p + (k0 + 1), col + c] = V2[1][c]
                V[(j0 + 1) * p + (k0 + 0), col + c] = V2[2][c]
                V[(j0 + 1) * p + (k0 + 1), col + c] = V2[3][c]
                # W: local (lk, li) -> global k=k0+lk, i=li
                W[(k0 + 0) * n + 0, col + c] = W2[0][c]
                W[(k0 + 1) * n + 0, col + c] = W2[1][c]
                W[(k0 + 0) * n + 1, col + c] = W2[2][c]
                W[(k0 + 1) * n + 1, col + c] = W2[3][c]
            col += 7

    # Schoolbook for i = 2 (all j in 0..3, k in 0..7)
    for j in range(m):
        for k in range(p):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1

    # Verify the construction
    if not _verify(U, V, W, n, m, p):
        # Fallback: full schoolbook rank 96
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Reduce rank
    if time.time() < deadline - 5:
        U, V, W = _remove_zero_cols(U, V, W)
        U, V, W = _try_reduce(U, V, W, n, m, p, deadline)

    # Final safety verify
    if not _verify(U, V, W, n, m, p):
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
