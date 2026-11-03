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
            # candidate redundant: confirm with a global verify
            if _verify(U_o, V_o, W_o, n, m, p, tol):
                U, V, W = U_o, V_o, W_o
                r -= 1
                changed = True
                break
        if not changed:
            break
    return U, V, W


def _try_reduce_pairs(U, V, W, n, m, p, deadline, tol=1e-9):
    """Try to reduce rank by finding linear dependencies among columns.
    
    Uses a faster verification approach that only checks affected tensor entries.
    """
    U, V, W = U.copy(), V.copy(), W.copy()
    r = U.shape[1]
    
    # Precompute which (i,j,k) triples are affected by each column
    # Column c contributes to entry (i,j,k) iff U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] != 0
    
    for c in range(r - 1, -1, -1):
        if time.time() > deadline:
            break
        
        U_o = np.delete(U, c, axis=1)
        V_o = np.delete(V, c, axis=1)
        W_o = np.delete(W, c, axis=1)
        
        # Quick check: if removing column c leaves the tensor unchanged,
        # then column c must have been redundant
        # This is equivalent to checking if the contribution of column c is zero
        # for all (i,j,k)
        
        # Faster verification: only check entries where column c has non-zero contribution
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        
        # Find indices where Uc, Vc, Wc are non-zero
        U_nonzero = np.nonzero(Uc)[0]
        V_nonzero = np.nonzero(Vc)[0]
        W_nonzero = np.nonzero(Wc)[0]
        
        if len(U_nonzero) == 0 or len(V_nonzero) == 0 or len(W_nonzero) == 0:
            # Column is zero, can be removed
            if _verify(U_o, V_o, W_o, n, m, p, tol):
                U, V, W = U_o, V_o, W_o
                r -= 1
                c = r - 1
                continue
        
        # Full verification
        if _verify(U_o, V_o, W_o, n, m, p, tol):
            U, V, W = U_o, V_o, W_o
            r -= 1
            c = r - 1
            continue
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 8

    U2, V2, W2 = _strassen_2x2x2()

    # Tiling strategy: Use Strassen blocks where possible
    # For 3x4x8: split into blocks
    # Rows: 3 = 2 + 1 (one 2x2 Strassen block, one schoolbook row)
    # Cols: 4 = 2 + 2 (two 2x2 Strassen blocks)
    # K: 8 = 2 + 2 + 2 + 2 (four 2x2 Strassen blocks)
    
    # Total rank calculation:
    # 2x2 Strassen blocks: 2 (row blocks) * 2 (col blocks) * 4 (k blocks) * 7 = 112
    # Schoolbook for row 2: 4 (cols) * 8 (k) = 32
    # Total = 144
    
    # But we can do better with a smarter tiling:
    # Use 2x2 Strassen for rows 0,1 and cols 0,1,2,3 with k split into 4 pairs
    # That's 2 col blocks * 4 k blocks * 7 = 56
    # Plus schoolbook for row 2: 4 * 8 = 32
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
        U, V, W = _try_reduce_pairs(U, V, W, n, m, p, deadline)
        
        # Second pass of reduction after first pass may have created new redundancies
        if time.time() < deadline - 2:
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
