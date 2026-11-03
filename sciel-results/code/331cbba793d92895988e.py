"""Matrix multiplication 3x4x8 with rank-reduced decomposition.

Strategy:
1. Build a structured decomposition using 2x2x2 Strassen blocks for the
   top 2 rows and schoolbook for the bottom row, giving rank 88.
2. Use vectorized numpy verification and iterative greedy column removal
   to find and eliminate linearly dependent (redundant) rank-1 terms.
3. The key structural insight: when two Strassen blocks share the same
   B-row (j) but different i-rows, their W contributions may be
   linearly dependent, allowing rank reduction.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    Local indices: A: 00,01,10,11; B: 00,01,10,11; C: 00,01,10,11
    """
    # M1 = (A00+A11)(B00+B11) -> C00+C11
    # M2 = (A10+A11)B00 -> C10-C11
    # M3 = A00(B01-B11) -> C01+C11
    # M4 = A11(B10-B00) -> C00+C10
    # M5 = (A00+A01)B11 -> C01-C00
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    return U, V, W


def _build_base_decomposition():
    """Build the base structured decomposition (rank 88).
    
    For each j-block (2 blocks of size 2):
      - Top 2 rows (i=0,1): 4 Strassen 2x2x2 blocks (rank 7 each) = 28
      - Bottom row (i=2): schoolbook 1x2x8 (rank 16)
    Total per j-block: 44, total: 88
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_2x2x2()
    
    col = 0
    for jb in range(2):  # j-block: 0 (j=0,1) or 1 (j=2,3)
        # Top: 2x2x8 using 4 Strassen blocks
        for kb in range(4):  # k-block: 0 (k=0,1), 1 (k=2,3), etc.
            j0 = 2 * jb
            j1 = 2 * jb + 1
            k0 = 2 * kb
            k1 = 2 * kb + 1
            
            for c in range(7):
                # U: rows for A[0,j0], A[0,j1], A[1,j0], A[1,j1]
                U[0 * m + j0, col + c] += U2[0][c]
                U[0 * m + j1, col + c] += U2[1][c]
                U[1 * m + j0, col + c] += U2[2][c]
                U[1 * m + j1, col + c] += U2[3][c]
                
                # V: rows for B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
                V[j0 * p + k0, col + c] += V2[0][c]
                V[j0 * p + k1, col + c] += V2[1][c]
                V[j1 * p + k0, col + c] += V2[2][c]
                V[j1 * p + k1, col + c] += V2[3][c]
                
                # W: rows for C[k0,0], C[k1,0], C[k0,1], C[k1,1]
                W[k0 * n + 0, col + c] += W2[0][c]
                W[k1 * n + 0, col + c] += W2[1][c]
                W[k0 * n + 1, col + c] += W2[2][c]
                W[k1 * n + 1, col + c] += W2[3][c]
            
            col += 7
        
        # Bottom: 1x2x8 schoolbook (rank 16)
        for j in range(2):
            jg = 2 * jb + j
            for k in range(8):
                U[2 * m + jg, col] = 1
                V[jg * p + k, col] = 1
                W[k * n + 2, col] = 1
                col += 1
    
    return U, V, W


def _verify_and_reduce(U, V, W, time_budget=100.0):
    """Verify decomposition and iteratively remove redundant columns.
    
    Uses vectorized numpy operations for speed.
    """
    n, m, p = 3, 4, 8
    rank = U.shape[1]
    
    # Precompute tensor contributions: T[i,j,k,c] = U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Shape: (n, m, p, rank)
    T = np.zeros((n, m, p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
    
    total = T.sum(axis=3)  # shape (n, m, p)
    
    # Verify initial validity
    if not np.allclose(total, 1.0, atol=1e-9):
        print("Initial decomposition invalid!")
        return U, V, W, False
    
    # Iterative greedy removal
    keep = np.ones(rank, dtype=bool)
    start = time.time()
    changed = True
    
    while changed and (time.time() - start < time_budget):
        changed = False
        for c in range(rank):
            if time.time() - start > time_budget:
                break
            if not keep[c]:
                continue
            # Check if column c is redundant
            new_total = total - T[:, :, :, c]
            if np.allclose(new_total, 1.0, atol=1e-9):
                keep[c] = False
                changed = True
                # Update total for subsequent checks
                total = new_total
    
    idx = np.where(keep)[0]
    if len(idx) < rank:
        return U[:, idx], V[:, idx], W[:, idx], True
    return U, V, W, True


def _schoolbook_fallback():
    """Return schoolbook decomposition (rank 96) as ultimate fallback."""
    n, m, p = 3, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _to_clean_list(mat):
    """Convert numpy array to list, rounding to nearest 0.5 if close."""
    mat_rounded = np.round(mat * 2) / 2
    if np.allclose(mat, mat_rounded, atol=1e-9):
        return mat_rounded.tolist()
    return mat.tolist()


def construct():
    start = time.time()
    
    # Build base decomposition
    U, V, W = _build_base_decomposition()
    
    # Verify and reduce
    time_for_reduction = min(100.0, TIME_LIMIT - (time.time() - start) - 5.0)
    U, V, W, valid = _verify_and_reduce(U, V, W, time_budget=time_for_reduction)
    
    if not valid:
        # Fallback to schoolbook
        U, V, W = _schoolbook_fallback()
    
    # Final verification
    n, m, p = 3, 4, 8
    rank = U.shape[1]
    T = np.zeros((n, m, p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
    total = T.sum(axis=3)
    
    if not np.allclose(total, 1.0, atol=1e-9):
        # Ultimate fallback
        U, V, W = _schoolbook_fallback()
    
    return {
        "U": _to_clean_list(U),
        "V": _to_clean_list(V),
        "W": _to_clean_list(W)
    }
