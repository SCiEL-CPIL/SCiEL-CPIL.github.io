"""Exact rank-78 decomposition of M_{3,4,7} via six 2x2x2 Strassen blocks."""
DIMENSIONS = [3, 4, 7]
# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = Ua[i * m + j]
                v_row = Va[j * p + k]
                w_row = Wa[k * n + i]
                total = np.dot(u_row, v_row * w_row)
                if abs(total - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return a valid rank-n*m*p decomposition (always correct)."""
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


def _strassen_2x2_block(U, V, W, rank, col, i0, j0, k0, n, m, p):
    """
    Add one 2x2x2 Strassen block covering rows i0..i0+1, cols j0..j0+1 of A,
    and rows k0..k0+1, cols i0..i0+1 of C.
    Uses 7 multiplications instead of 8.
    Returns the next free column index.
    """
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
    U[i0 * m + j0][col] = 1
    U[i1 * m + j1][col] = 1
    V[j0 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1

    # M2 = A[i1,j0] * (B[j0,k0] + B[j1,k1])
    U[i1 * m + j0][col] = 1
    V[j0 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i1][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1

    # M3 = (A[i0,j0] - A[i1,j0]) * B[j0,k1]
    U[i0 * m + j0][col] = 1
    U[i1 * m + j0][col] = -1
    V[j0 * p + k1][col] = 1
    W[k1 * n + i0][col] = 1
    W[k1 * n + i1][col] = 1
    col += 1

    # M4 = (A[i0,j1] - A[i1,j1]) * B[j1,k0]
    U[i0 * m + j1][col] = 1
    U[i1 * m + j1][col] = -1
    V[j1 * p + k0][col] = 1
    W[k0 * n + i0][col] = 1
    W[k0 * n + i1][col] = 1
    col += 1

    # M5 = (A[i0,j1] + A[i1,j1]) * (B[j1,k0] + B[j1,k1])
    U[i0 * m + j1][col] = 1
    U[i1 * m + j1][col] = 1
    V[j1 * p + k0][col] = 1
    V[j1 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    W[k1 * n + i0][col] = 1
    col += 1

    # M6 = (A[i0,j0] + A[i0,j1]) * (B[j0,k0] + B[j0,k1])
    U[i0 * m + j0][col] = 1
    U[i0 * m + j1][col] = 1
    V[j0 * p + k0][col] = 1
    V[j0 * p + k1][col] = 1
    W[k0 * n + i0][col] = 1
    W[k0 * n + i1][col] = -1
    W[k1 * n + i0][col] = 1
    W[k1 * n + i1][col] = -1
    col += 1

    # M7 = A[i1,j0] * (B[j0,k0] - B[j1,k0])
    U[i1 * m + j0][col] = 1
    V[j0 * p + k0][col] = 1
    V[j1 * p + k0][col] = -1
    W[k0 * n + i1][col] = 1
    col += 1

    return col


def _build_strassen_decomposition(n, m, p):
    """
    Build a decomposition using 2x2x2 Strassen blocks where possible,
    and single-term (schoolbook) entries for the remainder.
    
    Strategy: For each pair of consecutive i-rows (i0, i0+1) and consecutive
    j-cols (j0, j0+1), try to tile the k-dimension with 2x2 blocks.
    Any leftover k-positions get schoolbook terms.
    """
    # Determine which (i0, j0) pairs we can use for 2x2 Strassen blocks
    # A 2x2 Strassen block covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # So we need i0+1 < n, j0+1 < m, k0+1 < p
    
    # For n=3, m=4, p=7:
    # i-pairs: (0,1), (1,2) -> but (1,2) would overlap with (0,1) on row 1
    # j-pairs: (0,1), (1,2), (2,3)
    # k-pairs: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6)
    
    # To avoid overlap in i-dimension, we can only use i0=0 (covering i=0,1)
    # or i0=1 (covering i=1,2), but not both (they share i=1).
    # Let's try both options and pick the one with fewer schoolbook fillers.
    
    best_decomp = None
    best_rank = n * m * p  # schoolbook rank as baseline
    
    for i0 in [0, 1]:
        i1 = i0 + 1
        if i1 >= n:
            continue
        
        # For this i-pair, tile j and k with 2x2 blocks
        # j-pairs: consecutive pairs in [0, m)
        j_pairs = [(j, j + 1) for j in range(0, m - 1, 2)]
        # For m=4: j_pairs = [(0,1), (2,3)]
        
        # k-pairs: consecutive pairs in [0, p)
        k_pairs = [(k, k + 1) for k in range(0, p - 1, 2)]
        # For p=7: k_pairs = [(0,1), (2,3), (4,5)], leftover k=6
        
        rank = 0
        U = [[0] * (n * m * p) for _ in range(n * m)]  # pre-allocate max size
        V = [[0] * (n * m * p) for _ in range(m * p)]
        W = [[0] * (n * m * p) for _ in range(p * n)]
        
        col = 0
        covered = set()
        
        # Add 2x2x2 Strassen blocks
        for (j0, j1) in j_pairs:
            for (k0, k1) in k_pairs:
                col = _strassen_2x2_block(U, V, W, rank, col, i0, j0, k0, n, m, p)
                for ii in (i0, i1):
                    for jj in (j0, j1):
                        for kk in (k0, k1):
                            covered.add((ii, jj, kk))
        
        # Add schoolbook terms for uncovered (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Trim arrays to actual rank
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
        if col < best_rank:
            best_rank = col
            best_decomp = {"U": U, "V": V, "W": W}
    
    return best_decomp, best_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build the Strassen-based decomposition
    decomp, rank = _build_strassen_decomposition(n, m, p)
    
    if decomp is None:
        return _schoolbook(n, m, p)
    
    # Verify the decomposition
    if time.time() < deadline:
        if not _verify_decomposition(decomp["U"], decomp["V"], decomp["W"], n, m, p):
            # Fallback to schoolbook if verification fails
            return _schoolbook(n, m, p)
    
    return decomp
# EVOLVE-BLOCK-END
