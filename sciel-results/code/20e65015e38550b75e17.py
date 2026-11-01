"""Rank-42 exact tensor decomposition for 2x4 @ 4x6 matrix multiplication.

Structural approach: block-partition B into 3 column-blocks of width 2.
For each block r, compute C_r = A1 * B_r_top + A2 * B_r_bot where
  A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
  B_r_top = B[0:2, 2r:2r+2] (2x2), B_r_bot = B[2:4, 2r:2r+2] (2x2)
Each C_r is 2x2. Use Strassen (rank 7) for each 2x2x2 product.
Total rank = 3 blocks * 2 sub-blocks * 7 Strassen = 42.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def _strassen_coefficients():
    """Return the 7 Strassen product specifications for 2x2x2.
    Each product is (U_terms, V_terms) where each term is (index, coeff).
    U_terms: list of (i, j, coeff) for A[i,j]
    V_terms: list of (j, k, coeff) for B[j,k]
    Returns also W_terms: list of (i, k, coeff) for C[i,k]
    """
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    products = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)],
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(1,0,1), (1,1,-1)],
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(0,1,1), (1,1,1)],
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (1,0,1)],
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (0,1,1)],
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)],
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)],
        },
    ]
    return products


def _build_rank42_block():
    """Build rank-42 decomposition using block-partition + Strassen.
    
    For each output column-block r (0,1,2):
      C_r[i, kb] = sum_{j=0}^{1} A1[i,j] * B_r_top[j,kb] + sum_{j=0}^{1} A2[i,j] * B_r_bot[j,kb]
    
    A1 = A[:, 0:2], A2 = A[:, 2:4]
    B_r_top = B[0:2, 2r:2r+2], B_r_bot = B[2:4, 2r:2r+2]
    
    Global indexing:
      U row: i * 4 + a  (i in 0..1, a in 0..3)
      V row: a * 6 + b  (a in 0..3, b in 0..5)
      W row: b * 2 + i  (b in 0..5, i in 0..1)  [trace convention: W[k*n+i]]
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen = _strassen_coefficients()
    
    col = 0
    for r in range(3):
        # Column block r: B columns are 2r, 2r+1
        # Two sub-blocks: A1 (A cols 0,1) with B rows 0,1; A2 (A cols 2,3) with B rows 2,3
        
        for sub in range(2):
            # sub=0: A1 (cols 0,1 of A), B rows 0,1
            # sub=1: A2 (cols 2,3 of A), B rows 2,3
            a_offset = sub * 2  # offset into A columns
            b_row_offset = sub * 2  # offset into B rows
            
            for prod_idx in range(7):
                prod = strassen[prod_idx]
                
                # U: A[i, a_offset + j] -> row = i * 4 + (a_offset + j)
                for (i, j, coeff) in prod['U']:
                    U[i * m + (a_offset + j), col] += coeff
                
                # V: B[b_row_offset + j, 2r + kb] -> row = (b_row_offset + j) * 6 + (2r + kb)
                for (j, kb, coeff) in prod['V']:
                    V[(b_row_offset + j) * p + (2 * r + kb), col] += coeff
                
                # W: C[i, 2r + kb] -> row = (2r + kb) * n + i
                for (i, kb, coeff) in prod['W']:
                    W[(2 * r + kb) * n + i, col] += coeff
                
                col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None


def _to_output(mat):
    """Convert numpy array to list of lists with int where possible."""
    return [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in mat]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii * m + jj][col] = 1
                V[jj * p + kk][col] = 1
                W[kk * n + ii][col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Build rank-42 block-partition + Strassen decomposition
    U, V, W = _build_rank42_block()

    # Verify correctness
    valid, info = _verify(U, V, W, n, m, p)
    if valid:
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}

    # Check if close enough (numerical tolerance)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(U.shape[1]):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                max_err = max(max_err, abs(val - 1.0))

    if max_err < 1e-6:
        return {"U": _to_output(U), "V": _to_output(V), "W": _to_output(W)}

    # Fallback: schoolbook rank 48
    if time.time() > deadline - 5.0:
        U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}
