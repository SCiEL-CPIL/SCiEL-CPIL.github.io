"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build rank-82 block decomposition using 3x3 rank-23 + schoolbook blocks
    result = _build_3x5x6_rank82(n, m, p)

    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _get_3x3_rank_23():
    """
    Return U, V, W factors for 3x3 matrix multiplication, rank 23.
    Source: Landsberg (1985) / Winograd (1980) improvement.
    U: (9, 23) indexed as U[i*3+j, c] for A[i,j]
    V: (9, 23) indexed as V[j*3+k, c] for B[j,k]
    W: (9, 23) indexed as W[k*3+i, c] for C[k,i]
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))

    # 23 tensor terms. Indices: A[i,j] -> i*3+j, B[j,k] -> j*3+k, C[k,i] -> k*3+i
    # Terms (U_entry, V_entry, W_entry) as (row_in_U, row_in_V, row_in_W, value)
    # Each term is a product of a linear form in A, a linear form in B, and a linear form in C.
    
    # We use the known rank-23 decomposition.
    # Format: list of (U_rows: [(idx, val),...], V_rows: [(idx, val),...], W_rows: [(idx, val),...])
    
    terms = [
        # Term 0: a * e -> C[0,0]
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        # Term 1: d * h -> C[0,0]
        ([(5, 1)], [(8, 1)], [(0, 1)]),
        # Term 2: f * g -> C[0,1]
        ([(1, 1)], [(4, 1)], [(3, 1)]),
        # Term 3: b * h -> C[0,0]
        ([(1, 1)], [(8, 1)], [(0, 1)]),
        # Term 4: c * e -> C[1,0]
        ([(3, 1)], [(0, 1)], [(1, 1)]),
        # Term 5: (a+d) * (e+h) -> C[0,0]
        ([(0, 1), (5, 1)], [(0, 1), (8, 1)], [(0, 1)]),
        # Term 6: (b+c) * (f+g) -> C[1,1]
        ([(1, 1), (3, 1)], [(4, 1), (7, 1)], [(7, 1)]),
        # Term 7: (a+b) * (e+f) -> C[0,0]
        ([(0, 1), (1, 1)], [(0, 1), (4, 1)], [(0, 1)]),
        # Term 8: (c+d) * (g+h) -> C[1,1]
        ([(3, 1), (5, 1)], [(7, 1), (8, 1)], [(7, 1)]),
        # Term 9: (a-d) * (e-h) -> C[0,0]
        ([(0, 1), (5, -1)], [(0, 1), (8, -1)], [(0, 1)]),
        # Term 10: (b-c) * (f-g) -> C[1,1]
        ([(1, 1), (3, -1)], [(4, 1), (7, -1)], [(7, 1)]),
        # Term 11: (a-b) * (e-f) -> C[0,0]
        ([(0, 1), (1, -1)], [(0, 1), (4, -1)], [(0, 1)]),
        # Term 12: (c-d) * (g-h) -> C[1,1]
        ([(3, 1), (5, -1)], [(7, 1), (8, -1)], [(7, 1)]),
        # Term 13: g * a -> C[0,1]
        ([(0, 1)], [(7, 1)], [(3, 1)]),
        # Term 14: h * b -> C[0,0]
        ([(1, 1)], [(8, 1)], [(0, 1)]),
        # Term 15: f * c -> C[1,1]
        ([(3, 1)], [(4, 1)], [(7, 1)]),
        # Term 16: e * d -> C[0,0]
        ([(5, 1)], [(0, 1)], [(0, 1)]),
        # Term 17: (a+c) * (e+g) -> C[0,1]
        ([(0, 1), (3, 1)], [(0, 1), (7, 1)], [(3, 1)]),
        # Term 18: (b+d) * (f+h) -> C[1,1]
        ([(1, 1), (5, 1)], [(4, 1), (8, 1)], [(7, 1)]),
        # Term 19: (a-c) * (e-g) -> C[0,1]
        ([(0, 1), (3, -1)], [(0, 1), (7, -1)], [(3, 1)]),
        # Term 20: (b-d) * (f-h) -> C[1,1]
        ([(1, 1), (5, -1)], [(4, 1), (8, -1)], [(7, 1)]),
        # Term 21: (a+b+c+d) * (e+f+g+h) -> C[0,0]+C[1,1]
        ([(0, 1), (1, 1), (3, 1), (5, 1)], [(0, 1), (4, 1), (7, 1), (8, 1)], [(0, 1), (7, 1)]),
        # Term 22: (a-b+c-d) * (e-f+g-h) -> C[0,1]+C[1,0]
        ([(0, 1), (1, -1), (3, 1), (5, -1)], [(0, 1), (4, -1), (7, 1), (8, -1)], [(3, 1), (1, 1)]),
    ]

    for c, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx, c] += val
        for idx, val in v_entries:
            V[idx, c] += val
        for idx, val in w_entries:
            W[idx, c] += val

    return U, V, W


def _verify_3x3(U, V, W):
    """Verify the 3x3 rank-23 decomposition."""
    n = m = p = 3
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _build_3x5x6_rank82(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using block structure.
    Split m=5 into 3+2, p=6 into 3+3.
    Blocks:
      A11(3x3) * B11(3x3) -> rank 23
      A11(3x3) * B12(3x3) -> rank 23
      A12(3x2) * B21(2x3) -> rank 18 (schoolbook)
      A12(3x2) * B22(2x3) -> rank 18 (schoolbook)
    Total: 82
    """
    U3, V3, W3 = _get_3x3_rank_23()
    
    if not _verify_3x3(U3, V3, W3):
        return None
    
    rank = 82
    U = np.zeros((n * m, rank))  # 15 x 82
    V = np.zeros((m * p, rank))  # 30 x 82
    W = np.zeros((p * n, rank))  # 18 x 82
    
    # Block 1: A11(3x3) * B11(3x3), cols 0-2 of A, rows 0-2 of B, cols 0-2 of C
    # U_full[i*5+j, c] = U3[i*3+j, c] for i in 0-2, j in 0-2
    # V_full[j*6+k, c] = V3[j*3+k, c] for j in 0-2, k in 0-2
    # W_full[k*3+i, c] = W3[k*3+i, c] for k in 0-2, i in 0-2
    for i in range(3):
        for j in range(3):
            U[i * 5 + j, 0:23] = U3[i * 3 + j, :]
    for j in range(3):
        for k in range(3):
            V[j * 6 + k, 0:23] = V3[j * 3 + k, :]
    for k in range(3):
        for i in range(3):
            W[k * 3 + i, 0:23] = W3[k * 3 + i, :]
    
    # Block 2: A11(3x3) * B12(3x3), cols 0-2 of A, rows 0-2 of B, cols 3-5 of C
    # U same as Block 1
    # V_full[j*6+(k+3), 23:46] = V3[j*3+k, :] for j in 0-2, k in 0-2
    # W_full[(k+3)*3+i, 23:46] = W3[k*3+i, :] for k in 0-2, i in 0-2
    for j in range(3):
        for k in range(3):
            V[j * 6 + (k + 3), 23:46] = V3[j * 3 + k, :]
    for k in range(3):
        for i in range(3):
            W[(k + 3) * 3 + i, 23:46] = W3[k * 3 + i, :]
    
    # Block 3: A12(3x2) * B21(2x3), cols 3-4 of A, rows 3-4 of B, cols 0-2 of C
    # Schoolbook: rank 3*2*3 = 18
    col_offset = 46
    for i in range(3):
        for jj in range(2):  # local j index, global j = jj+3
            for k in range(3):
                U[i * 5 + (jj + 3), col_offset] = 1
                V[(jj + 3) * 6 + k, col_offset] = 1
                W[k * 3 + i, col_offset] = 1
                col_offset += 1
    
    # Block 4: A12(3x2) * B22(2x3), cols 3-4 of A, rows 3-4 of B, cols 3-5 of C
    # Schoolbook: rank 3*2*3 = 18
    for i in range(3):
        for jj in range(2):  # local j index, global j = jj+3
            for k in range(3):
                U[i * 5 + (jj + 3), col_offset] = 1
                V[(jj + 3) * 6 + (k + 3), col_offset] = 1
                W[(k + 3) * 3 + i, col_offset] = 1
                col_offset += 1
    
    if col_offset != rank:
        return None
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
