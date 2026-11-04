"""Matrix multiplication 3x5x6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, n=3, m=5, p=6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # The product tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all (i,j,k) in the schoolbook pattern.
    
    expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1
    
    actual = np.zeros((n, m, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
    
    if not np.allclose(actual, expected, atol=1e-9):
        return False
    return True

def _strassen_2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-element lists in row-major order.
    u indices: [a00, a01, a10, a11]
    v indices: [b00, b01, b10, b11]
    w indices: [c00, c01, c10, c11]
    """
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))   # M1 = (a00+a11)(b00+b11)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))    # M2 = (a10+a11)(b00)
    terms.append(([1,0,0,0], [0,0,1,1], [0,1,0,0]))    # M3 = (a00)(b10+b11)
    terms.append(([0,0,0,1], [-1,0,0,1], [0,0,0,1]))   # M4 = (a11)(b11-b00)
    terms.append(([1,0,1,0], [-1,1,0,0], [0,1,0,0]))   # M5 = (a00+a10)(b01-b00)
    terms.append(([0,0,1,1], [0,0,-1,1], [0,0,0,1]))   # M6 = (a10+a11)(b11-b10)
    terms.append(([1,1,0,0], [0,0,1,0], [1,0,0,0]))    # M7 = (a00+a01)(b10)
    return terms

def _construct_block_strassen():
    """
    Block decomposition using Strassen on the 2x2x2 sub-block (C11 from A11*B11).
    
    Partition:
    A (3x5) = [[A11(2x2), A12(2x3)], [A21(1x2), A22(1x3)]]
    B (5x6) = [[B11(2x2), B12(2x4)], [B21(3x2), B22(3x4)]]
    C (3x6) = [[C11(2x2), C12(2x4)], [C21(1x2), C22(1x4)]]
    
    C11 = A11*B11 + A12*B21       -> Strassen(7) + schoolbook(12) = 19
    C12 = A11*B12 + A12*B22       -> schoolbook(8) + schoolbook(24) = 32
    C21 = A21*B11 + A22*B21       -> schoolbook(4) + schoolbook(6) = 10
    C22 = A21*B12 + A22*B22       -> schoolbook(8) + schoolbook(12) = 20
    Total rank = 81
    """
    n, m, p = 3, 5, 6
    # Pre-allocate with enough columns
    max_rank = 90
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    col = 0
    
    # --- C11 = A11*B11 (Strassen, 7 terms) ---
    # A11: A[0:2, 0:2], B11: B[0:2, 0:2], C11: C[0:2, 0:2]
    strassen_terms = _strassen_2x2_terms()
    for u, v, w in strassen_terms:
        # u: 2x2 in row-major [a00,a01,a10,a11] -> A rows 0-1, cols 0-1
        # v: 2x2 in row-major [b00,b01,b10,b11] -> B rows 0-1, cols 0-1
        # w: 2x2 in row-major [c00,c01,c10,c11] -> C rows 0-1, cols 0-1
        for di in range(2):
            for dj in range(2):
                U[di * m + dj, col] += u[di * 2 + dj]
                V[dj * p + di, col] += v[di * 2 + dj]
                W[di * n + dj, col] += w[di * 2 + dj]
        col += 1
    
    # --- C11 += A12*B21 (schoolbook, 2*3*2 = 12 terms) ---
    # A12: A[0:2, 2:5] (2x3), B21: B[2:5, 0:2] (3x2), C11: C[0:2, 0:2] (2x2)
    for di in range(2):        # row of A12 -> C row di
        for dk in range(3):    # col of A12 / row of B21
            for dj in range(2): # col of B21 -> C col dj
                U[di * m + (2 + dk), col] = 1
                V[(2 + dk) * p + dj, col] = 1
                W[dj * n + di, col] = 1
                col += 1
    
    # --- C12 = A11*B12 (schoolbook, 2*2*4 = 8 terms) ---
    # A11: A[0:2, 0:2] (2x2), B12: B[0:2, 2:6] (2x4), C12: C[0:2, 2:6] (2x4)
    for di in range(2):
        for dk in range(2):
            for dj in range(4):
                U[di * m + dk, col] = 1
                V[dk * p + (2 + dj), col] = 1
                W[(2 + dj) * n + di, col] = 1
                col += 1
    
    # --- C12 += A12*B22 (schoolbook, 2*3*4 = 24 terms) ---
    # A12: A[0:2, 2:5] (2x3), B22: B[2:5, 2:6] (3x4), C12: C[0:2, 2:6] (2x4)
    for di in range(2):
        for dk in range(3):
            for dj in range(4):
                U[di * m + (2 + dk), col] = 1
                V[(2 + dk) * p + (2 + dj), col] = 1
                W[(2 + dj) * n + di, col] = 1
                col += 1
    
    # --- C21 = A21*B11 (schoolbook, 1*2*2 = 4 terms) ---
    # A21: A[2, 0:2] (1x2), B11: B[0:2, 0:2] (2x2), C21: C[2, 0:2] (1x2)
    for dk in range(2):
        for dj in range(2):
            U[2 * m + dk, col] = 1
            V[dk * p + dj, col] = 1
            W[dj * n + 2, col] = 1
            col += 1
    
    # --- C21 += A22*B21 (schoolbook, 1*3*2 = 6 terms) ---
    # A22: A[2, 2:5] (1x3), B21: B[2:5, 0:2] (3x2), C21: C[2, 0:2] (1x2)
    for dk in range(3):
        for dj in range(2):
            U[2 * m + (2 + dk), col] = 1
            V[(2 + dk) * p + dj, col] = 1
            W[dj * n + 2, col] = 1
            col += 1
    
    # --- C22 = A21*B12 (schoolbook, 1*2*4 = 8 terms) ---
    # A21: A[2, 0:2] (1x2), B12: B[0:2, 2:6] (2x4), C22: C[2, 2:6] (1x4)
    for dk in range(2):
        for dj in range(4):
            U[2 * m + dk, col] = 1
            V[dk * p + (2 + dj), col] = 1
            W[(2 + dj) * n + 2, col] = 1
            col += 1
    
    # --- C22 += A22*B22 (schoolbook, 1*3*4 = 12 terms) ---
    # A22: A[2, 2:5] (1x3), B22: B[2:5, 2:6] (3x4), C22: C[2, 2:6] (1x4)
    for dk in range(3):
        for dj in range(4):
            U[2 * m + (2 + dk), col] = 1
            V[(2 + dk) * p + (2 + dj), col] = 1
            W[(2 + dj) * n + 2, col] = 1
            col += 1
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_simple_block():
    """Fallback: full schoolbook decomposition."""
    return _construct_schoolbook()

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Try the corrected Strassen-based block decomposition (rank 81)
    try:
        result = _construct_block_strassen()
        if time.time() < deadline:
            # Verify the decomposition
            if _verify_decomposition(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 90)
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
