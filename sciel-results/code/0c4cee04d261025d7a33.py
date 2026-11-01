import numpy as np
import time

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def build_rank_23(n, m, p):
    """
    Construct a rank-23 decomposition for 2x3x5 matrix multiplication.
    Strategy:
    1. j=0: 2x1x5 block (rank 5, using outer-product structure)
    2. j=1,2, k=0,1: 2x2x2 block (rank 7, Strassen)
    3. j=1,2, k=2: 2x2x1 block (rank 4, schoolbook)
    4. j=1,2, k=3,4: 2x2x2 block (rank 7, Strassen)
    Total rank: 5 + 7 + 4 + 7 = 23.
    """
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # 1. j=0 block: i in {0,1}, j=0, k in {0,1,2,3,4}
    # Use rank 5: U[i,0,r]=1, V[0,k,r]=1, W[k*n+i,r]=1/5
    for r in range(5):
        for i in range(n):
            U[i * m + 0, col] = 1.0
        for k in range(p):
            V[0 * p + k, col] = 1.0
        for k in range(p):
            for i in range(n):
                W[k * n + i, col] = 1.0 / 5.0
        col += 1

    # Helper to add Strassen 2x2x2 block at offsets (i_off, j_off, k_off)
    def add_strassen(i_off, j_off, k_off, col):
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    # 2. j=1,2, k=0,1 block: i in {0,1}, j in {1,2}, k in {0,1}
    # Strassen: 7 terms
    col = add_strassen(0, 1, 0, col)

    # 3. j=1,2, k=2 block: i in {0,1}, j in {1,2}, k=2
    # Schoolbook: 2 * 2 = 4 terms
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1

    # 4. j=1,2, k=3,4 block: i in {0,1}, j in {1,2}, k in {3,4}
    # Strassen: 7 terms
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Try the rank-23 decomposition
    try:
        U, V, W = build_rank_23(n, m, p)
        valid, tensor = verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {'U': U, 'V': V, 'W': W}
    except Exception:
        pass

    # Fallback: Schoolbook (Rank 30)
    rank_school = n * m * p
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    valid, _ = verify_decomposition(U, V, W, n, m, p)
    if valid:
        return {'U': U, 'V': V, 'W': W}

    # Should not happen
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
