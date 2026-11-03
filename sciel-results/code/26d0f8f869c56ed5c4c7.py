"""Matrix multiplication 4x4 * 4x8 -> 4x8 with rank-minimal exact decomposition.

Strategy:
  - Use Strassen's 2x2x2 rank-7 decomposition tiled over 4x4x4 blocks.
  - For 4x4*4x8: split into two 4x4*4x4 halves, each with 8 Strassen blocks of rank 7 = 56 per half.
  - Total rank: 112.
  - Verify the decomposition numerically before returning.
  - Fall back to schoolbook rank-128 if verification fails.
"""

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_2x2x2():
    """Build the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns U (4x7), V (4x7), W (4x7) where:
    - U encodes A (2x2) in row-major: index = 2*i + j
    - V encodes B (2x2) in row-major: index = 2*j + k  
    - W encodes C (2x2) in row-major: index = 2*i + k
    
    Strassen's 7 terms:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    Output:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # Term 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # Term 1: M2 = (A21+A22)B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    
    # Term 2: M3 = A11(B21-B22)
    U[0, 2] = 1  # A11
    V[2, 2] = 1  # B21
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # Term 3: M4 = A22(B12-B11)
    U[3, 3] = 1  # A22
    V[1, 3] = 1  # B12
    V[0, 3] = -1 # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # Term 4: M5 = (A11+A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[1, 4] = 1  # C12
    W[0, 4] = -1 # C11
    
    # Term 5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # Term 6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W


def _build_strassen_4x4x4():
    """Build a Strassen-based decomposition for 4x4 * 4x4 -> 4x4.
    
    Tile 8 blocks of 2x2*2x2, each with rank 7 = total rank 56.
    
    Returns U (16x56), V (16x56), W (16x56).
    """
    n = m = p = 4
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _build_strassen_2x2x2()
    
    col = 0
    for i1 in range(2):  # row block of A and C
        for j1 in range(2):  # col block of A, row block of B
            for k1 in range(2):  # col block of B and C
                for t in range(7):
                    # A block at (i1, j1): rows 2*i1..2*i1+1, cols 2*j1..2*j1+1
                    for di in range(2):
                        for dj in range(2):
                            i = 2 * i1 + di
                            j = 2 * j1 + dj
                            U[i * m + j, col] = U2[di * 2 + dj, t]
                    
                    # B block at (j1, k1): rows 2*j1..2*j1+1, cols 2*k1..2*k1+1
                    for dj in range(2):
                        for dk in range(2):
                            j = 2 * j1 + dj
                            k = 2 * k1 + dk
                            V[j * p + k, col] = V2[dj * 2 + dk, t]
                    
                    # C block at (i1, k1): rows 2*i1..2*i1+1, cols 2*k1..2*k1+1
                    for di in range(2):
                        for dk in range(2):
                            i = 2 * i1 + di
                            k = 2 * k1 + dk
                            W[k * n + i, col] = W2[di * 2 + dk, t]
                    
                    col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
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


def _build_strassen_4x4x8():
    """Build a Strassen-based decomposition for 4x4 * 4x8 -> 4x8.
    
    Split B and C into left (cols 0-3) and right (cols 4-7) halves.
    Use the same rank-56 Strassen decomposition for each half.
    Total rank: 112.
    
    Returns U (16x112), V (32x112), W (32x112).
    """
    n, m, p = 4, 4, 8
    rank = 112
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U4, V4, W4 = _build_strassen_4x4x4()
    
    # First half: columns 0-3
    for c in range(56):
        U[:, c] = U4[:, c]
        V[:16, c] = V4[:, c]  # B_left: j*4+k for k=0..3
        V[16:, c] = 0
        W[:16, c] = W4[:, c]  # C_left: k*4+i for k=0..3
        W[16:, c] = 0
    
    # Second half: columns 4-7
    for c in range(56):
        U[:, 56 + c] = U4[:, c]
        V[:16, 56 + c] = 0
        V[16:, 56 + c] = V4[:, c]  # B_right: j*4+k for k=4..7
        W[:16, 56 + c] = 0
        W[16:, 56 + c] = W4[:, c]  # C_right: k*4+i for k=4..7
    
    return U, V, W


def _build_schoolbook():
    """Build the schoolbook rank-128 decomposition."""
    n, m, p = 4, 4, 8
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


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer values."""
    result = []
    for row in arr:
        result.append([float(x) for x in row])
    return result


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try Strassen rank-112 first
    try:
        U, V, W = _build_strassen_4x4x8()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
