import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _elapsed(start):
    return time.time() - start

def _time_left(start, deadline=145.0):
    return deadline - _elapsed(start)

def _verify(U, V, W, n=3, m=3, p=8, seed=42):
    """Verify if U, V, W decompose the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    np.random.seed(seed)
    A = np.random.randint(-10, 10, (n, m)).astype(float)
    B = np.random.randint(-10, 10, (m, p)).astype(float)
    C = A @ B
    
    C_decomp = np.zeros((n, p))
    for c in range(U.shape[1]):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        
        XY = (u_c * A) @ (v_c * B)
        C_decomp += XY * w_c.T
    
    return np.allclose(C, C_decomp, atol=1e-6)

def _construct_rank_68(n=3, m=3, p=8):
    rank = 68
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, :]
    # This is 2x2 * 2x8.
    # We split B[0:2, :] into four 2x2 blocks.
    # For each block, use Strassen (rank 7).
    # Total 4 * 7 = 28 terms.
    
    # Strassen terms for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5 + M6
    # C10 = M2 + M4 + M7
    # C11 = M1 - M3 + M7
    
    for k0 in [0, 2, 4, 6]:
        # M1
        U[0*2 + 0, col] = 1  # A00
        U[1*2 + 1, col] = 1  # A11
        V[0*8 + k0, col] = 1  # B00
        V[1*8 + (k0+1), col] = 1  # B11
        W[k0*3 + 0, col] = 1  # C00
        W[(k0+1)*3 + 1, col] = 1  # C11
        col += 1
        
        # M2
        U[1*2 + 0, col] = 1  # A10
        U[1*2 + 1, col] = 1  # A11
        V[0*8 + k0, col] = 1  # B00
        W[k0*3 + 1, col] = 1  # C10
        col += 1
        
        # M3
        U[0*2 + 0, col] = 1  # A00
        V[0*8 + (k0+1), col] = 1  # B01
        V[1*8 + (k0+1), col] = -1  # B11
        W[(k0+1)*3 + 0, col] = 1  # C01
        col += 1
        
        # M4
        U[1*2 + 1, col] = 1  # A11
        V[1*8 + k0, col] = 1  # B10
        V[0*8 + k0, col] = -1  # B00
        W[k0*3 + 1, col] = 1  # C10
        col += 1
        
        # M5
        U[0*2 + 0, col] = 1  # A00
        U[0*2 + 1, col] = 1  # A01
        V[1*8 + (k0+1), col] = 1  # B11
        W[(k0+1)*3 + 0, col] = 1  # C01
        col += 1
        
        # M6
        U[1*2 + 0, col] = 1  # A10
        U[0*2 + 0, col] = -1  # A00
        V[0*8 + k0, col] = 1  # B00
        V[0*8 + (k0+1), col] = 1  # B01
        W[k0*3 + 0, col] = 1  # C00
        W[k0*3 + 1, col] = 1  # C10
        col += 1
        
        # M7
        U[0*2 + 1, col] = 1  # A01
        U[1*2 + 1, col] = -1  # A11
        V[1*8 + k0, col] = 1  # B10
        V[1*8 + (k0+1), col] = 1  # B11
        W[(k0+1)*3 + 0, col] = -1  # C01
        W[(k0+1)*3 + 1, col] = 1  # C11
        col += 1

    # Part 2: A[0:2, 2] * B[2, :]
    # This is 2x1 * 1x8.
    # C0k += A02 * B2k
    # C1k += A12 * B2k
    # Total 16 terms.
    for k in range(8):
        # Term for C0k
        U[0*3 + 2, col] = 1  # A02
        V[2*8 + k, col] = 1  # B2k
        W[k*3 + 0, col] = 1  # C0k
        col += 1
        
        # Term for C1k
        U[1*3 + 2, col] = 1  # A12
        V[2*8 + k, col] = 1  # B2k
        W[k*3 + 1, col] = 1  # C1k
        col += 1

    # Part 3: A[2, 0:2] * B[0:2, :]
    # This is 1x2 * 2x8.
    # C2k += A20 * B0k + A21 * B1k
    # Total 16 terms.
    for k in range(8):
        # Term for A20 * B0k
        U[2*3 + 0, col] = 1  # A20
        V[0*8 + k, col] = 1  # B0k
        W[k*3 + 2, col] = 1  # C2k
        col += 1
        
        # Term for A21 * B1k
        U[2*3 + 1, col] = 1  # A21
        V[1*8 + k, col] = 1  # B1k
        W[k*3 + 2, col] = 1  # C2k
        col += 1

    # Part 4: A[2, 2] * B[2, :]
    # This is 1x1 * 1x8.
    # C2k += A22 * B2k
    # Total 8 terms.
    for k in range(8):
        U[2*3 + 2, col] = 1  # A22
        V[2*8 + k, col] = 1  # B2k
        W[k*3 + 2, col] = 1  # C2k
        col += 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 145.0
    
    # Construct the rank 68 decomposition
    U, V, W = _construct_rank_68(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
