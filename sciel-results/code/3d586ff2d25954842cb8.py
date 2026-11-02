"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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


def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W matrices of shape (4, 7), (4, 7), (4, 7) respectively.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))  # (4, 7)
    V = np.zeros((m * p, rank))  # (4, 7)
    W = np.zeros((p * n, rank))  # (4, 7)
    
    # Index conventions: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # A indices: i in {0,1}, j in {0,1} -> row i*2+j
    # B indices: j in {0,1}, k in {0,1} -> row j*2+k
    # C indices: i in {0,1}, k in {0,1} -> W[k*2+i, c]
    
    # M1 = (A11+A22)*(B11+B22)
    # A11 = A[0,0] = row 0, A22 = A[1,1] = row 3
    # B11 = B[0,0] = row 0, B22 = B[1,1] = row 3
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0] (k=0, i=0)
    W[3, 0] = 1  # C[1,1] (k=1, i=1)
    
    # M2 = (A21+A22)*B11
    # A21 = A[1,0] = row 2, A22 = A[1,1] = row 3
    # B11 = B[0,0] = row 0
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[0, 1] = 0  # C[0,0]
    W[2, 1] = 1  # C[1,0] (k=0, i=1)
    
    # M3 = A11*(B12-B22)
    # A11 = A[0,0] = row 0
    # B12 = B[0,1] = row 1, B22 = B[1,1] = row 3
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1 # B[1,1]
    W[1, 2] = 1  # C[0,1] (k=1, i=0)
    W[3, 2] = 1  # C[1,1] (k=1, i=1)
    
    # M4 = A22*(B21-B11)
    # A22 = A[1,1] = row 3
    # B21 = B[1,0] = row 2, B11 = B[0,0] = row 0
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1 # B[0,0]
    W[0, 3] = 1  # C[0,0] (k=0, i=0)
    W[2, 3] = 1  # C[1,0] (k=0, i=1)
    
    # M5 = (A11+A12)*B22
    # A11 = A[0,0] = row 0, A12 = A[0,1] = row 1
    # B22 = B[1,1] = row 3
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1 # C[0,0] (k=0, i=0)
    W[1, 4] = 1  # C[0,1] (k=1, i=0)
    
    # M6 = (A21-A11)*(B11+B12)
    # A21 = A[1,0] = row 2, A11 = A[0,0] = row 0
    # B11 = B[0,0] = row 0, B12 = B[0,1] = row 1
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1 # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[3, 5] = 1  # C[1,1] (k=1, i=1)
    
    # M7 = (A12-A22)*(B21+B22)
    # A12 = A[0,1] = row 1, A22 = A[1,1] = row 3
    # B21 = B[1,0] = row 2, B22 = B[1,1] = row 3
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1 # A[1,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[0, 6] = 1  # C[0,0] (k=0, i=0)
    W[1, 6] = 0  # C[0,1]
    W[3, 6] = 1  # C[1,1] (k=1, i=1)
    
    return U, V, W


def _build_decomposition():
    """Build the rank-44 decomposition for 2x3x8 matrix multiplication.
    
    Strategy: Split m=3 into m1=1 and m2=2.
    - 2x1x8 part: rank 16 (schoolbook)
    - 2x2x8 part: Use Strassen on each 2x2x2 block.
      2x2x8 = 2x2x2 * identity(4) in some sense, but more precisely:
      We split the 8 columns into 4 groups of 2.
      Each 2x2x2 block uses Strassen (rank 7), total 4*7 = 28.
      Total: 16 + 28 = 44.
    """
    n, m, p = 2, 3, 8
    m1, m2 = 1, 2
    
    rank = 16 + 28  # 44
    U = np.zeros((n * m, rank))  # (6, 44)
    V = np.zeros((m * p, rank))  # (24, 44)
    W = np.zeros((p * n, rank))  # (16, 44)
    
    col = 0
    
    # Part 1: 2x1x8 (schoolbook), j=0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1  # A[i, 0]
            V[0 * p + k, col] = 1  # B[0, k]
            W[k * n + i, col] = 1  # C[i, k]
            col += 1
    
    # Part 2: 2x2x8 using Strassen on 4 blocks of 2x2x2
    # The 2x2 part uses A[:, 1:3] and B[1:3, :]
    # Split the 8 columns into 4 groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    for block in range(4):
        k_offset = block * 2
        # Get Strassen decomposition for this block
        U_s, V_s, W_s = _strassen_2x2()
        
        for c in range(7):
            # U: rows correspond to A[i, j] where i in {0,1}, j in {1,2} (the 2x2 part)
            # In the full U, row index is i*m + j = i*3 + j
            # For the 2x2 part, j_local in {0,1} maps to j_global = 1 + j_local
            for i_local in range(2):
                for j_local in range(2):
                    val = U_s[i_local * 2 + j_local, c]
                    if val != 0:
                        U[i_local * m + (1 + j_local), col] += val
            
            # V: rows correspond to B[j, k] where j in {1,2}, k in {k_offset, k_offset+1}
            # In the full V, row index is j*p + k
            # For the 2x2 part, j_local in {0,1} maps to j_global = 1 + j_local
            # k_local in {0,1} maps to k_global = k_offset + k_local
            for j_local in range(2):
                for k_local in range(2):
                    val = V_s[j_local * 2 + k_local, c]
                    if val != 0:
                        V[(1 + j_local) * p + (k_offset + k_local), col] += val
            
            # W: rows correspond to C[i, k] where i in {0,1}, k in {k_offset, k_offset+1}
            # In the full W, row index is k*n + i
            # k_local in {0,1} maps to k_global = k_offset + k_local
            for i_local in range(2):
                for k_local in range(2):
                    val = W_s[k_local * 2 + i_local, c]
                    if val != 0:
                        W[(k_offset + k_local) * n + i_local, col] += val
            
            col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly represents matrix multiplication."""
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_arr = np.array(U) if not isinstance(U, np.ndarray) else U
    V_arr = np.array(V) if not isinstance(V, np.ndarray) else V
    W_arr = np.array(W) if not isinstance(W, np.ndarray) else W
    
    C_computed = np.zeros((n, p))
    rank = U_arr.shape[1]
    for c in range(rank):
        X = np.zeros(n)
        for i in range(n):
            for j in range(m):
                X[i] += U_arr[i * m + j, c] * A[i, j]
        Y = np.zeros(p)
        for k in range(p):
            for j in range(m):
                Y[k] += V_arr[j * p + k, c] * B[j, k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += X[i] * Y[k] * W_arr[k * n + i, c]
    
    return np.allclose(C_computed, C_expected, atol=tol)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the rank-44 decomposition
    U, V, W = _build_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if verification fails
    U0, V0, W0 = _schoolbook(n, m, p)
    return {
        "U": U0.tolist(),
        "V": V0.tolist(),
        "W": W0.tolist()
    }
# EVOLVE-BLOCK-END
