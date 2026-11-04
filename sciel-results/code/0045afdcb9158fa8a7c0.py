"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return a rank-n*m*p exact decomposition (schoolbook)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return Strassen 2x2x2 factor matrices.
    U: (4,7) rows indexed by i*2+j
    V: (4,7) rows indexed by j*2+k
    W: (4,7) rows indexed by k*2+i
    """
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U2, V2, W2

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1) > tol:
                    return False
    return True

def _strassen_kron_4x4():
    """
    Construct rank-49 decomposition for 4x4 via Kronecker product of Strassen.
    Returns dict with U, V, W as lists of lists.
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker product
    U4 = np.kron(U2, U2)  # (16, 49)
    V4 = np.kron(V2, V2)  # (16, 49)
    W4 = np.kron(W2, W2)  # (16, 49)
    
    # Permute rows to match linear index convention:
    # U row: i*4+j, V row: j*4+k, W row: k*4+i
    
    # For np.kron(A, B) with A, B both (4,7):
    # Row index R = i_A * 4 + i_B
    # where i_A is row index in A, i_B is row index in B
    
    # For U: A rows indexed by i_out*2+j_out, B rows indexed by i_in*2+j_in
    # R = (i_out*2+j_out)*4 + (i_in*2+j_in)
    # Target S = i*4+j = (i_out*2+i_in)*4 + (j_out*2+j_in)
    
    perm_u = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out = i // 2
        i_in = i % 2
        j_out = j // 2
        j_in = j % 2
        a_idx = i_out * 2 + j_out
        b_idx = i_in * 2 + j_in
        R = a_idx * 4 + b_idx
        perm_u[S] = R
    
    # For V: A rows indexed by j_out*2+k_out, B rows indexed by j_in*2+k_in
    # R = (j_out*2+k_out)*4 + (j_in*2+k_in)
    # Target S = j*4+k = (j_out*2+j_in)*4 + (k_out*2+k_in)
    
    perm_v = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out = j // 2
        j_in = j % 2
        k_out = k // 2
        k_in = k % 2
        a_idx = j_out * 2 + k_out
        b_idx = j_in * 2 + k_in
        R = a_idx * 4 + b_idx
        perm_v[S] = R
    
    # For W: A rows indexed by k_out*2+i_out, B rows indexed by k_in*2+i_in
    # R = (k_out*2+i_out)*4 + (k_in*2+i_in)
    # Target S = k*4+i = (k_out*2+k_in)*4 + (i_out*2+i_in)
    
    perm_w = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out = k // 2
        k_in = k % 2
        i_out = i // 2
        i_in = i % 2
        a_idx = k_out * 2 + i_out
        b_idx = k_in * 2 + i_in
        R = a_idx * 4 + b_idx
        perm_w[S] = R
    
    U4_fixed = U4[perm_u, :]
    V4_fixed = V4[perm_v, :]
    W4_fixed = W4[perm_w, :]
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    via the tensor (Kronecker) product of the rank-7 Strassen scheme.
    Includes explicit deadline guard and fallback to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen Kronecker product (rank 49)
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    result = _strassen_kron_4x4()
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Verify correctness
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
