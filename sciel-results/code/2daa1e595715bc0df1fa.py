"""Matrix multiplication 4x4x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using Kronecker product of Strassen (rank 7) with proper index reordering.
    Falls back to rank-64 schoolbook if verification fails.
    """
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s kill

    n, m, p = DIMENSIONS

    # Strassen 2x2 decomposition (rank 7) - pre-verified
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Verify 2x2 Strassen decomposition using vectorized check
    # T2[i,j,k] = sum_c U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] should be 1
    valid_2x2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = int(np.dot(U2[i*2+j, :], V2[j*2+k, :] * W2[k*2+i, :]))
                if s != 1:
                    valid_2x2 = False
                    break
            if not valid_2x2:
                break
        if not valid_2x2:
            break

    if not valid_2x2:
        return _schoolbook(n, m, p)

    if time.time() - start_time > DEADLINE:
        return _schoolbook(n, m, p)

    # Kronecker product to get 4x4 decomposition (rank 49)
    U4_kron = np.kron(U2, U2)  # shape (16, 49)
    V4_kron = np.kron(V2, V2)  # shape (16, 49)
    W4_kron = np.kron(W2, W2)  # shape (16, 49)

    # Apply bit-interleaving permutation to reorder rows
    # np.kron(A,B) row index R = row_A * 4 + row_B
    # For U: target S = i*4+j, where i=i1*2+i2, j=j1*2+j2
    #   kron row_A = i1*2+j1, kron row_B = i2*2+j2
    #   R = (i1*2+j1)*4 + (i2*2+j2)
    # For V: target S = j*4+k, where j=j1*2+j2, k=k1*2+k2
    #   kron row_A = j1*2+k1, kron row_B = j2*2+k2
    #   R = (j1*2+k1)*4 + (j2*2+k2)
    # For W: target S = k*4+i, where k=k1*2+k2, i=i1*2+i2
    #   kron row_A = k1*2+i1, kron row_B = k2*2+i2
    #   R = (k1*2+i1)*4 + (k2*2+i2)

    def compute_perm_u(S):
        i = S // 4
        j = S % 4
        i1, i2 = i // 2, i % 2
        j1, j2 = j // 2, j % 2
        return (i1 * 2 + j1) * 4 + (i2 * 2 + j2)

    def compute_perm_v(S):
        j = S // 4
        k = S % 4
        j1, j2 = j // 2, j % 2
        k1, k2 = k // 2, k % 2
        return (j1 * 2 + k1) * 4 + (j2 * 2 + k2)

    def compute_perm_w(S):
        k = S // 4
        i = S % 4
        k1, k2 = k // 2, k % 2
        i1, i2 = i // 2, i % 2
        return (k1 * 2 + i1) * 4 + (k2 * 2 + i2)

    perm_U = np.array([compute_perm_u(S) for S in range(16)])
    perm_V = np.array([compute_perm_v(S) for S in range(16)])
    perm_W = np.array([compute_perm_w(S) for S in range(16)])

    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    # Verify the 4x4 decomposition using vectorized check
    # T[i,j,k] should be 1 if i==k else 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                s = int(np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :]))
                if s != expected:
                    return _schoolbook(n, m, p)

    if time.time() - start_time > DEADLINE:
        return _schoolbook(n, m, p)

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

def _schoolbook(n, m, p):
    """Return the naive rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
