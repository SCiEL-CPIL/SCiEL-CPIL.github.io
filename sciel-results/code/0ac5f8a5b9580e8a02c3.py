import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = 49
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen constants for 4x4x4
    s = [
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    t = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    w = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # s, t, w are 8x16, 8x16, 8x16
    for r in range(8):
        for i in range(4):
            for j in range(4):
                if s[r, i * 4 + j] != 0:
                    U[i * m + j, r] = s[r, i * 4 + j]
                if t[r, i * 4 + j] != 0:
                    V[i * m + j, r] = t[r, i * 4 + j]
                if w[r, i * 4 + j] != 0:
                    W[i * m + j, r] = w[r, i * 4 + j]
    
    # For k in 4..6, use a different set of terms
    # Actually, Strassen works for 4x4x4. We need to handle 4x4x7.
    # We can view B as 4x7 = 4x4 + 4x3.
    # But the tensor is fixed.
    # Let's use the fact that we can compute 4x4x4 and then handle the remaining columns.
    # Wait, the previous code had a bug: it only filled the first 8 columns of U, V, W.
    # U is 16x49, V is 28x49, W is 28x49.
    # The Strassen decomposition gives 8 terms for 4x4x4.
    # We need 49 terms total? No, rank is the number of terms.
    # The trivial rank is 4*4*7 = 112.
    # Strassen for 4x4x4 gives rank 8.
    # For 4x4x7, we can compute the first 4 columns with Strassen (rank 8) and the last 3 columns trivially (rank 4*3=12).
    # Total rank = 8 + 12 = 20.
    
    # Let's restart the construction.
    U = np.zeros((n * m, 20))
    V = np.zeros((m * p, 20))
    W = np.zeros((p * n, 20))
    
    # First 8 columns: Strassen for 4x4x4 (k=0..3)
    for r in range(8):
        for i in range(4):
            for j in range(4):
                if s[r, i * 4 + j] != 0:
                    U[i * m + j, r] = s[r, i * 4 + j]
                if t[r, i * 4 + j] != 0:
                    V[i * m + j, r] = t[r, i * 4 + j]
                if w[r, i * 4 + j] != 0:
                    W[j * p + i, r] = w[r, i * 4 + j]  # W index is k*n+i, here k=j, n=4, i=i -> j*4+i
    
    # Wait, W index is k*n+i. In Strassen, the output is C = AB.
    # C is 4x4. C[i, k] = sum_j A[i, j] B[j, k].
    # So W[k*n+i, r] corresponds to C[i, k].
    # In the Strassen decomposition, the output term w_r is a 4x4 matrix.
    # w_r[i, k] is the coefficient for C[i, k].
    # So W[k*n+i, r] = w_r[i, k].
    
    # Let's re-index w. w is 8x16. w[r, i*4+k] is the coefficient for C[i, k].
    # So W[k*4+i, r] = w[r, i*4+k].
    
    # Let's redo the first 8 columns.
    U = np.zeros((n * m, 20))
    V = np.zeros((m * p, 20))
    W = np.zeros((p * n, 20))
    
    for r in range(8):
        for i in range(4):
            for j in range(4):
                U[i * m + j, r] = s[r, i * 4 + j]
                V[j * p + i, r] = t[r, i * 4 + j]  # V index is j*p+k. Here k=i? No.
                # V is m x p. V[j, k].
                # In Strassen, B is 4x4. B[j, k].
                # So V[j*p+k, r] = t[r, j*4+k].
                # Let's check t. t is 8x16. t[r, j*4+k] is the coefficient for B[j, k].
                # So V[j*p+k, r] = t[r, j*4+k].
                # Here p=7, but we are only using k=0..3.
                V[j * p + i, r] = t[r, j * 4 + i]
                W[i * n + j, r] = w[r, i * 4 + j]  # W index is k*n+i. Here k=j, i=i.
                # Wait, C[i, k]. W[k*n+i, r].
                # So W[j*n+i, r] = w[r, i*4+j].
    
    # Let's be very careful with indices.
    # U[i*m+j, r] = s[r, i*4+j]
    # V[j*p+k, r] = t[r, j*4+k]
    # W[k*n+i, r] = w[r, i*4+k]
    
    U = np.zeros((n * m, 20))
    V = np.zeros((m * p, 20))
    W = np.zeros((p * n, 20))
    
    for r in range(8):
        for i in range(4):
            for j in range(4):
                U[i * m + j, r] = s[r, i * 4 + j]
                for k in range(4):
                    V[j * p + k, r] = t[r, j * 4 + k]
                    W[k * n + i, r] = w[r, i * 4 + k]
    
    # Next 12 columns: Trivial for k=4..6
    col = 8
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
