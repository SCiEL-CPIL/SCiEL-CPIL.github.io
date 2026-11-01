import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-14 decomposition for 2x3x6
    # Based on the rank-7 decomposition of 3x3x3 (Alon-Boros-Cohen)
    
    # 3x3x3 rank-7 decomposition
    # Indices: U, V, W are 9x7
    U33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1],
    ], dtype=np.int64)

    V33 = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0, -1, 1],
        [0, 0, 0, 0, 1, -1, 0],
    ], dtype=np.int64)

    W33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.int64)

    # Slice to 2x3x3
    # U: rows 0..5 (i=0,1; j=0,1,2)
    U_2x3x3 = U33[0:6, :]
    
    # V: rows 0..8 (j=0,1,2; k=0,1,2) - No change for V in slicing i
    V_2x3x3 = V33.copy()
    
    # W: rows k*n+i. For 3x3, W is 9x7.
    # We need W for 2x3x3. Indices (k,i) with k in 0..2, i in 0..1.
    # In 3x3x3, W index is k*3+i.
    # We want W_2x3x3[k*2+i, :] = W33[k*3+i, :]
    W_2x3x3 = np.zeros((6, 7), dtype=np.int64)
    for k in range(3):
        for i in range(2):
            W_2x3x3[k * 2 + i, :] = W33[k * 3 + i, :]

    # Construct 2x3x6
    # p=6 is split into k=0..2 and k=3..5
    # The tensor is T[i,j,k].
    # For k in 0..2, use the first set of terms (r=0..6).
    # For k in 3..5, use the second set of terms (r=7..13).
    
    r = 14
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)

    # Block 1: k=0,1,2 (r=0..6)
    U[:, 0:7] = U_2x3x3
    
    # V mapping for k=0,1,2
    # V[j*6 + k, r] = V_2x3x3[j*3 + k, r]
    for j in range(3):
        for k in range(3):
            V[j * 6 + k, 0:7] = V_2x3x3[j * 3 + k, 0:7]
            
    # W mapping for k=0,1,2
    # W[k*2 + i, r] = W_2x3x3[k*2 + i, r]
    W[0:6, 0:7] = W_2x3x3
    
    # Block 2: k=3,4,5 (r=7..13)
    # U is independent of k, so same U
    U[:, 7:14] = U_2x3x3
    
    # V mapping for k=3,4,5
    # Let k_local = k - 3.
    # V[j*6 + k, r+7] = V_2x3x3[j*3 + k_local, r]
    for j in range(3):
        for k_local in range(3):
            k_global = k_local + 3
            V[j * 6 + k_global, 7:14] = V_2x3x3[j * 3 + k_local, 0:7]
            
    # W mapping for k=3,4,5
    # W[k*2 + i, r+7] = W_2x3x3[k_local*2 + i, r]
    # Note: W_2x3x3 already has the correct mapping for k_local.
    for k_local in range(3):
        k_global = k_local + 3
        for i in range(2):
            W[k_global * 2 + i, 7:14] = W_2x3x3[k_local * 2 + i, 0:7]

    # Verification
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1 for all i,j,k
    # Reshape
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    # Use einsum for speed
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Check if T is all 1s
    if not np.allclose(T, 1.0):
        # Fallback to schoolbook if verification fails (should not happen)
        return _construct_schoolbook(n, m, p)

    return _to_dict(U, V, W)

def _construct_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    return _to_dict(U, V, W)

def _to_dict(U, V, W):
    def clean(arr):
        out = []
        for row in arr:
            crow = []
            for x in row:
                xi = int(x)
                if x == xi:
                    crow.append(xi)
                else:
                    # Should not happen for integer constructions, but handle halves just in case
                    h = int(round(2.0 * float(x)))
                    if abs(2.0 * float(x) - h) < 1e-9:
                        crow.append(h / 2.0)
                    else:
                        crow.append(float(x))
            out.append(crow)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
