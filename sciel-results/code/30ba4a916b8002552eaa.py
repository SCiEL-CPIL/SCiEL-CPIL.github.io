import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build rank-28 baseline
    # Strategy: 
    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:2] (Rank 7)
    # 2. Strassen for A[0:2, 0:2] * B[0:2, 2:4] (Rank 7)
    # 3. Schoolbook for A[0:2, 0:2] * B[0:2, 4] (Rank 4)
    # 4. Schoolbook for A[0:2, 2] * B[2, 0:2] (Rank 4)
    # 5. Schoolbook for A[0:2, 2] * B[2, 2:4] (Rank 4)
    # 6. Schoolbook for A[0:2, 2] * B[2, 4] (Rank 2)
    # Total: 7 + 7 + 4 + 4 + 4 + 2 = 28
    
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U28[i1*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        # M2 = (A10 + A11) B00
        U28[i2*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1
        W28[k1*n+i2, col] = 1; W28[k2*n+i2, col] = -1
        col += 1
        # M3 = A00 (B01 - B11)
        U28[i1*m+j1, col] = 1
        V28[j1*p+k2, col] = 1; V28[j2*p+k2, col] = -1
        W28[k2*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        # M4 = A11 (B10 - B00)
        U28[i2*m+j2, col] = 1
        V28[j2*p+k1, col] = 1; V28[j1*p+k1, col] = -1
        W28[k1*n+i1, col] = 1; W28[k1*n+i2, col] = 1
        col += 1
        # M5 = (A00 + A01) B11
        U28[i1*m+j1, col] = 1; U28[i1*m+j2, col] = 1
        V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = -1; W28[k2*n+i1, col] = 1
        col += 1
        # M6 = (A10 - A00)(B00 + B01)
        U28[i2*m+j1, col] = 1; U28[i1*m+j1, col] = -1
        V28[j1*p+k1, col] = 1; V28[j1*p+k2, col] = 1
        W28[k2*n+i2, col] = 1
        col += 1
        # M7 = (A01 - A11)(B10 + B11)
        U28[i1*m+j2, col] = 1; U28[i2*m+j2, col] = -1
        V28[j2*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U28[i*m+j, col] = 1
                    V28[j*p+k, col] = 1
                    W28[k*n+i, col] = 1
                    col += 1

    # Apply decomposition
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_schoolbook([0, 1], [0, 1], [4])
    add_schoolbook([0, 1], [2], [0, 1])
    add_schoolbook([0, 1], [2], [2, 3])
    add_schoolbook([0, 1], [2], [4])

    def verify(U, V, W):
        # Efficient verification using einsum
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Note: W is indexed by k*n+i, so W[k,i,c]
        # We reshape to 3D and use tensordot or einsum
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Sum over c: U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a contraction of 3 tensors along the last axis.
        # res[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
        # We can do this in two steps:
        # temp[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c]
        # res[i,j,k] = sum_c temp[i,j,k] * W_3d[k,i,c] -- NO, c is summed in both.
        # It's a 3-way product.
        # Let's use a loop over c for clarity and speed since rank is small (28).
        res = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            Uc = U_3d[:, :, c]
            Vc = V_3d[:, :, c]
            Wc = W_3d[:, :, c]
            # res[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            # Wc is p x n. Wc[k,i] is Wc.T[i,k]? No.
            # W is indexed by k*n+i. So W_3d[k, i, c].
            # We need to multiply Uc[i,j] * Vc[j,k] * Wc[k,i].
            # This is not a standard matrix product.
            # Let's stick to the explicit loop or careful einsum.
            # np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            pass
        
        # Using einsum is much faster
        res = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        target = np.ones((n, m, p))
        return np.allclose(res, target, atol=1e-9)

    if not verify(U28, V28, W28):
        # Fallback to schoolbook rank 30
        U30 = np.zeros((n * m, 30))
        V30 = np.zeros((m * p, 30))
        W30 = np.zeros((p * n, 30))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U30[i*m+j, col] = 1
                    V30[j*p+k, col] = 1
                    W30[k*n+i, col] = 1
                    col += 1
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}

    # Try to reduce rank by removing one redundant term
    for remove_idx in range(28):
        if time.time() > deadline - 10:
            break
        U_test = np.delete(U28, remove_idx, axis=1)
        V_test = np.delete(V28, remove_idx, axis=1)
        W_test = np.delete(W28, remove_idx, axis=1)
        if verify(U_test, V_test, W_test):
            return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}

    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
