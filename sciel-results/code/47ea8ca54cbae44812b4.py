import numpy as np
import random

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor T
    # T is a 3D array of shape (dim_u, dim_v, dim_w)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # Try ranks from 35 down to 30
    # 36 is the current best, so we try to beat it.
    for rank in range(35, 29, -1):
        # Try multiple random initializations
        for trial in range(10):
            # Initialize U, V, W
            # Using small random values to avoid large gradients initially
            U = np.random.randn(dim_u, rank) * 0.1
            V = np.random.randn(dim_v, rank) * 0.1
            W = np.random.randn(dim_w, rank) * 0.1
            
            # ALS optimization
            # Minimize || T - sum_r U[:,r] V[:,r]^T W[:,r]^T ||_F^2
            # This is equivalent to minimizing || T_reshaped - U (V * W) ||
            # Let's use a simple gradient descent or ALS.
            # ALS is faster for tensor CP decomposition.
            
            # Reshape T to matrix? No, 3-way tensor.
            # We can use the mode-unfolding trick.
            # T_{(1)} is (dim_u, dim_v*dim_w)
            # T_{(2)} is (dim_v, dim_u*dim_w)
            # T_{(3)} is (dim_w, dim_u*dim_v)
            
            T_1 = T.reshape(dim_u, dim_v * dim_w)
            T_2 = T.reshape(dim_v, dim_u * dim_w)
            T_3 = T.reshape(dim_w, dim_u * dim_v)
            
            for iter in range(100):
                # Update U
                # T_1 approx U * (V * W)_1
                # (V * W)_1 is (dim_v*dim_w, rank)
                # V_flat = V.reshape(dim_v * rank, 1)? No.
                # Let's construct the Khatri-Rao product.
                # V_W = V.T * W.T? No.
                # The term is sum_r U[:,r] (V[:,r] \otimes W[:,r])
                # So we need the Khatri-Rao product of V and W.
                # V is (dim_v, rank), W is (dim_w, rank)
                # V_W is (dim_v*dim_w, rank)
                # V_W[i*dim_w + j, r] = V[i, r] * W[j, r]
                
                V_W = np.zeros((dim_v * dim_w, rank), dtype=np.float64)
                for r in range(rank):
                    V_W[:, r] = np.kron(V[:, r], W[:, r])
                
                # Solve U * V_W = T_1
                # U = T_1 @ V_W^+
                try:
                    U_new = T_1 @ np.linalg.pinv(V_W)
                except np.linalg.LinAlgError:
                    break
                
                # Update V
                U_W = np.zeros((dim_u * dim_w, rank), dtype=np.float64)
                for r in range(rank):
                    U_W[:, r] = np.kron(U_new[:, r], W[:, r])
                try:
                    V_new = T_2 @ np.linalg.pinv(U_W)
                except np.linalg.LinAlgError:
                    break
                    
                # Update W
                U_V = np.zeros((dim_u * dim_v, rank), dtype=np.float64)
                for r in range(rank):
                    U_V[:, r] = np.kron(U_new[:, r], V_new[:, r])
                try:
                    W_new = T_3 @ np.linalg.pinv(U_V)
                except np.linalg.LinAlgError:
                    break
                    
                U, V, W = U_new, V_new, W_new
                
            # Check error
            # Compute approx tensor
            # approx[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
            # This is expensive to compute fully for verification, but we can check a subset or compute norm.
            # Let's compute the full tensor for small sizes.
            approx = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
            for r in range(rank):
                approx += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
            
            err = np.linalg.norm(approx - T)
            if err < 1e-6:
                # Round to nearest 0.5
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness with rounded values
                approx_int = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
                for r in range(rank):
                    approx_int += np.einsum('u,v,w->uvw', U_int[:, r], V_int[:, r], W_int[:, r])
                
                if np.allclose(approx_int, T, atol=1e-9):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
            
            # If error is low but rounding failed, try other trials/ranks
            
    # Fallback to current known rank 36 (Strassen)
    # Re-implement the Strassen logic from the seed to ensure we have a valid fallback
    rank = 36
    U = [[0]*rank for _ in range(dim_u)]
    V = [[0]*rank for _ in range(dim_v)]
    W = [[0]*rank for _ in range(dim_w)]
    
    def strassen_222(i_off, j_off, k_off):
        comps = []
        comps.append({'U': {(i_off,j_off):1,(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1,(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):1,(i_off+1,k_off+1):1}})
        comps.append({'U': {(i_off+1,j_off):1,(i_off+1,j_off+1):1}, 'V': {(j_off,k_off):1}, 'W': {(i_off+1,k_off):1,(i_off+1,k_off+1):-1}})
        comps.append({'U': {(i_off,j_off):1}, 'V': {(j_off,k_off+1):1,(j_off+1,k_off+1):-1}, 'W': {(i_off,k_off+1):1,(i_off+1,k_off+1):1}})
        comps.append({'U': {(i_off+1,j_off+1):1}, 'V': {(j_off+1,k_off):1,(j_off,k_off):-1}, 'W': {(i_off,k_off):1,(i_off+1,k_off):1}})
        comps.append({'U': {(i_off,j_off):1,(i_off,j_off+1):1}, 'V': {(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):-1,(i_off,k_off+1):1}})
        comps.append({'U': {(i_off+1,j_off):1,(i_off,j_off):-1}, 'V': {(j_off,k_off):1,(j_off,k_off+1):1}, 'W': {(i_off+1,k_off+1):1}})
        comps.append({'U': {(i_off,j_off+1):1,(i_off+1,j_off+1):-1}, 'V': {(j_off+1,k_off):1,(j_off+1,k_off+1):1}, 'W': {(i_off,k_off):1}})
        return comps

    col = 0
    for comp in strassen_222(0, 0, 0):
        for (i,j), val in comp['U'].items(): U[i*m+j][col] += val
        for (j,k), val in comp['V'].items(): V[j*p+k][col] += val
        for (i,k), val in comp['W'].items(): W[k*n+i][col] += val
        col += 1
    for comp in strassen_222(0, 0, 2):
        for (i,j), val in comp['U'].items(): U[i*m+j][col] += val
        for (j,k), val in comp['V'].items(): V[j*p+k][col] += val
        for (i,k), val in comp['W'].items(): W[k*n+i][col] += val
        col += 1
    for i in range(2):
        for j in range(2):
            U[i*m+j][col] = 1; V[j*p+4][col] = 1; W[4*n+i][col] = 1; col += 1
    for comp in strassen_222(0, 2, 0):
        for (i,j), val in comp['U'].items(): U[i*m+j][col] += val
        for (j,k), val in comp['V'].items(): V[j*p+k][col] += val
        for (i,k), val in comp['W'].items(): W[k*n+i][col] += val
        col += 1
    for comp in strassen_222(0, 2, 2):
        for (i,j), val in comp['U'].items(): U[i*m+j][col] += val
        for (j,k), val in comp['V'].items(): V[j*p+k][col] += val
        for (i,k), val in comp['W'].items(): W[k*n+i][col] += val
        col += 1
    for i in range(2):
        for j in range(2, 4):
            U[i*m+j][col] = 1; V[j*p+4][col] = 1; W[4*n+i][col] = 1; col += 1
            
    return {"U": U, "V": V, "W": W}
