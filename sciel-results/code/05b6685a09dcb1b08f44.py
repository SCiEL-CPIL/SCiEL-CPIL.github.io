"""Tensor decomposition for 2x3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]


def construct():
    start_time = time.time()
    deadline = 150.0  # seconds

    n, m, p = 2, 3, 8

    # Build the multiplication tensor T of shape (n*m, m*p, p*n) = (6, 24, 16)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def _verify_rank(rank, max_time=140.0):
        """Try to find an exact rank decomposition. Returns (U, V, W) or None."""
        # Use iterative approach: start with a random initialization and do
        # multiplicative updates / gradient descent on the tensor reconstruction
        #
        # For small rank, we can try a direct approach using the structure.
        
        # Strategy: build a structured decomposition and verify.
        # We'll try different rank values and structured ansatze.
        
        # First, let's try a block-based approach with Strassen
        # 2x2x2 Strassen has rank 7
        
        # Partition: split m=3 into (2,1), split p=8 into four blocks of 2
        # For each (i-block of 2, k-block of 2) pair: 2x2x2 -> rank 7 (Strassen)
        # For the m=1, k=2 part: 2x1x2 -> rank 4 (schoolbook)
        # Total: 4 * (7 + 4) = 44
        
        # Alternative: try to share structure
        # Let's build the rank-44 decomposition and then try to reduce rank
        # by finding linear dependencies among the rank-1 tensors.
        
        if rank >= 44:
            # Build rank-44 decomposition
            U_list = []
            V_list = []
            W_list = []
            
            for k_block in range(4):  # k in {0,1}, {2,3}, {4,5}, {6,7}
                k0 = k_block * 2
                
                # Strassen for (i in {0,1}, j in {0,1}, k in {k0, k0+1})
                # 2x2x2 Strassen with 7 terms
                # a = A[i0,j0], b = A[i0,j1], c = A[i1,j0], d = A[i1,j1]
                # e = B[j0,k0], f = B[j0,k0+1], g = B[j1,k0], h = B[j1,k0+1]
                
                # m1 = (a+d)(e+h)
                u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
                u1[0*m+0] = 1; u1[1*m+1] = 1
                v1[0*p+k0] = 1; v1[1*p+(k0+1)] = 1
                w1[k0*n+0] = 1; w1[(k0+1)*n+1] = 1
                U_list.append(u1); V_list.append(v1); W_list.append(w1)
                
                # m2 = (c+d)e
                u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
                u2[1*m+0] = 1; u2[1*m+1] = 1
                v2[0*p+k0] = 1
                w2[k0*n+1] = 1; w2[(k0+1)*n+1] = -1
                U_list.append(u2); V_list.append(v2); W_list.append(w2)
                
                # m3 = a(f-h)
                u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
                u3[0*m+0] = 1
                v3[0*p+(k0+1)] = 1; v3[1*p+(k0+1)] = -1
                w3[(k0+1)*n+0] = 1; w3[(k0+1)*n+1] = 1
                U_list.append(u3); V_list.append(v3); W_list.append(w3)
                
                # m4 = d(g-e)
                u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
                u4[1*m+1] = 1
                v4[1*p+k0] = 1; v4[0*p+k0] = -1
                w4[k0*n+0] = 1; w4[k0*n+1] = 1
                U_list.append(u4); V_list.append(v4); W_list.append(w4)
                
                # m5 = (a+b)h
                u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
                u5[0*m+0] = 1; u5[0*m+1] = 1
                v5[1*p+(k0+1)] = 1
                w5[k0*n+0] = -1; w5[(k0+1)*n+0] = 1
                U_list.append(u5); V_list.append(v5); W_list.append(w5)
                
                # m6 = (c-a)(e+f)
                u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
                u6[1*m+0] = 1; u6[0*m+0] = -1
                v6[0*p+k0] = 1; v6[0*p+(k0+1)] = 1
                w6[(k0+1)*n+1] = 1
                U_list.append(u6); V_list.append(v6); W_list.append(w6)
                
                # m7 = (b-d)(g+h)
                u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
                u7[0*m+1] = 1; u7[1*m+1] = -1
                v7[1*p+k0] = 1; v7[1*p+(k0+1)] = 1
                w7[k0*n+0] = 1
                U_list.append(u7); V_list.append(v7); W_list.append(w7)
                
                # Schoolbook for (i in {0,1}, j=2, k in {k0, k0+1})
                for ii in range(2):
                    for kk in range(2):
                        u_sb = np.zeros(n*m); v_sb = np.zeros(m*p); w_sb = np.zeros(p*n)
                        u_sb[ii*m + 2] = 1
                        v_sb[2*p + k0 + kk] = 1
                        w_sb[(k0+kk)*n + ii] = 1
                        U_list.append(u_sb); V_list.append(v_sb); W_list.append(w_sb)
            
            total_r = len(U_list)
            if total_r > rank:
                return None
            
            U_arr = np.array(U_list).T  # (n*m, total_r)
            V_arr = np.array(V_list).T  # (m*p, total_r)
            W_arr = np.array(W_list).T  # (p*n, total_r)
            
            # Verify
            recon = np.tensordot(U_arr, V_arr, axes=(1, 1))
            recon = np.tensordot(recon, W_arr, axes=(2, 1))
            if np.allclose(recon, T, atol=1e-10):
                if total_r == rank:
                    return (U_arr, V_arr, W_arr)
                else:
                    # Pad or slice
                    if total_r < rank:
                        # Pad with zeros
                        U_arr = np.pad(U_arr, ((0,0),(0,rank-total_r)))
                        V_arr = np.pad(V_arr, ((0,0),(0,rank-total_r)))
                        W_arr = np.pad(W_arr, ((0,0),(0,rank-total_r)))
                        return (U_arr, V_arr, W_arr)
                    else:
                        return (U_arr[:, :rank], V_arr[:, :rank], W_arr[:, :rank])
            return None
        
        # For rank < 44, try optimization
        if time.time() > deadline - 10:
            return None
        
        # Try to find a rank-r decomposition using iterative method
        # Initialize with random vectors and do multiplicative updates
        best_err = np.inf
        best_UVW = None
        
        n_trials = 3
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            U_r = np.random.randn(n*m, rank) * 0.1
            V_r = np.random.randn(m*p, rank) * 0.1
            W_r = np.random.randn(p*n, rank) * 0.1
            
            # Normalize columns
            for col_idx in range(rank):
                nu = np.linalg.norm(U_r[:, col_idx])
                nv = np.linalg.norm(V_r[:, col_idx])
                nw = np.linalg.norm(W_r[:, col_idx])
                if nu > 1e-10:
                    U_r[:, col_idx] /= nu
                if nv > 1e-10:
                    V_r[:, col_idx] /= nv
                if nw > 1e-10:
                    W_r[:, col_idx] /= nw
            
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute current reconstruction
                # T_approx[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
                # For updating U: fix V, W
                # T[i, :, :] should match sum_r U[i,r] * (V[:,r] outer W[:,r])
                
                # Compute effective V*W tensor: shape (m*p, p*n, rank)
                # VW[j,k,r] = V[j,r]*W[k,r]
                VW = V_r[:, :, None] * W_r[:, None, :]  # (m*p, p*n, rank)
                # VW shape: (m*p, p*n, rank)
                
                # For each i, T[i, j, k] = sum_r U[i,r] * VW[j,k,r]
                # So U[i,:] should be the solution to: VW_flat.T @ U[i,:] = T[i,:,:].flatten()
                # Where VW_flat has shape (m*p * p*n, rank)
                VW_flat = VW.reshape(m*p * p*n, rank)
                
                # Solve for U
                for i in range(n*m):
                    target = T[i, :, :].flatten()
                    # Least squares: minimize ||VW_flat @ U[i,:] - target||
                    # U[i,:] = (VW_flat.T @ VW_flat)^{-1} VW_flat.T @ target
                    A = VW_flat.T @ VW_flat
                    b = VW_flat.T @ target
                    # Regularize
                    A += 1e-8 * np.eye(rank)
                    try:
                        U_r[i, :] = np.linalg.solve(A, b)
                    except:
                        U_r[i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Update V
                UW = U_r[:, :, None] * W_r[:, None, :]  # (n*m, p*n, rank)
                UW_flat = UW.reshape(n*m * p*n, rank)
                for j in range(m*p):
                    target = T[:, j, :].reshape(-1)  # (n*m * p*n,)
                    A = UW_flat.T @ UW_flat
                    b = UW_flat.T @ target
                    A += 1e-8 * np.eye(rank)
                    try:
                        V_r[j, :] = np.linalg.solve(A, b)
                    except:
                        V_r[j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Update W
                UV = U_r[:, :, None] * V_r[:, None, :]  # (n*m, m*p, rank)
                UV_flat = UV.reshape(n*m * m*p, rank)
                for k in range(p*n):
                    target = T[:, :, k].reshape(-1)
                    A = UV_flat.T @ UV_flat
                    b = UV_flat.T @ target
                    A += 1e-8 * np.eye(rank)
                    try:
                        W_r[k, :] = np.linalg.solve(A, b)
                    except:
                        W_r[k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Check convergence
                recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                err = np.linalg.norm(recon - T)
                if err < best_err:
                    best_err = err
                    best_UVW = (U_r.copy(), V_r.copy(), W_r.copy())
                
                if err < 1e-10:
                    break
        
        if best_UVW is not None and best_err < 1e-10:
            U_arr, V_arr, W_arr = best_UVW
            # Round to nearest half-integer
            U_arr = np.round(U_arr * 2) / 2
            V_arr = np.round(V_arr * 2) / 2
            W_arr = np.round(W_arr * 2) / 2
            recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
            if np.allclose(recon, T, atol=1e-10):
                return (U_arr, V_arr, W_arr)
        
        return None

    # Try ranks from 44 downward
    for rank in range(44, 39, -1):
        result = _verify_rank(rank)
        if result is not None:
            U_arr, V_arr, W_arr = result
            return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    
    # Fallback: return rank-48 schoolbook
    U = [[0]*48 for _ in range(6)]
    V = [[0]*48 for _ in range(24)]
    W = [[0]*48 for _ in range(16)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(8):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
