import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 110
    
    # Target Tensor T (flattened)
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(i,k) else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0

    # Schoolbook Rank
    rank_school = n*m*p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U_s[i*m+j, r] = 1
                V_s[j*p+k, r] = 1
                W_s[k*n+i, r] = 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_s, V_s, W_s

    def verify(U, V, W, tol=1e-6):
        # Check full tensor
        # Reshape U, V, W to (n, m, r), (m, p, r), (p, n, r)
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        # Contract
        # Sum over r
        # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Note: W index is k*n+i. In W_3d, index is [k, i, r].
        r = U.shape[1]
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
        
        # Compare with T
        # T is 1 at valid, 0 elsewhere
        # We need T_recon to match T
        # T_flat is easier
        T_flat = T.flatten()
        T_recon_flat = T_recon.flatten()
        # But T_recon is indexed by i,j,k. T is indexed by u,v,w.
        # The mapping is not direct flatten.
        # We must check specific entries.
        
        # Check 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_recon[i, j, k] - 1.0) > tol:
                        return False
        # Check 0s
        # Iterate all u, v, w
        for u in range(n*m):
            i, j = divmod(u, m)
            for v in range(m*p):
                j2, k = divmod(v, p)
                for w in range(p*n):
                    k2, i2 = divmod(w, n)
                    # Valid if i==i2, j==j2, k==k2
                    if i == i2 and j == j2 and k == k2:
                        continue
                    # This is an invalid entry, should be 0
                    # T_recon doesn't have this index directly in (i,j,k) form?
                    # T_recon[i, j, k] corresponds to u=i*m+j, v=j*p+k, w=k*n+i.
                    # So T_recon only covers the "diagonal" valid entries?
                    # NO. The tensor T has dimensions (nm, mp, pn).
                    # T_recon[i,j,k] is the value at u=i*m+j, v=j*p+k, w=k*n+i.
                    # This is ONLY the valid entries.
                    # The invalid entries are NOT in T_recon.
                    # So we must compute the value at invalid entries explicitly.
                    
        # This is too slow.
        # Better: Construct the full T_recon tensor of shape (nm, mp, pn).
        U_flat = U
        V_flat = V
        W_flat = W
        # T_recon_full[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
        # This is a tensor contraction.
        # Can do: T_recon_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        # This is O(nm * mp * pn * r).
        # 9 * 15 * 15 * 17 = 34k ops. Very fast.
        
        T_recon_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        # Compare T_recon_full with T
        if not np.allclose(T_recon_full, T, atol=tol):
            return False
        return True

    # Optimization Loop
    for r in range(17, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(trial * 100 + r)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Adam Optimizer
            mU = np.zeros_like(U)
            vU = np.zeros_like(U)
            mV = np.zeros_like(V)
            vV = np.zeros_like(V)
            mW = np.zeros_like(W)
            vW = np.zeros_like(W)
            b1 = 0.9
            b2 = 0.999
            eps = 1e-8
            lr = 0.01
            
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Compute T_recon
                T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
                loss = np.sum((T_recon - T)**2)
                
                if loss < 1e-10:
                    if verify(U, V, W):
                        best_rank = r
                        best_U, best_V, best_W = U, V, W
                        found = True
                        break
                
                # Gradients
                # dL/du = 2 * (T_recon - T) * V * W
                # dL/dv = 2 * (T_recon - T) * U * W
                # dL/dw = 2 * (T_recon - T) * U * V
                
                diff = T_recon - T
                dU = 2 * np.einsum('uvw,vr,wr->ur', diff, V, W)
                dV = 2 * np.einsum('uvw,ur,wr->vr', diff, U, W)
                dW = 2 * np.einsum('uvw,ur,vr->wr', diff, U, V)
                
                # Adam Update
                t = step + 1
                mU = b1*mU + (1-b1)*dU
                vU = b2*vU + (1-b2)*(dU**2)
                mU_hat = mU / (1 - b1**t)
                vU_hat = vU / (1 - b2**t)
                U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
                
                mV = b1*mV + (1-b1)*dV
                vV = b2*vV + (1-b2)*(dV**2)
                mV_hat = mV / (1 - b1**t)
                vV_hat = vV / (1 - b2**t)
                V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
                
                mW = b1*mW + (1-b1)*dW
                vW = b2*vW + (1-b2)*(dW**2)
                mW_hat = mW / (1 - b1**t)
                vW_hat = vW / (1 - b2**t)
                W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if found:
                break
        
        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
