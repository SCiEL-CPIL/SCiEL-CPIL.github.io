import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 115
    
    def try_rank(r, nm, mp, pn, T, max_iter=500):
        """Try to find rank-r decomposition using iterative method."""
        # Initialize randomly
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U: for each u, U[u,:] should satisfy sum_t U[u,t]*V[v,t]*W[w,t] = T[u,v,w]
            # Fix V, W: U[u,:] = argmin sum_{v,w} (T[u,v,w] - sum_t U[u,t]*V[v,t]*W[w,t])^2
            # This is a linear system in U[u,:]
            # Let M_u[v,w] = sum_t V[v,t]*W[w,t], which is V @ W.T reshaped
            # Actually, for fixed u, we want: sum_t U[u,t] * (V[:,t] * W[:,t])_reshaped = T[u,:,:].reshape(-1)
            # Let B_u = V @ W.T  (shape mp x pn)
            # Then T[u,:,:].reshape(-1) = U[u,:] @ B_u.reshape(-1)? No, that's not right.
            
            # Actually: sum_t U[u,t] * V[v,t] * W[w,t] for all v,w
            # = sum_t U[u,t] * (V[:,t] * W[:,t])_flattened
            # Let A_u = matrix where A_u[v*w + w', t] = V[v,t] * W[w',t]
            # This is complicated. Let me use a simpler approach.
            
            # For each u, solve: min_U sum_{v,w} (T[u,v,w] - sum_t U[u,t]*V[v,t]*W[w,t])^2
            # The normal equations: U[u,:] @ (V*W) @ (V*W).T = T[u,:,:] @ (V*W)
            # where (V*W) is the Khatri-Rao product
            
            # Khatri-Rao product of V (mp x r) and W (pn x r):
            # KR[t, v*pn + w] = V[v,t] * W[w,t]
            # Shape: (r, mp*pn)
            KR = np.zeros((r, mp * pn))
            for t in range(r):
                KR[t, :] = (V[:, t].reshape(mp, 1) * W[:, t].reshape(1, pn)).flatten()
            
            # For each u: U[u,:] @ KR = T[u,:,:].flatten()
            # So U[u,:] = lstsq(KR.T, T[u,:,:].flatten())
            rhs = T.reshape(nm, -1)  # (nm, mp*pn)
            # U = rhs @ pinv(KR.T)  but this is expensive
            
            # Use lstsq for each u
            U_new = np.zeros((nm, r))
            for u in range(nm):
                try:
                    U_new[u, :], _, _, _ = np.linalg.lstsq(KR.T, rhs[u, :], rcond=None)
                except:
                    pass
            
            # Similarly update V and W
            KR2 = np.zeros((r, nm * pn))
            for t in range(r):
                KR2[t, :] = (U[:, t].reshape(nm, 1) * W[:, t].reshape(1, pn)).flatten()
            rhs2 = T.reshape(mp, -1)  # Wait, T is (nm, mp, pn), we need to reshape for V update
            # For V: sum_t U[u,t]*V[v,t]*W[w,t] = T[u,v,w]
            # Fix U, W: V[v,:] = argmin sum_{u,w} (T[u,v,w] - sum_t V[v,t]*U[u,t]*W[w,t])^2
            # KR3[t, u*pn + w] = U[u,t] * W[w,t]
            KR3 = np.zeros((r, nm * pn))
            for t in range(r):
                KR3[t, :] = (U[:, t].reshape(nm, 1) * W[:, t].reshape(1, pn)).flatten()
            rhs3 = T.transpose(1, 0, 2).reshape(mp, -1)  # (mp, nm*pn)
            V_new = np.zeros((mp, r))
            for v in range(mp):
                try:
                    V_new[v, :], _, _, _ = np.linalg.lstsq(KR3.T, rhs3[v, :], rcond=None)
                except:
                    pass
            
            # For W: sum_t U[u,t]*V[v,t]*W[w,t] = T[u,v,w]
            # Fix U, V: W[w,:] = argmin sum_{u,v} (T[u,v,w] - sum_t W[w,t]*U[u,t]*V[v,t])^2
            KR4 = np.zeros((r, nm * mp))
            for t in range(r):
                KR4[t, :] = (U[:, t].reshape(nm, 1) * V[:, t].reshape(1, mp)).flatten()
            rhs4 = T.transpose(2, 0, 1).reshape(pn, -1)  # (pn, nm*mp)
            W_new = np.zeros((pn, r))
            for w in range(pn):
                try:
                    W_new[w, :], _, _, _ = np.linalg.lstsq(KR4.T, rhs4[w, :], rcond=None)
                except:
                    pass
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_recon = np.zeros((nm, mp, pn))
            for t in range(r):
                T_recon += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
            err = np.max(np.abs(T_recon - T))
            if err < 1e-10:
                return (U, V, W)
        
        return None
    
    # Build the full tensor
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try block-wise approach
    # Split p=10 into 5 blocks of size 2
    best_result = None
    best_rank = nm * mp * pn
    
    # Try rank 11 for 2x3 * 3x2 block
    for r in [11, 12, 13]:
        if time.time() > deadline - 10:
            break
        
        T_block = np.zeros((6, 6, 4))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    T_block[i*3+j, j*2+k, k*2+i] = 1.0
        
        result = try_rank(r, 6, 6, 4, T_block, max_iter=200)
        if result is not None:
            U_b, V_b, W_b = result
            total_rank = 5 * r
            U_full = np.zeros((nm, total_rank))
            V_full = np.zeros((mp, total_rank))
            W_full = np.zeros((pn, total_rank))
            
            for b in range(5):
                off = b * r
                U_full[:, off:off+r] = U_b
                for t in range(r):
                    for j in range(3):
                        for kl in range(2):
                            V_full[j*10 + b*2 + kl, off+t] = V_b[j*2+kl, t]
                    for kl in range(2):
                        for i in range(2):
                            W_full[(b*2+kl)*2 + i, off+t] = W_b[kl*2+i, t]
            
            # Verify
            T_recon = np.zeros((nm, mp, pn))
            for t in range(total_rank):
                T_recon += np.outer(np.outer(U_full[:, t], V_full[:, t]), W_full[:, t])
            if np.allclose(T_recon, T, atol=1e-10):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = (U_full, V_full, W_full)
                    break
    
    if best_result is not None:
        U, V, W = best_result
    else:
        # Fallback schoolbook
        rank = nm * mp * pn
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
