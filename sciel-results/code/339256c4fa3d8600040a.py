import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    nmp = nm * p
    
    # Target tensor is all 1s
    target = np.ones((n, m, p), dtype=float)
    
    # Schoolbook decomposition
    schoolbook_rank = nmp
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, idx] = 1.0
                V_sb[j*p+k, idx] = 1.0
                W_sb[k*n+i, idx] = 1.0
                idx += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140  # 10s margin
    
    # Try lower ranks
    for rank in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
            
        # Multiple random restarts
        max_restarts = 20 if rank < 40 else 10
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Normalize columns
            for c in range(rank):
                U[:, c] /= (np.linalg.norm(U[:, c]) + 1e-10)
                V[:, c] /= (np.linalg.norm(V[:, c]) + 1e-10)
                W[:, c] /= (np.linalg.norm(W[:, c]) + 1e-10)
            
            # ALS Iterations
            max_iter = 500
            prev_loss = float('inf')
            loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                    
                # Compute Residual R
                R = target.copy()
                for c in range(rank):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    w_c_T = w_c.T
                    term = u_c[:, :, np.newaxis] * v_c[np.newaxis, :, :] * w_c_T[:, np.newaxis, :]
                    R -= term
                
                loss = np.sum(R**2)
                
                if loss < 1e-12:
                    break
                
                # Update U
                for c in range(rank):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    w_c_T = w_c.T
                    num = np.sum(R * v_c[np.newaxis, :, :] * w_c_T[:, np.newaxis, :], axis=2)
                    den = np.sum((v_c[np.newaxis, :, :] * w_c_T[:, np.newaxis, :])**2, axis=2)
                    mask = den > 1e-10
                    u_c_new = np.zeros_like(u_c)
                    u_c_new[mask] = num[mask] / den[mask]
                    U[:, c] = u_c_new.reshape(nm)
                    
                # Update V
                for c in range(rank):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    w_c_T = w_c.T
                    num = np.sum(R * u_c[:, :, np.newaxis] * w_c_T[:, np.newaxis, :], axis=0)
                    den = np.sum((u_c[:, :, np.newaxis] * w_c_T[:, np.newaxis, :])**2, axis=0)
                    mask = den > 1e-10
                    v_c_new = np.zeros_like(v_c)
                    v_c_new[mask] = num[mask] / den[mask]
                    V[:, c] = v_c_new.reshape(mp)
                    
                # Update W
                for c in range(rank):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    num = np.sum(R * u_c[:, :, np.newaxis] * v_c[np.newaxis, :, :], axis=1)
                    den = np.sum((u_c[:, :, np.newaxis] * v_c[np.newaxis, :, :])**2, axis=1)
                    mask = den > 1e-10
                    w_c_new = np.zeros((n, p))
                    w_c_new[mask] = num[mask] / den[mask]
                    W[:, c] = w_c_new.T.reshape(pn)

            if loss < 1e-9:
                # Verify
                T_recon = np.zeros((n, m, p))
                for c in range(rank):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    T_recon += u_c[:, :, np.newaxis] * v_c[np.newaxis, :, :] * w_c.T[:, np.newaxis, :]
                
                if np.allclose(T_recon, 1.0, atol=1e-8):
                    best_rank = rank
                    # Try rounding to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    T_recon_r = np.zeros((n, m, p))
                    for c in range(rank):
                        u_c = U_rounded[:, c].reshape(n, m)
                        v_c = V_rounded[:, c].reshape(m, p)
                        w_c = W_rounded[:, c].reshape(p, n)
                        T_recon_r += u_c[:, :, np.newaxis] * v_c[np.newaxis, :, :] * w_c.T[:, np.newaxis, :]
                    
                    if np.allclose(T_recon_r, 1.0, atol=1e-8):
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                    else:
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W.tolist()
                    found = True
                    break
                    
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
