import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook decomposition
    rank_schoolbook = n*m*p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_decomp = (U_sb, V_sb, W_sb)
    
    def verify_decomposition(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:,t], V[:,t])
            T_recon += np.einsum('ab,c->abc', uv, W[:,t])
        return np.max(np.abs(T_recon - T))
    
    # Try various ranks using ALS with multiple initializations
    np.random.seed(42)
    
    for target_rank in [110, 100, 90, 80, 70, 60, 50, 40]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(15):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with some structure
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # ALS iterations with convergence check
            prev_err = float('inf')
            for it in range(300):
                if time.time() > deadline - 15:
                    break
                
                # Fix V, W: solve for U
                X = np.einsum('bt,ct->bct', V, W).reshape(target_rank, mp*pn).T
                for a in range(nm):
                    U[a, :] = np.linalg.lstsq(X, T[a, :, :].reshape(-1), rcond=None)[0]
                
                # Fix U, W: solve for V
                X = np.einsum('at,ct->act', U, W).reshape(target_rank, nm*pn).T
                for b in range(mp):
                    V[b, :] = np.linalg.lstsq(X, T[:, b, :].reshape(-1), rcond=None)[0]
                
                # Fix U, V: solve for W
                X = np.einsum('at,bt->abt', U, V).reshape(target_rank, nm*mp).T
                for c in range(pn):
                    W[c, :] = np.linalg.lstsq(X, T[:, :, c].reshape(-1), rcond=None)[0]
                
                # Check convergence
                err = verify_decomposition(U, V, W)
                if err < 1e-10:
                    break
                if abs(err - prev_err) < 1e-12 and it > 50:
                    break
                prev_err = err
            
            err = verify_decomposition(U, V, W)
            if err < 1e-10:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_decomp = (U.copy(), V.copy(), W.copy())
                found = True
                break  # Found a valid decomposition for this rank, try lower
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
