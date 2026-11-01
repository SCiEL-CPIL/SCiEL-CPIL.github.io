import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_a = T.reshape(nm, -1)  # (nm, mp*pn)
    T_b = T.transpose(1,0,2).reshape(mp, -1)  # (mp, nm*pn)
    T_c = T.transpose(2,0,1).reshape(pn, -1)  # (pn, nm*mp)
    
    def check_decomposition(U, V, W, T):
        """Check if U,V,W exactly decompose T."""
        resid = 0.0
        for r in range(U.shape[1]):
            # outer(outer(U[:,r], V[:,r]), W[:,r])
            temp = np.outer(U[:,r], V[:,r])  # (nm, mp)
            full = np.outer(temp, W[:,r])  # (nm, mp, pn)
            resid += np.sum((full - T)**2)
        return resid
    
    def try_rank(R, seed=42, max_iter=500):
        if time.time() > deadline - 15:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(nm, R)
        V = rng.randn(mp, R)
        W = rng.randn(pn, R)
        
        # Normalize columns to prevent scaling issues
        for r in range(R):
            norm = np.sqrt(np.sum(U[:,r]**2) + np.sum(V[:,r]**2) + np.sum(W[:,r]**2))
            if norm > 0:
                U[:,r] /= norm
                V[:,r] /= norm
                W[:,r] /= norm
        
        prev_resid = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 15:
                return None
            
            # Solve U: T_a = U @ B.T where B[:,r] = ravel(outer(V[:,r], W[:,r]))
            B = np.zeros((mp*pn, R))
            for r in range(R):
                B[:,r] = np.outer(V[:,r], W[:,r]).ravel()
            U_new, _, _, _ = np.linalg.lstsq(B.T, T_a.T, rcond=None)
            U = U_new.T
            
            # Solve V: T_b = V @ A.T where A[:,r] = ravel(outer(U[:,r], W[:,r]))
            A = np.zeros((nm*pn, R))
            for r in range(R):
                A[:,r] = np.outer(U[:,r], W[:,r]).ravel()
            V_new, _, _, _ = np.linalg.lstsq(A.T, T_b.T, rcond=None)
            V = V_new.T
            
            # Solve W: T_c = W @ C.T where C[:,r] = ravel(outer(U[:,r], V[:,r]))
            C = np.zeros((nm*mp, R))
            for r in range(R):
                C[:,r] = np.outer(U[:,r], V[:,r]).ravel()
            W_new, _, _, _ = np.linalg.lstsq(C.T, T_c.T, rcond=None)
            W = W_new.T
            
            # Check convergence
            if it % 20 == 0:
                resid = check_decomposition(U, V, W, T)
                if resid < 1e-10:
                    break
                if resid > prev_resid * 1.01:
                    # Diverging, try to restart
                    break
                prev_resid = resid
        
        resid = check_decomposition(U, V, W, T)
        if resid < 1e-8:
            return (U, V, W)
        return None
    
    # Try ranks from low to high
    for R in range(18, 55):
        if time.time() > deadline - 25:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(R, R+10):
            if time.time() > deadline - 25:
                break
            result = try_rank(R, seed=seed)
            if result is not None:
                U, V, W = result
                # Round to nearest integer/half-integer if close
                def round_to_half(x):
                    return np.round(x * 2) / 2
                
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                
                # Check if rounded version still works
                if check_decomposition(U_r, V_r, W_r, T) < 1e-8:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If not found with multiple seeds, try next rank
    
    # Fallback: schoolbook
    rank = nm * mp
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
