import numpy as np
import time

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Pre-compute target tensor for fast verification
    # Shape: (nm, mp, pn)
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    def verify_decomposition(U, V, W, tol=1e-9):
        """
        Efficiently verify if U, V, W decompose the matrix multiplication tensor.
        Uses tensor contraction: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        """
        rank = U.shape[1]
        T_comp = np.zeros((nm, mp, pn))
        for r in range(rank):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Check against target
        if not np.allclose(T_comp, T_target, atol=tol, rtol=0):
            return False
        return True

    def snap_to_half_integers(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    def try_rank(r, max_iters=20, seed=42):
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 0.1
        W = rng.randn(pn, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Solve M @ x = ones(p)
                    M = np.outer(V[j*p:(j+1)*p, :], W[i:pn:n, :])
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # Solve M @ x = ones(n)
                    M = np.outer(U[:nm:n, :], W[k:pn:n, :])
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for i in range(n):
                for k in range(p):
                    # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # Solve M @ x = ones(m)
                    M = np.outer(U[i*m:(i+1)*m, :], V[k:mp:p, :])
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Early check with loose tolerance
            if it % 5 == 4:
                if verify_decomposition(U, V, W, tol=1e-6):
                    break
        
        # Final verification with strict tolerance
        if not verify_decomposition(U, V, W, tol=1e-9):
            return None
        
        # Snap to half-integers
        U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
        
        # Verify snapped version
        if not verify_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
            return None
        
        return U_snap, V_snap, W_snap

    # Try ranks from lower bound to 90
    # Lower bound: max(nm, mp, pn) = 30
    # We will try a few random restarts for each rank if time permits
    best_result = None
    
    # We expect rank 90 to be the answer for 3x5x6 if no structure is exploited
    # But we must try lower ranks first to be sure.
    # To save time, we limit the number of restarts per rank.
    
    for r in range(30, 91):
        if time.time() > deadline - 5:
            break
        
        # Try 3 random restarts for each rank
        found = False
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            result = try_rank(r, max_iters=15, seed=r * 100 + restart)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W, r)
                found = True
                break
        
        if found:
            break  # Found the minimum rank
    
    if best_result is None:
        # Fallback to schoolbook rank 90
        r = 90
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        # Verify schoolbook (should be exact)
        if not verify_decomposition(U, V, W, tol=1e-9):
            # This should never happen
            pass
            
        best_result = (U, V, W, r)
    
    U, V, W, r = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
