import numpy as np
import time

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    nm = n * m  # 15
    mp = m * p  # 30
    pn = p * n  # 18
    
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
    
    def structured_init(rank, seed=42):
        """
        Initialize U, V, W using the structure of the matrix multiplication tensor.
        Each column of the decomposition should contribute to specific (i,j,k) triples.
        """
        rng = np.random.RandomState(seed)
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Assign each rank component to a subset of (i,j,k) triples
        # We'll distribute the 90 triples across rank components
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for r in range(rank):
            # Assign some triples to this component
            # Use a structured pattern: group by i, then j, then k
            start = r * len(triples) // rank
            end = (r + 1) * len(triples) // rank
            if start >= end:
                # Add small random perturbation to avoid degeneracy
                U[:, r] = rng.randn(nm) * 0.01
                V[:, r] = rng.randn(mp) * 0.01
                W[:, r] = rng.randn(pn) * 0.01
                continue
            
            for idx in range(start, end):
                i, j, k = triples[idx]
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
        
        # Add small random perturbation to break symmetry
        U += rng.randn(nm, rank) * 0.001
        V += rng.randn(mp, rank) * 0.001
        W += rng.randn(pn, rank) * 0.001
        
        return U, V, W
    
    def try_rank(r, max_iters=50, seed=42):
        if time.time() > deadline:
            return None
        
        # Use structured initialization
        U, V, W = structured_init(r, seed)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
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
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err += (s - 1.0) ** 2
            err = np.sqrt(err / (n * m * p))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                break
            
            # Early check with loose tolerance
            if it % 10 == 9:
                if verify_decomposition(U, V, W, tol=1e-6):
                    break
        
        # Final verification with strict tolerance
        if not verify_decomposition(best_U, best_V, best_W, tol=1e-9):
            return None
        
        # Snap to half-integers
        U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
        
        # Verify snapped version
        if not verify_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
            return None
        
        return U_snap, V_snap, W_snap
    
    def schoolbook_decomposition():
        """Schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
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
        
        return U, V, W, rank
    
    # Try ranks from lower bound to 90
    # Lower bound: max(nm, mp, pn) = 30
    best_result = None
    
    # We expect rank 90 to be the answer if no structure is exploited
    # But we must try lower ranks first to be sure.
    for r in range(30, 91):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts for each rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            result = try_rank(r, max_iters=30, seed=r * 100 + restart)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W, r)
                found = True
                break
        
        if found:
            break  # Found the minimum rank
    
    if best_result is None:
        # Fallback to schoolbook rank 90
        U, V, W, r = schoolbook_decomposition()
        best_result = (U, V, W, r)
    
    U, V, W, r = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
