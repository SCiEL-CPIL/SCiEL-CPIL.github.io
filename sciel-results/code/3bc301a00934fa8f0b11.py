import time
import numpy as np
import random

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor as a 3D array
    # T[a, b, c] = 1 if exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    def make_decomposition(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        # Compute sum of rank-1 tensors
        result = np.zeros_like(T)
        for r in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(result, T, atol=tol)
    
    # Try to find lower rank decompositions using random search
    # Start from rank 64 and try to reduce
    
    # First, try a simple approach: random initialization + optimization
    # For each candidate rank, try multiple random restarts
    
    def try_rank(R, n_restarts=5):
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Simple gradient-free approach: iteratively update one factor at a time
            # Use ALS (Alternating Least Squares) style
            for it in range(100):
                if time.time() > deadline:
                    return None
                
                # Update U
                for a in range(nm):
                    # T[a,:,:] = sum_r U[a,r] * V[:,r] outer W[:,r]
                    # For fixed a, we want to find U[a,:] such that
                    # T[a,:,:] ≈ sum_r U[a,r] * V[:,r] * W[:,r]
                    # This is a linear system: for each (b,c), T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # Let M_bc_r = V[b,r] * W[c,r], then T[a,b,c] = sum_r U[a,r] * M_bc_r
                    # Reshape: T[a] is (mp, pn), M is (mp*pn, R)
                    M = (V.T * W.T).T  # This isn't right, let me think...
                    # Actually: sum_r U[a,r] * V[b,r] * W[c,r]
                    # Let's form a matrix A where A[(b,c), r] = V[b,r] * W[c,r]
                    A = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, R)
                    t_a = T[a].reshape(mp * pn)
                    # Solve: A @ U[a,:] ≈ t_a
                    try:
                        U[a, :] = np.linalg.lstsq(A, t_a, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(mp):
                    A = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, R)
                    t_b = T[:, b, :].reshape(nm * pn)
                    try:
                        V[b, :] = np.linalg.lstsq(A, t_b, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c in range(pn):
                    A = np.einsum('ar,br->abr', U, V).reshape(nm * mp, R)
                    t_c = T[:, :, c].reshape(nm * mp)
                    try:
                        W[c, :] = np.linalg.lstsq(A, t_c, rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                if check_decomposition(U, V, W, tol=1e-8):
                    return U, V, W
            
            # Final check
            if check_decomposition(U, V, W, tol=1e-6):
                return U, V, W
        
        return None
    
    # Try ranks from lower to upper
    # Lower bound: at least max(nm, mp, pn) / something... 
    # Actually, a trivial lower bound is max(nm, mp, pn) but that's not tight
    # For this problem, let's try from a reasonable range
    
    # Known: schoolbook is 64. Try to go lower.
    # The tensor has 64 non-zeros, so rank >= 1 is trivial.
    # But we need exact decomposition.
    
    # Let's try ranks: 64, 60, 56, 52, ... down to some point
    # But this might take too long. Let's be strategic.
    
    # Try a few promising ranks
    candidate_ranks = [64, 60, 56, 52, 48, 44, 40, 36, 32]
    
    for R in candidate_ranks:
        if time.time() > deadline:
            break
        
        result = try_rank(R, n_restarts=3)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_U = U
            best_V = V
            best_W = W
            break  # Found a valid decomposition, try lower ranks
    
    # If no lower rank found, use schoolbook
    if best_U is None:
        best_rank = n * m * p
        U = np.zeros((nm, best_rank))
        V = np.zeros((mp, best_rank))
        W = np.zeros((pn, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_U = U
        best_V = V
        best_W = W
    
    # Convert to lists for output
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
