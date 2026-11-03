"""Matrix multiplication 5x5x5: search for low-rank exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    best_rank = n * m * p  # schoolbook rank = 125
    best_factors = None

    # The tensor for 5x5x5 matrix mult: T[i,j,k] = 1 for all i,j,k
    # We need rank-r factors U(25,r), V(25,r), W(25,r) such that:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Strategy: Use structured search. Start with schoolbook and try to reduce rank
    # by finding linear dependencies among the 125 basis tensors.
    
    # Each rank-1 term corresponds to a choice of (i0,j0,k0) pattern.
    # The schoolbook uses 125 terms: one for each (i,j,k).
    # We want to find a smaller set of "gates" that span the same space.
    
    # Alternative: Directly solve using the fact that we need
    # U ⊗ V ⊗ W (in tensor product sense with the index mapping) = ones(5,5,5)
    
    # Let's use a parametrized approach: assume U, V, W have a specific structure
    # and solve for free parameters.
    
    # Actually, let's try a more systematic ALS with exact verification.
    
    def check_decomposition(U, V, W, atol=1e-8):
        """Check if U, V, W form an exact decomposition."""
        r = U.shape[1]
        # Compute the resulting tensor
        result = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            # result[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        return np.max(np.abs(result - 1.0)) < atol
    
    def als_decompose(rank, max_iter=200, tol=1e-12, seed=42):
        """ALS for rank-r decomposition of all-ones 5x5x5 tensor."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except:
                        x = np.zeros(rank)
                    U_new[i * m + j, :] = x
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    except:
                        x = np.zeros(rank)
                    V_new[j * p + k, :] = x
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except:
                        x = np.zeros(rank)
                    W_new[k * n + i, :] = x
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c]
                        err += (val - 1.0) ** 2
            err = np.sqrt(err / (n * m * p))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            
            U, V, W = U_new, V_new, W_new
            
            if err < tol:
                break
        
        return best_U, best_V, best_W, best_err

    # Try ranks from 55 upward (5x5x5 should need more than 4x4x4's 49)
    for r in range(55, 126):
        if time.time() > deadline - 15:
            break
        
        max_seeds = 2 if r < 70 else 1
        found = False
        
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            
            U, V, W, err = als_decompose(r, max_iter=150, tol=1e-12, seed=seed * 1000 + r)
            
            if err < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_decomposition(U_r, V_r, W_r, atol=1e-6):
                    best_rank = r
                    best_factors = (U_r, V_r, W_r)
                    found = True
                    break
                elif check_decomposition(U, V, W, atol=1e-8):
                    best_rank = r
                    best_factors = (U, V, W)
                    found = True
                    break
        
        if found:
            break
    
    # Fallback to schoolbook
    if best_factors is None or best_rank >= n * m * p:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = best_factors
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
