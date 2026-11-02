import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    # Precompute index mappings for vectorized operations
    # We need to compute T_decomp = sum_r U_r V_r W_r
    # U_r is (n, m), V_r is (m, p), W_r is (p, n)
    # T_decomp[i, j, k] = sum_r U_r[i, j] V_r[j, k] W_r[k, i]
    
    # To vectorize, we can use einsum.
    # T_decomp = np.einsum('ijr,jkr,ki r->ijk', U, V, W) ? No, U is (nm, R)
    # Let's reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    
    def verify_decomposition_fast(U, V, W, tol=1e-9):
        """Faster verification using tensor operations."""
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        T = np.einsum('ij,jk->ijk', A, B)
        
        U_r = U.T.reshape(R, n, m) # R, n, m
        V_r = V.T.reshape(R, m, p) # R, m, p
        W_r = W.T.reshape(R, p, n) # R, p, n
        
        # T_decomp[i,j,k] = sum_r U_r[r,i,j] V_r[r,j,k] W_r[r,k,i]
        T_decomp = np.einsum('rij,rjk,rki->ijk', U_r, V_r, W_r)
        
        return np.allclose(T, T_decomp, atol=tol)

    def gradient_step_vectorized(U, V, W, lr=0.01):
        """One gradient descent step to minimize ||T - T_decomp||^2."""
        # Use a fixed test matrix for stability? Or random?
        # Using random makes it stochastic gradient descent.
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        T = np.einsum('ij,jk->ijk', A, B)
        
        R = U.shape[1]
        U_r = U.T.reshape(R, n, m)
        V_r = V.T.reshape(R, m, p)
        W_r = W.T.reshape(R, p, n)
        
        T_decomp = np.einsum('rij,rjk,rki->ijk', U_r, V_r, W_r)
        residual = T_decomp - T
        loss = np.sum(residual**2)
        
        # Gradients
        # dL/dU_r[i,j] = sum_{k} residual[i,j,k] V_r[r,j,k] W_r[r,k,i]
        # dU_r = einsum('ijk,rjk,rki->rij', residual, V_r, W_r)
        dU_r = np.einsum('ijk,rjk,rki->rij', residual, V_r, W_r)
        
        # dL/dV_r[j,k] = sum_{i} residual[i,j,k] U_r[r,i,j] W_r[r,k,i]
        # dV_r = einsum('ijk,rij,rki->rjk', residual, U_r, W_r)
        dV_r = np.einsum('ijk,rij,rki->rjk', residual, U_r, W_r)
        
        # dL/dW_r[k,i] = sum_{j} residual[i,j,k] U_r[r,i,j] V_r[r,j,k]
        # dW_r = einsum('ijk,rij,rjk->rki', residual, U_r, V_r)
        dW_r = np.einsum('ijk,rij,rjk->rki', residual, U_r, V_r)
        
        dU = dU_r.reshape(R, nm).T
        dV = dV_r.reshape(R, mp).T
        dW = dW_r.reshape(R, pn).T
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        return U, V, W, loss

    # Try ranks from rank_sb - 1 down
    # Start with 83, 82, etc.
    for R in range(rank_sb - 1, max(rank_sb - 20, 50), -1):
        if time.time() > deadline:
            break
            
        found = False
        # Increase restarts if time permits
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U, V, W = random_decomposition(R, seed=42 + restart)
            
            # Do a few gradient steps
            # Adaptive learning rate?
            loss = float('inf')
            for step in range(50): # Increased steps
                if time.time() > deadline:
                    break
                U, V, W, loss = gradient_step_vectorized(U, V, W, lr=0.01)
                if loss < 1e-10:
                    break
                    
            if loss < 1e-9:
                # Verify more carefully
                if verify_decomposition_fast(U, V, W, tol=1e-8):
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def random_decomposition(R, seed=None):
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(nm, R)
    V = np.random.randn(mp, R)
    W = np.random.randn(pn, R)
    return U, V, W
