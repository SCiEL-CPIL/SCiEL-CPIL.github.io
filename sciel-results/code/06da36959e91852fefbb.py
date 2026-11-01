import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify exact decomposition of the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        
        if U.shape != (n*m, r) or V.shape != (m*p, r) or W.shape != (p*n, r):
            return False
        
        # Build the target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Reconstruct tensor from decomposition
        T_rec = np.zeros((n*m, m*p, p*n))
        for c in range(r):
            T_rec += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
        
        # Check all entries
        if np.max(np.abs(T_rec - T)) > tol:
            return False
        return True
    
    def try_rank_als(r, seed, max_iter=200):
        """Try to find rank-r decomposition using ALS with multiple restarts."""
        np.random.seed(seed)
        
        # Try several random initializations
        for attempt in range(3):
            if time.time() > deadline - 15:
                return None
            
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Build target tensor
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] = 1.0
            
            prev_residual = float('inf')
            converged = False
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Update U: solve U @ (V⊙W) = T (reshaped)
                # Khatri-Rao of V and W
                KR_VW = np.einsum('bc,dc->bdc', V, W).reshape(m*p*p*n, r)
                T_flat = T.reshape(n*m, m*p*p*n).T  # (m*p*p*n, n*m)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(KR_VW.T, T_flat, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Update V
                KR_UW = np.einsum('ac,dc->adc', U, W).reshape(n*m*p*n, r)
                T_flat_v = T.reshape(m*p, n*m*p*n).T
                try:
                    V_new, _, _, _ = np.linalg.lstsq(KR_UW.T, T_flat_v, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Update W
                KR_UV = np.einsum('ac,bc->abc', U, V).reshape(n*m*m*p, r)
                T_flat_w = T.reshape(p*n, n*m*m*p).T
                try:
                    W_new, _, _, _ = np.linalg.lstsq(KR_UV.T, T_flat_w, rcond=None)
                    W = W_new.T
                except:
                    break
                
                # Check convergence
                T_rec = np.zeros((n*m, m*p, p*n))
                for c in range(r):
                    T_rec += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
                residual = np.max(np.abs(T_rec - T))
                
                if residual < 1e-10:
                    converged = True
                    break
                if residual > prev_residual * 1.01:
                    break
                prev_residual = residual
            
            if converged and verify_decomposition(U, V, W, tol=1e-8):
                return U, V, W
        
        return None
    
    def try_rank_gradient(r, seed, max_iter=500):
        """Try gradient descent with momentum for rank-r decomposition."""
        np.random.seed(seed)
        
        for attempt in range(2):
            if time.time() > deadline - 15:
                return None
            
            U = np.random.randn(n*m, r) * 0.05
            V = np.random.randn(m*p, r) * 0.05
            W = np.random.randn(p*n, r) * 0.05
            
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] = 1.0
            
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Compute current tensor
                T_rec = np.zeros((n*m, m*p, p*n))
                for c in range(r):
                    T_rec += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
                
                loss = np.sum((T_rec - T)**2)
                
                if loss < 1e-12:
                    break
                if loss > prev_loss * 1.1 and it > 10:
                    break
                prev_loss = loss
                
                # Compute gradients
                # dL/dU[a,c] = 2 * sum_{b,d} (T_rec[a,b,d] - T[a,b,d]) * V[b,c] * W[d,c]
                diff = T_rec - T
                grad_U = np.einsum('abd,bc,dc->ac', diff, V, W)
                grad_V = np.einsum('abd,ac,dc->bc', diff, U, W)
                grad_W = np.einsum('abd,ac,bc->dc', diff, U, V)
                
                # Update with momentum
                vU = momentum * vU + lr * grad_U
                vV = momentum * vV + lr * grad_V
                vW = momentum * vW + lr * grad_W
                
                U -= vU
                V -= vV
                W -= vW
                
                # Decay learning rate
                lr *= 0.995
            
            if verify_decomposition(U, V, W, tol=1e-7):
                return U, V, W
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Also, for matrix multiplication, rank >= max(n*m, m*p, p*n)
    
    best_result = None
    
    # First try ALS for lower ranks
    for r in range(27, 55):
        if time.time() > deadline - 20:
            break
        
        # Try a few seeds
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            result = try_rank_als(r, seed=r*100+seed, max_iter=100)
            if result is not None:
                U, V, W = result
                best_result = (r, U, V, W)
                break
        
        if best_result is not None:
            break
    
    # If ALS didn't work, try gradient descent
    if best_result is None:
        for r in range(27, 55):
            if time.time() > deadline - 20:
                break
            
            for seed in range(2):
                if time.time() > deadline - 20:
                    break
                result = try_rank_gradient(r, seed=r*200+seed, max_iter=300)
                if result is not None:
                    U, V, W = result
                    best_result = (r, U, V, W)
                    break
            
            if best_result is not None:
                break
    
    if best_result is not None:
        r, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
