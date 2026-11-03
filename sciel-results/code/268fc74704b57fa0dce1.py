import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 6]
DEADLINE_SECONDS = 140  # Leave 10s margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Precompute index mappings for fast tensor reconstruction
    # For each (i,j,k) triple, we need:
    #   U index: i*m + j
    #   V index: j*p + k  
    #   W index: k*n + i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m + j, j*p + k, k*n + i))
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the tensor (within tol)."""
        R = U.shape[1]
        for (ui, vi, wi) in indices:
            val = np.dot(U[ui], V[vi]) * 0  # placeholder, need element-wise
            # Actually need sum over r of U[ui,r]*V[vi,r]*W[wi,r]
            val = np.sum(U[ui] * V[vi] * W[wi])
            if abs(val - 1.0) > tol:
                return False
        return True

    def check_exact_fast(U, V, W, tol=1e-9):
        """Vectorized exact check."""
        R = U.shape[1]
        # Build tensors: for each (ui,vi,wi) in indices, compute sum_r U[ui,r]*V[vi,r]*W[wi,r]
        # Reshape into 3D tensors
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.all(np.abs(T - 1.0) <= tol)

    def round_to_half(U, V, W):
        """Round to nearest half-integer."""
        return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

    def try_round_and_check(U, V, W, tol=1e-9):
        """Round to half-integers and check."""
        Ur, Vr, Wr = round_to_half(U, V, W)
        if check_exact_fast(Ur, Vr, Wr, tol):
            return Ur, Vr, Wr
        return None

    def april_step(U, V, W, lr=0.1):
        """One APRIL iteration: update each factor using the other two."""
        R = U.shape[1]
        # Compute effective tensors
        # For U update: U[i,j,r] -= lr * sum_{j',k'} V[j',k',r] * W[k',i,r] * (T[i,j',k'] - 1)
        # More precisely, gradient of loss w.r.t. U[ui,r] = sum_{vi,wi: (ui,vi,wi) in indices} V[vi,r]*W[wi,r]*(T[ui,vi,wi]-1)
        
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        # Compute T
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        # Update U: dU[i,j,r] = sum_{k} V[j,k,r]*W[k,i,r]*(T[i,j,k]-1)
        dU = np.einsum('jkr,kir,ijk->ijr', V_t, W_t, T - 1.0)
        U_t -= lr * dU
        
        # Recompute T with updated U
        T = np.einsum('ijr,jkr,kir->ijr', U_t, V_t, W_t)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        # Update V: dV[j,k,r] = sum_{i} U[i,j,r]*W[k,i,r]*(T[i,j,k]-1)
        dV = np.einsum('ijr,kir,ijk->jkr', U_t, W_t, T - 1.0)
        V_t -= lr * dV
        
        # Recompute T with updated V
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        # Update W: dW[k,i,r] = sum_{j} U[i,j,r]*V[j,k,r]*(T[i,j,k]-1)
        dW = np.einsum('ijr,jkr,ijk->kir', U_t, V_t, T - 1.0)
        W_t -= lr * dW
        
        return U_t.reshape(n*m, R), V_t.reshape(m*p, R), W_t.reshape(p*n, R)

    def april_refine(U, V, W, max_iter=2000, lr_init=0.05, tol=1e-10):
        """Run APRIL refinement until convergence or max_iter."""
        R = U.shape[1]
        U_t = U.reshape(n, m, R).copy()
        V_t = V.reshape(m, p, R).copy()
        W_t = W.reshape(p, n, R).copy()
        
        lr = lr_init
        best_loss = np.inf
        best_state = (U_t.copy(), V_t.copy(), W_t.copy())
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE_SECONDS:
                break
            
            # Compute T
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            loss = np.sum((T - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_state = (U_t.copy(), V_t.copy(), W_t.copy())
            
            if loss < tol:
                break
            
            # Adaptive learning rate
            if it % 100 == 0 and it > 0:
                # Decay learning rate slowly
                lr *= 0.95
            
            # APRIL step
            dU = np.einsum('jkr,kir,ijk->ijr', V_t, W_t, T - 1.0)
            U_t -= lr * dU
            
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            
            dV = np.einsum('ijr,kir,ijk->jkr', U_t, W_t, T - 1.0)
            V_t -= lr * dV
            
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            
            dW = np.einsum('ijr,jkr,ijk->kir', U_t, V_t, T - 1.0)
            W_t -= lr * dW
        
        # Return best state found
        U_best, V_best, W_best = best_state
        return U_best.reshape(n*m, R), V_best.reshape(m*p, R), W_best.reshape(p*n, R)

    def schoolbook():
        """Return the schoolbook decomposition."""
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
        return U, V, W

    # Try to find lower-rank decompositions
    # Start with ranks below schoolbook (72)
    ranks_to_try = [48, 50, 52, 55, 58, 60, 64, 66, 68, 70]
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank
    
    np.random.seed(42)
    
    for R in ranks_to_try:
        if time.time() - start_time > DEADLINE_SECONDS - 30:
            break
        
        # Multiple random restarts
        for restart in range(5):
            if time.time() - start_time > DEADLINE_SECONDS - 30:
                break
            
            # Random initialization
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Run APRIL refinement
            U, V, W = april_refine(U, V, W, max_iter=3000, lr_init=0.05)
            
            # Try rounding to half-integers
            result = try_round_and_check(U, V, W, tol=1e-9)
            if result is not None:
                Ur, Vr, Wr = result
                # Verify
                if check_exact_fast(Ur, Vr, Wr, tol=1e-9):
                    if R < best_rank:
                        best_rank = R
                        best_result = (Ur, Vr, Wr)
                        print(f"Found rank {R} decomposition!")
                        break
        
        if best_result is not None and best_rank <= R:
            break
    
    # If we found a better decomposition, return it
    if best_result is not None:
        Ur, Vr, Wr = best_result
        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
