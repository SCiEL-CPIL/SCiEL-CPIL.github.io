"""Matrix multiplication 2x5x6 - block decomposition with term elimination."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
n, m, p = DIMENSIONS
DEADLINE = time.time() + 140  # safety margin before 150s kill

def _reconstruct_tensor(U, V, W):
    """Reconstruct T[i,j,k] from factor matrices.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return T

def _verify(U, V, W, tol=1e-9):
    """Verify decomposition. Returns max error."""
    T = _reconstruct_tensor(U, V, W)
    return np.max(np.abs(T - 1.0))

def _schoolbook():
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_random_rank(target_rank, n_trials=10, max_iter=200, lr=0.05, seed=42):
    """Try to find a rank-target_rank decomposition via random init + gradient descent."""
    for trial in range(n_trials):
        if time.time() > DEADLINE:
            return None
        
        rng = np.random.RandomState(seed + trial * 7)
        U = rng.randn(n*m, target_rank) * 0.3
        V = rng.randn(m*p, target_rank) * 0.3
        W = rng.randn(p*n, target_rank) * 0.3
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            
            # Compute reconstruction and error
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            err = np.max(np.abs(T - 1.0))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-8:
                return best_U, best_V, best_W
            
            # Compute gradients
            # dL/dU[i*m+j,r] = sum_k (T[i,j,k]-1) * V_r[j,k,r] * W_r[k,i,r]
            # dL/dV[j*p+k,r] = sum_i (T[i,j,k]-1) * U_r[i,j,r] * W_r[k,i,r]
            # dL/dW[k*n+i,r] = sum_j (T[i,j,k]-1) * U_r[i,j,r] * V_r[j,k,r]
            
            E = T - 1.0  # (n, m, p)
            
            # Gradient for U: (n, m, rank)
            gU = np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
            # Gradient for V: (m, p, rank)
            gV = np.einsum('ijk,ijr,kir->jpr', E, U_r, W_r)
            # Gradient for W: (p, n, rank)
            gW = np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
            
            # Update
            U -= lr * gU.reshape(n*m, target_rank)
            V -= lr * gV.reshape(m*p, target_rank)
            W -= lr * gW.reshape(p*n, target_rank)
        
        # Check best found
        if best_err < 1e-8:
            return best_U, best_V, best_W
    
    return None

def construct():
    start_time = time.time()
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook()
    err_sb = _verify(U_sb, V_sb, W_sb)
    if err_sb > 1e-6:
        raise ValueError("Schoolbook verification failed")
    
    best_rank = n * m * p
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Try from 30 upward
    
    for target_rank in range(30, best_rank):
        if time.time() > DEADLINE - 10:
            break
        
        result = _try_random_rank(target_rank, n_trials=15, max_iter=300, lr=0.05, seed=target_rank)
        if result is not None:
            U, V, W = result
            err = _verify(U, V, W)
            if err < 1e-7:
                best_rank = target_rank
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                # Found a valid decomposition, try to go lower
                continue
        # If we can't find one at this rank, higher ranks are more likely
        # but let's still try a few more
    
    return best_result
