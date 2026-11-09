import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _verify_decomposition(U, V, W, atol=1e-9):
    """Check exact decomposition of the matmul tensor."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    # Reshape for einsum
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    # S[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    S = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r, optimize=True)
    E = np.ones((n, m, p))
    
    return np.allclose(S, E, atol=atol)

def _round_to_half_integers(arr):
    """Round array to nearest half-integer (0.5 increments)."""
    return np.round(arr * 2) / 2

def _als_decompose(n, m, p, rank, rng, max_iters=100, tol=1e-10):
    """
    Attempt to find a rank-r decomposition using Alternating Least Squares.
    This is a new approach: Optimization-based search.
    """
    U = rng.normal(0, 1, size=(n * m, rank))
    V = rng.normal(0, 1, size=(m * p, rank))
    W = rng.normal(0, 1, size=(p * n, rank))
    
    # Pre-compute target tensor slices for efficient updates?
    # T[i,j,k] = 1.
    # We want min || T - sum U_t V_t W_t ||_F^2
    
    # For each t, we can update U_t, V_t, W_t.
    # However, standard ALS for CP decomposition is:
    # U_t = argmin || T - sum U_t V_t W_t ||_F^2
    # This is equivalent to solving a linear system for each t.
    
    # Let's implement a simplified ALS step.
    # Actually, for small tensors, we can just use gradient descent on the flattened parameters.
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    lr = 0.01
    
    for it in range(max_iters):
        # Compute gradient
        # S = einsum('ijt,jkt,kit->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
        # Error = S - 1
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        S = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r, optimize=True)
        Err = S - 1.0
        
        # Gradient w.r.t U[i,j,t]:
        # dL/dU[i,j,t] = -2 * sum_{k} Err[i,j,k] * V[j,k,t] * W[k,i,t]
        # Note: V is (m,p,rank), so V[j,k,t]. W is (p,n,rank), so W[k,i,t].
        
        # We can compute this via einsum
        # grad_U[i,j,t] = -2 * einsum('ijk,jkt,kit->ijt', Err, V_r, W_r)
        grad_U = -2 * np.einsum('ijk,jkt,kit->ijt', Err, V_r, W_r, optimize=True)
        grad_V = -2 * np.einsum('ijk,ijt,kit->jkt', Err, U_r, W_r, optimize=True)
        grad_W = -2 * np.einsum('ijk,ijt,jkt->kit', Err, U_r, V_r, optimize=True)
        
        # Update
        U -= lr * grad_U.reshape(n*m, rank)
        V -= lr * grad_V.reshape(m*p, rank)
        W -= lr * grad_W.reshape(p*n, rank)
        
        # Check convergence
        if np.linalg.norm(Err) < tol:
            break
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p # 54
    
    # Baseline: Schoolbook
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
                
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    rng = np.random.default_rng(42)
    start_time = time.time()
    time_limit = 90.0 # Leave buffer
    
    # Try ranks lower than schoolbook
    # We prioritize ranks that are multiples or related to sub-problems
    # But we'll just iterate down.
    for target_rank in range(schoolbook_rank - 1, 30, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few initializations for this rank
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
                
            U_cand, V_cand, W_cand = _als_decompose(n, m, p, target_rank, rng, max_iters=50)
            
            # Round to half-integers
            U_rounded = _round_to_half_integers(U_cand)
            V_rounded = _round_to_half_integers(V_cand)
            W_rounded = _round_to_half_integers(W_cand)
            
            # Verify
            if _verify_decomposition(U_rounded, V_rounded, W_rounded):
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                break # Found one, try lower rank
        
    # Return best found
    U, V, W = best_U, best_V, best_W
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
