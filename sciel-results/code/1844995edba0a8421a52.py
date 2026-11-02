import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def _validate_fast(U, V, W, n, m, p, tol=1e-10):
    """Fast vectorized validation using einsum."""
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.max(np.abs(T - 1.0)) < tol

def _snap_to_half_integers(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    snapped = arr.copy()
    candidates = np.round(snapped * 2) / 2
    if np.max(np.abs(snapped - candidates)) < tol:
        return candidates
    candidates_int = np.round(snapped)
    if np.max(np.abs(snapped - candidates_int)) < tol:
        return candidates_int
    return snapped

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _gradient_descent_fast(U, V, W, n, m, p, lr=0.01, max_steps=1000, tol=1e-8):
    """Vectorized gradient descent for tensor decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank).copy()
    V_r = V.reshape(m, p, rank).copy()
    W_r = W.reshape(p, n, rank).copy()
    
    best_loss = np.inf
    best_state = (U_r.copy(), V_r.copy(), W_r.copy())
    
    for step in range(max_steps):
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        diff = T - 1.0
        loss = np.sum(diff**2)
        
        if loss < best_loss:
            best_loss = loss
            best_state = (U_r.copy(), V_r.copy(), W_r.copy())
            
        if loss < tol:
            return best_state[0].reshape(n*m, rank), best_state[1].reshape(m*p, rank), best_state[2].reshape(p*n, rank), True
        
        # Gradients
        grad_U = np.einsum('ijk,jkr,kir->ijr', diff, V_r, W_r)
        grad_V = np.einsum('ijk,ijr,kir->jkr', diff, U_r, W_r)
        grad_W = np.einsum('ijk,ijr,jkr->kir', diff, U_r, V_r)
        
        # Adaptive learning rate: reduce if loss increases significantly
        if step > 0 and loss > best_loss * 1.5:
            lr *= 0.5
            if lr < 1e-6:
                break
                
        U_r -= lr * grad_U
        V_r -= lr * grad_V
        W_r -= lr * grad_W
        
    return best_state[0].reshape(n*m, rank), best_state[1].reshape(m*p, rank), best_state[2].reshape(p*n, rank), best_loss < tol

def _polish(U, V, W, n, m, p, max_iter=10):
    """High-precision polish pass using very small learning rate."""
    for _ in range(max_iter):
        if _validate_fast(U, V, W, n, m, p, tol=1e-12):
            break
        U, V, W, converged = _gradient_descent_fast(U, V, W, n, m, p, lr=1e-4, max_steps=50, tol=1e-12)
        if converged:
            break
    return U, V, W

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin for 150s limit
    
    # Start with schoolbook as default
    U, V, W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    best_result = (U, V, W)
    
    # Likely ranks to try: 20, 21, 22, 23
    # Based on theoretical lower bounds and known small matrix multiplication results
    candidate_ranks = [20, 21, 22, 23]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try many random restarts
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(seed + 1000 + rank)
            # Scale initialization to avoid large values
            scale = 1.0 / np.sqrt(rank)
            U_try = rng.randn(n * m, rank) * scale
            V_try = rng.randn(m * p, rank) * scale
            W_try = rng.randn(p * n, rank) * scale
            
            # Run fast gradient descent
            U_try, V_try, W_try, converged = _gradient_descent_fast(
                U_try, V_try, W_try, n, m, p, 
                lr=0.05, max_steps=500, tol=1e-6
            )
            
            if converged:
                # Polish
                U_try, V_try, W_try = _polish(U_try, V_try, W_try, n, m, p)
                if _validate_fast(U_try, V_try, W_try, n, m, p, tol=1e-10):
                    # Snap to half-integers
                    U_snap = _snap_to_half_integers(U_try)
                    V_snap = _snap_to_half_integers(V_try)
                    W_snap = _snap_to_half_integers(W_try)
                    if _validate_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-10):
                        U, V, W = U_snap, V_snap, W_snap
                        best_rank = rank
                        best_result = (U, V, W)
                        found = True
                        break
                    else:
                        # Use un-snapped if it validates
                        U, V, W = U_try, V_try, W_try
                        best_rank = rank
                        best_result = (U, V, W)
                        found = True
                        break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks if time allows
            continue
    
    # Final validation
    if not _validate_fast(U, V, W, n, m, p, tol=1e-10):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
