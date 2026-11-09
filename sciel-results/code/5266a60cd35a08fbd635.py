"""Rewritten program for 3x5x7 matrix multiplication tensor rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM, MP, PN = N*M, M*P, P*N
TOTAL = N*M*P

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            u_row = U[i*m+j]  # (rank,)
            for k in range(p):
                val = np.sum(u_row * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _is_halfint(arr, tol=1e-8):
    """Check if all entries are within tol of half-integers."""
    doubled = arr * 2.0
    nearest = np.round(doubled)
    return np.max(np.abs(doubled - nearest)) < tol

def _schoolbook_decomposition(n, m, p):
    """Generate the standard schoolbook decomposition of rank n*m*p."""
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

def _loss_and_grads_fast(U, V, W, n, m, p):
    """Compute loss and gradients using vectorized operations where possible."""
    rank = U.shape[1]
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for k in range(p):
            w_row = W[k*n+i]  # (rank,)
            for j in range(m):
                u_row = U[i*m+j]  # (rank,)
                v_row = V[j*p+k]  # (rank,)
                val = np.sum(u_row * v_row * w_row)
                err = val - 1.0
                loss += err * err
                
                uvw = v_row * w_row  # (rank,)
                uwr = u_row * w_row
                uvv = u_row * v_row
                
                grad_U[i*m+j] += err * uvw
                grad_V[j*p+k] += err * uwr
                grad_W[k*n+i] += err * uvv
    
    return loss, grad_U, grad_V, grad_W

def _optimize(U, V, W, n, m, p, max_iters=5000, lr_init=0.05, tol=1e-10, time_limit=None):
    """Optimize the decomposition using gradient descent with adaptive learning rate."""
    rank = U.shape[1]
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = lr_init
    prev_loss = float('inf')
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        loss, gU, gV, gW = _loss_and_grads_fast(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Adaptive learning rate
        if loss > prev_loss:
            lr *= 0.5
        elif it > 0 and loss < prev_loss * 0.9:
            lr = min(lr * 1.1, 0.5)
        prev_loss = loss
        
        # Gradient step
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Periodic check for half-integer convergence
        if it % 200 == 0 and best_loss < 1e-4:
            U_h = _round_to_halfint(best_U)
            V_h = _round_to_halfint(best_V)
            W_h = _round_to_halfint(best_W)
            if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
                return U_h, V_h, W_h, best_loss
    
    # Final check
    if best_loss < 1e-4:
        U_h = _round_to_halfint(best_U)
        V_h = _round_to_halfint(best_V)
        W_h = _round_to_halfint(best_W)
        if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
            return U_h, V_h, W_h, best_loss
    
    return best_U, best_V, best_W, best_loss

def _try_rank_warm_start(r, n, m, p, time_limit=None):
    """
    Try to find a rank-r decomposition by removing (n*m*p - r) terms from the schoolbook
    decomposition and optimizing the remaining terms.
    This is a warm start strategy.
    """
    if time_limit is not None and time.time() > time_limit:
        return None
    
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    full_rank = n * m * p
    terms_to_remove = full_rank - r
    
    if terms_to_remove <= 0:
        return U_school, V_school, W_school
    
    # We try to remove random terms and see if the remaining ones can adjust to cover the tensor.
    # Since the tensor is "sparse" in the sense that each term covers one (i,j,k), 
    # removing a term creates a "hole". The other terms must expand to cover it.
    
    # We will try a few random subsets of terms to remove.
    # For small numbers of removals, we can try all combinations or random samples.
    
    num_attempts = 10
    indices = list(range(full_rank))
    
    for attempt in range(num_attempts):
        if time_limit is not None and time.time() > time_limit:
            break
            
        # Select terms to keep
        if terms_to_remove < full_rank:
            keep_indices = list(range(full_rank))
            # Randomly select indices to remove
            remove_indices = np.random.choice(full_rank, terms_to_remove, replace=False)
            keep_indices = [i for i in keep_indices if i not in remove_indices]
        else:
            keep_indices = []
            
        if not keep_indices:
            continue
            
        U_init = U_school[:, keep_indices]
        V_init = V_school[:, keep_indices]
        W_init = W_school[:, keep_indices]
        
        # Optimize
        U_opt, V_opt, W_opt, loss = _optimize(
            U_init, V_init, W_init, n, m, p, 
            max_iters=2000, lr_init=0.05, tol=1e-10, time_limit=time_limit
        )
        
        if loss < 1e-8:
            U_h = _round_to_halfint(U_opt)
            V_h = _round_to_halfint(V_opt)
            W_h = _round_to_halfint(W_opt)
            if _verify_fast(U_h, V_h, W_h, n, m, p, tol=1e-6):
                return U_h, V_h, W_h
                
    return None

def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # 1. Schoolbook Decomposition (Rank 105)
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # 2. Try to improve using Warm Start (removing terms from schoolbook)
    # We try ranks 104, 103, 102... down to a certain limit or time limit.
    # Since the search space is discrete and hard, we limit the number of attempts.
    
    # We prioritize higher ranks (closer to schoolbook) as they are more likely to be solvable.
    for r in range(best_rank - 1, best_rank - 10, -1):
        if time.time() > deadline - 5:
            break
            
        result = _try_rank_warm_start(r, n, m, p, time_limit=deadline - 5)
        
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            # If we find a lower rank, we can continue trying to go lower,
            # but we must be careful with time.
            
    return best_result
