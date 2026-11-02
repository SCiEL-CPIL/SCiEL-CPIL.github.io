import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensors: U[i,j,c], V[j,k,c], W[k,i,c]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum for efficiency
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    # Check if T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _compute_loss_and_grads(U, V, W, n, m, p):
    """Compute loss and gradients for tensor decomposition.
    
    Loss = 0.5 * sum_{i,j,k} (T[i,j,k] - 1)^2
    where T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    """
    r = U.shape[1]
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    residual = T - 1.0
    loss = 0.5 * np.sum(residual**2)
    
    # Gradients:
    # dL/dU[i,j,c] = sum_k residual[i,j,k] * V[j,k,c] * W[k,i,c]
    # dL/dV[j,k,c] = sum_i residual[i,j,k] * U[i,j,c] * W[k,i,c]
    # dL/dW[k,i,c] = sum_j residual[i,j,k] * U[i,j,c] * V[j,k,c]
    
    # Efficient computation using einsum
    # dU[i,j,c] = sum_k resid[i,j,k] * V[j,k,c] * W[k,i,c]
    # = sum_k resid[i,j,k] * (V[j,k,c] * W[k,i,c])
    # Let A[j,k,i,c] = V[j,k,c] * W[k,i,c], then dU[i,j,c] = sum_k resid[i,j,k] * A[j,k,i,c]
    
    # More efficient: 
    # dU[i,j,c] = sum_k resid[i,j,k] * V[j,k,c] * W[k,i,c]
    # We can compute this as: for each c, dU[:, :, c] = einsum('ijk,jkc,kic->ij', resid, V_t, W_t)
    # But that's O(n*m*p*r) which is fine for small dimensions
    
    dU = np.einsum('ijk,jkc,kic->ijc', residual, V_t, W_t)
    dV = np.einsum('ijk,kic,ijc->jkc', residual, W_t, U_t)
    dW = np.einsum('ijk,ijc,jkc->kic', residual, U_t, V_t)
    
    return loss, dU.reshape(n*m, r), dV.reshape(m*p, r), dW.reshape(p*n, r)

def _gradient_descent(U, V, W, n, m, p, max_iters=200, lr_init=0.01, lr_decay=0.99, deadline=None, tol=1e-12):
    """Run gradient descent to minimize the tensor decomposition loss."""
    r = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    lr = lr_init
    prev_loss = np.inf
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        loss, dU, dV, dW = _compute_loss_and_grads(U, V, W, n, m, p)
        
        if loss < tol:
            return U, V, W, loss, True
        
        # Check for convergence
        if prev_loss - loss < 1e-15:
            break
        prev_loss = loss
        
        # Gradient step with learning rate decay
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        lr *= lr_decay
    
    final_loss, _, _, _ = _compute_loss_and_grads(U, V, W, n, m, p)
    return U, V, W, final_loss, final_loss < tol

def _try_optimization(n, m, p, target_rank, num_trials, deadline, best_rank):
    """Try random initializations followed by gradient descent."""
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with small values
        scale = 0.1
        U = scale * np.random.randn(n * m, target_rank)
        V = scale * np.random.randn(m * p, target_rank)
        W = scale * np.random.randn(p * n, target_rank)
        
        # Run gradient descent
        U_opt, V_opt, W_opt, loss, converged = _gradient_descent(
            U, V, W, n, m, p, 
            max_iters=300, lr_init=0.05, lr_decay=0.995,
            deadline=deadline, tol=1e-12
        )
        
        if converged and _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
            return U_opt, V_opt, W_opt
    
    return None

def _try_structured_patterns(n, m, p, deadline, best_rank):
    """Try various structured patterns that might give lower rank."""
    candidate_ranks = [45, 42, 40, 38, 36, 35, 34, 33, 32, 30]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        # Try optimization for this rank
        result = _try_optimization(n, m, p, target_rank, 5, deadline, best_rank)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find better decompositions
    result = _try_structured_patterns(n, m, p, deadline, best_rank)
    if result is not None:
        U_try, V_try, W_try = result
        if U_try.shape[1] < best_rank:
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                best_rank = U_try.shape[1]
    
    # If time remains, try more ranks
    if time.time() < deadline - 20:
        for target_rank in [28, 26, 25, 24, 23, 22, 21, 20]:
            if time.time() > deadline - 10:
                break
            if target_rank >= best_rank:
                continue
            result = _try_optimization(n, m, p, target_rank, 3, deadline, best_rank)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = U_try.shape[1]
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
