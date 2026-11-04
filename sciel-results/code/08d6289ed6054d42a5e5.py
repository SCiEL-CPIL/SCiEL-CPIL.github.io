import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 3]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """
    Verify that the decomposition U, V, W correctly decomposes the 3x3 matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape for easier index handling: U (n, m, r), V (m, p, r), W (p, n, r)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if all entries are 1
    return np.allclose(T, 1.0, atol=tol)

def _is_half_integer(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol

def _check_half_integers(U, V, W, tol=1e-9):
    for M in [U, V, W]:
        for row in M:
            for x in row:
                if not _is_half_integer(x, tol):
                    return False
    return True

def _get_schoolbook(n=3, m=3, p=3):
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

def _round_to_half_ints(M, tol=1e-6):
    """Round a matrix to nearest half-integers and check if close enough."""
    M_rounded = [[round(x * 2) / 2 for x in row] for row in M]
    M_np = np.array(M, dtype=float)
    M_round_np = np.array(M_rounded, dtype=float)
    if np.max(np.abs(M_np - M_round_np)) < tol:
        return M_rounded
    return None

def _compute_loss_and_grad(U, V, W, n=3, m=3, p=3):
    """
    Compute loss and gradients for the tensor decomposition.
    U, V, W are numpy arrays of shape (n*m, r), (m*p, r), (p*n, r).
    """
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Loss: sum (T - 1)^2
    diff = T - 1.0
    loss = np.sum(diff**2)
    
    # Gradients
    # dLoss/dT = 2 * (T - 1)
    dLoss_dT = 2.0 * diff
    
    # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] (summed over k? No, T is indexed by i,j,k)
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # dT[i,j,k]/dU[i,j,c] = V[j,k,c] * W[k,i,c]
    # dLoss/dU[i,j,c] = sum_{k} dLoss_dT[i,j,k] * V[j,k,c] * W[k,i,c]
    
    # dLoss/dU = einsum('ijk,jkc,kic->ijc', dLoss_dT, V_reshaped, W_reshaped)
    dU = np.einsum('ijk,jkc,kic->ijc', dLoss_dT, V_reshaped, W_reshaped)
    
    # dT[i,j,k]/dV[j,k,c] = U[i,j,c] * W[k,i,c]
    # dLoss/dV[j,k,c] = sum_{i} dLoss_dT[i,j,k] * U[i,j,c] * W[k,i,c]
    # dLoss/dV = einsum('ijk,ijc,kic->jkc', dLoss_dT, U_reshaped, W_reshaped)
    dV = np.einsum('ijk,ijc,kic->jkc', dLoss_dT, U_reshaped, W_reshaped)
    
    # dT[i,j,k]/dW[k,i,c] = U[i,j,c] * V[j,k,c]
    # dLoss/dW[k,i,c] = sum_{j} dLoss_dT[i,j,k] * U[i,j,c] * V[j,k,c]
    # dLoss/dW = einsum('ijk,ijc,jkc->kic', dLoss_dT, U_reshaped, V_reshaped)
    dW = np.einsum('ijk,ijc,jkc->kic', dLoss_dT, U_reshaped, V_reshaped)
    
    return loss, dU.reshape(n*m, r), dV.reshape(m*p, r), dW.reshape(p*n, r)

def _optimize_rank(r, n=3, m=3, p=3, time_limit=30.0, n_restarts=10, seed=42):
    """
    Use gradient descent to find a rank-r decomposition.
    Returns (U, V, W) as lists of lists, or None if failed.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    best_loss = float('inf')
    best_solution = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Gradient descent with adaptive learning rate
        lr = 0.01
        prev_loss = float('inf')
        
        for step in range(1000):
            if time.time() - start_time > time_limit:
                break
            
            loss, dU, dV, dW = _compute_loss_and_grad(U, V, W, n, m, p)
            
            if loss < best_loss:
                best_loss = loss
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-10:
                break
                
            # Simple momentum-like update or just gradient step
            # Check for divergence
            if not np.isfinite(loss):
                break
                
            # Adaptive LR: if loss increases, reduce LR
            if loss > prev_loss * 1.1:
                lr *= 0.5
            else:
                lr = min(lr * 1.05, 0.1)
                
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            prev_loss = loss
        
        # After optimization, round to half-integers
        if best_loss < 1e-4: # Only try to round if we got close
            U_r = _round_to_half_ints(best_solution[0].tolist())
            V_r = _round_to_half_ints(best_solution[1].tolist())
            W_r = _round_to_half_ints(best_solution[2].tolist())
            
            if U_r is not None and V_r is not None and W_r is not None:
                if _verify_decomposition(U_r, V_r, W_r):
                    if _check_half_integers(U_r, V_r, W_r):
                        return {"U": U_r, "V": V_r, "W": W_r}
    
    return None

def _find_decomposition(n=3, m=3, p=3, max_rank=27, time_limit=100.0):
    """
    Attempt to find a rank-minimal decomposition.
    Try ranks from 23 down to 19, then fall back to schoolbook.
    """
    start_time = time.time()
    
    # Try to find solutions with decreasing rank
    # Focus on 23, 24, 25, 26 as these are the most likely candidates for improvement over 27
    for r in [23, 24, 25, 26]:
        if time.time() - start_time > time_limit - 5:
            break
        
        # Allocate more time for higher ranks (easier to find) and less for 23
        time_for_this_rank = 20.0
        result = _optimize_rank(r, n, m, p, time_limit=time_for_this_rank, n_restarts=5)
        
        if result is not None:
            return result
    
    # Fallback to Schoolbook
    return _get_schoolbook(n, m, p)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Deadline guard
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Try to find a better decomposition
    result = _find_decomposition(n, m, p, max_rank=27, time_limit=DEADLINE - SAFETY_MARGIN)
    
    # Safety check: ensure we didn't exceed time
    if time.time() > deadline:
        return _get_schoolbook(n, m, p)
        
    return result
