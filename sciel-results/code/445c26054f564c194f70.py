"""Tensor decomposition search for 3x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
n, m, p = DIMENSIONS
N = n * m  # 9
M = m * p  # 18
P = p * n  # 18

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify(U, V, W, tol=1e-8):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The multiplication tensor has T[i,j,k] = delta (i.e., 1 for all (i,j,k) in the standard basis)
    # Actually, the matrix multiplication tensor is:
    # T[i, j, k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k
    # The decomposition requires: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Efficient check using einsum
    U3 = U.reshape(n, m, rank)  # U[i,j,c]
    V3 = V.reshape(m, p, rank)  # V[j,k,c]
    W3 = W.reshape(p, n, rank)  # W[k,i,c]
    
    # Compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a tensor contraction
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=tol)

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 54."""
    rank = n * m * p
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_landscape(rank, max_iter=2000, seed=42):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Uses a smooth optimization approach on the tensor equation.
    """
    if _time_left() < 10:
        return None
    
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(N, rank)
    V = rng.randn(M, rank)
    W = rng.randn(P, rank)
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Use gradient descent on the residual
    # Residual R[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1
    # Loss = sum_{i,j,k} R[i,j,k]^2
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if _time_left() < 15:
            break
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Forward pass
        R = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1.0
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_loss < 1e-10:
                break
        
        # Compute gradients
        # dL/dU[i,j,c] = 2 * sum_{k} R[i,j,k] * V[j,k,c] * W[k,i,c]
        # dL/dV[j,k,c] = 2 * sum_{i} R[i,j,k] * U[i,j,c] * W[k,i,c]
        # dL/dW[k,i,c] = 2 * sum_{j} R[i,j,k] * U[i,j,c] * V[j,k,c]
        
        dU = np.einsum('ijk,jkc,kic->ijc', R, V3, W3) * 2
        dV = np.einsum('ijk,ijc,kic->jkc', R, U3, W3) * 2
        dW = np.einsum('ijk,ijc,jkc->kic', R, U3, V3) * 2
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive learning rate
        if it % 100 == 99:
            if loss < best_loss * 0.9:
                lr *= 1.2
            else:
                lr *= 0.8
            lr = max(lr, 1e-5)
    
    if best_loss < 1e-8:
        # Verify more carefully
        U3 = best_U.reshape(n, m, rank)
        V3 = best_V.reshape(m, p, rank)
        W3 = best_W.reshape(p, n, rank)
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        if np.allclose(result, 1.0, atol=1e-7):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _try_integer_search(rank, max_iter=5000, seed=0):
    """
    Try to find an integer/half-integer decomposition using random restarts
    and local search.
    """
    if _time_left() < 10:
        return None
    
    rng = np.random.RandomState(seed)
    
    for restart in range(10):
        if _time_left() < 15:
            break
        
        # Initialize with small random integers/half-integers
        U = rng.choice([-2, -1, -0.5, 0.5, 1, 2], size=(N, rank))
        V = rng.choice([-2, -1, -0.5, 0.5, 1, 2], size=(M, rank))
        W = rng.choice([-2, -1, -0.5, 0.5, 1, 2], size=(P, rank))
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for it in range(max_iter // 10):
            if _time_left() < 15:
                break
            
            R = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1.0
            loss = np.sum(R**2)
            
            if loss < 1e-8:
                result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                if np.allclose(result, 1.0, atol=1e-7):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Random perturbation
            if rng.random() < 0.3:
                idx = rng.randint(0, N)
                col = rng.randint(0, rank)
                U[idx, col] += rng.choice([-1, 1]) * 0.5
                U3 = U.reshape(n, m, rank)
            elif rng.random() < 0.6:
                idx = rng.randint(0, M)
                col = rng.randint(0, rank)
                V[idx, col] += rng.choice([-1, 1]) * 0.5
                V3 = V.reshape(m, p, rank)
            else:
                idx = rng.randint(0, P)
                col = rng.randint(0, rank)
                W[idx, col] += rng.choice([-1, 1]) * 0.5
                W3 = W.reshape(p, n, rank)
    
    return None

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook()
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But tensor rank is likely higher. Let's try a range.
    
    for rank in range(24, 55):  # Try ranks 24 to 54
        if _time_left() < 30:
            break
        
        # Try landscape method
        result = _try_landscape(rank, max_iter=500, seed=rank)
        if result is not None:
            return result
        
        if _time_left() < 20:
            break
        
        # Try integer search for small ranks
        if rank <= 36:
            result = _try_integer_search(rank, max_iter=2000, seed=rank)
            if result is not None:
                return result
    
    # Fallback to schoolbook
    return schoolbook
