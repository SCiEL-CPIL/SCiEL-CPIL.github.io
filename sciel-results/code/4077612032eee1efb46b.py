"""Matrix multiplication tensor decomposition search for 3x4 * 4x4."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]
N, M, P = DIMENSIONS

# Deadline guard
_START_TIME = time.time()
_TIMEOUT = 140.0  # leave 10s safety margin

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] where output index is (k*n+i, j) for W[k*n+i, r].
    
    The tensor we need to decompose: T[(i,j), (j,k), (k,i)] = 1 for valid indices.
    Actually, the matrix multiplication tensor is T[i,j,k] = 1 where:
    - U has shape (n*m, rank), indexed by (i,j)
    - V has shape (m*p, rank), indexed by (j,k)  
    - W has shape (p*n, rank), indexed by (k,i)
    
    The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    # We want: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is equivalent to the tensor T[i,j,k] = 1
    return np.ones((N, M, P))

def _check_decomposition(U, V, W, rank):
    """Check if the decomposition is exact (within floating point tolerance)."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # Reshape and check
    U_reshaped = U.reshape(N, M, rank)
    V_reshaped = V.reshape(M, P, rank)
    W_reshaped = W.reshape(P, N, rank)
    
    # Compute the tensor product sum over rank dimension
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if all entries are close to 1
    return np.allclose(result, 1.0, atol=1e-9)

def _check_decomposition_fast(U, V, W, rank):
    """Faster check using matrix multiplication tricks."""
    # For fixed j: sum_r U[:,j,r] * V[j,:,r] * W[:,j,r]... no that's not right.
    # Let's just use the einsum approach but with numpy for speed
    U_r = U.reshape(N, M, rank)
    V_r = V.reshape(M, P, rank)
    W_r = W.reshape(P, N, rank)
    
    # For each j: compute B_j[k,i] = sum_r V[j,k,r] * W[k,i,r]
    # Then for each i: check sum_k B_j[k,i] * U[i,j,k]... no.
    # Actually: T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r
    # = U[i,j,:] dot (V[j,k,:] * W[k,i,:])  -- element-wise product then dot
    
    # Alternative: for each j, form M_j[k,i] = sum_r V[j,k,r]*W[k,i,r]
    # Then T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # This isn't directly M_j[k,i]*U[i,j,k] because U is also summed over r.
    
    # Let's just do it directly with einsum - it's small enough
    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=1e-9)

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W

def _random_init(rank, rng):
    """Initialize random factor matrices."""
    U = rng.standard_normal((N * M, rank))
    V = rng.standard_normal((M * P, rank))
    W = rng.standard_normal((P * N, rank))
    # Normalize columns to avoid numerical issues
    for r in range(rank):
        norm = np.sqrt(np.sum(U[:,r]**2) + np.sum(V[:,r]**2) + np.sum(W[:,r]**2))
        if norm > 1e-10:
            U[:,r] /= norm
            V[:,r] /= norm
            W[:,r] /= norm
    return U, V, W

def _gradient_step(U, V, W, lr, rng):
    """One gradient descent step to minimize ||T - U*V*W||^2."""
    U_r = U.reshape(N, M, -1)
    V_r = V.reshape(M, P, -1)
    W_r = W.reshape(P, N, -1)
    rank = U.shape[1]
    
    # Compute T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Gradient w.r.t. U[i,j,r] = sum over k: (T[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
    # grad_U[i,j,r] = einsum('ijk,jkr,kir->ijr', (T-1), V_r, W_r)
    error = T - 1.0
    grad_U = np.einsum('ijk,jkr,kir->ijr', error, V_r, W_r)
    grad_V = np.einsum('ijk,ijr,kir->jkr', error, U_r, W_r)
    grad_W = np.einsum('ijk,ijr,jkr->kir', error, U_r, V_r)
    
    U -= lr * grad_U.reshape(N*M, rank)
    V -= lr * grad_V.reshape(M*P, rank)
    W -= lr * grad_W.reshape(P*N, rank)
    
    return U, V

def _try_rank(rank, rng, max_iters=500):
    """Try to find a decomposition of given rank using gradient descent."""
    if _time_left() < 5.0:
        return None
    
    U, V, W = _random_init(rank, rng)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if _time_left() < 10.0:
            break
        
        U, V, W = _gradient_step(U, V, W, lr, rng)
        
        # Compute loss
        U_r = U.reshape(N, M, rank)
        V_r = V.reshape(M, P, rank)
        W_r = W.reshape(P, N, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        loss = np.sum((T - 1.0)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            lr *= 0.95
        
        # Check for exact solution
        if best_loss < 1e-12:
            # Verify with higher precision
            if _check_decomposition(best_U, best_V, best_W, rank):
                return best_U, best_V, best_W
    
    # Final check
    if best_loss < 1e-9:
        if _check_decomposition(best_U, best_V, best_W, rank):
            return best_U, best_V, best_W
    
    return None

def _try_structural(rank, rng):
    """Try structural approaches: exploit symmetries in 3x4*4x4."""
    if _time_left() < 5.0:
        return None
    
    # For 3x4 * 4x4, the tensor has some structure.
    # Try: use the fact that n=3 is small.
    # A decomposition of rank r means we have r triples (u_r, v_r, w_r)
    # such that sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    
    # Try random search with structured initialization
    for attempt in range(20):
        if _time_left() < 5.0:
            break
        
        U = rng.standard_normal((N * M, rank))
        V = rng.standard_normal((M * P, rank))
        W = rng.standard_normal((P * N, rank))
        
        # Normalize
        for r in range(rank):
            n_u = np.linalg.norm(U[:,r])
            n_v = np.linalg.norm(V[:,r])
            n_w = np.linalg.norm(W[:,r])
            if n_u > 1e-10 and n_v > 1e-10 and n_w > 1e-10:
                scale = (n_u * n_v * n_w) ** (1/3)
                U[:,r] /= n_u
                V[:,r] /= n_v
                W[:,r] /= n_w
        
        # Gradient descent
        lr = 0.005
        best_loss = float('inf')
        for it in range(300):
            if _time_left() < 5.0:
                break
            
            U_r = U.reshape(N, M, rank)
            V_r = V.reshape(M, P, rank)
            W_r = W.reshape(P, N, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            error = T - 1.0
            
            grad_U = np.einsum('ijk,jkr,kir->ijr', error, V_r, W_r)
            grad_V = np.einsum('ijk,ijr,kir->jkr', error, U_r, W_r)
            grad_W = np.einsum('ijk,ijr,jkr->kir', error, U_r, V_r)
            
            U -= lr * grad_U.reshape(N*M, rank)
            V -= lr * grad_V.reshape(M*P, rank)
            W -= lr * grad_W.reshape(P*N, rank)
            
            T = np.einsum('ijr,jkr,kir->ijk', U.reshape(N,M,rank), V.reshape(M,P,rank), W.reshape(P,N,rank))
            loss = np.sum((T - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
            
            if loss < 1e-12:
                break
            
            if it % 100 == 0:
                lr *= 0.9
        
        if best_loss < 1e-9:
            if _check_decomposition(U, V, W, rank):
                return U, V, W
    
    return None

def construct():
    """Find rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication."""
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Start from schoolbook rank = 48
    school_U, school_V, school_W = _schoolbook()
    best_rank = N * M * P
    best_result = (school_U, school_V, school_W)
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Upper bound: n*m*p = 48
    min_rank = max(N*M, M*P, P*N)  # 16
    
    # Try each rank
    for rank in range(min_rank, best_rank + 1):
        if _time_left() < 15.0:
            break
        
        # Try structural approach
        result = _try_structural(rank, rng)
        if result is not None:
            U, V, W = result
            if _check_decomposition(U, V, W, rank):
                best_rank = rank
                best_result = (U, V, W)
                break  # Found minimal, stop
        
        # Try gradient-based
        if _time_left() < 10.0:
            break
        result = _try_rank(rank, rng, max_iters=300)
        if result is not None:
            U, V, W = result
            if _check_decomposition(U, V, W, rank):
                best_rank = rank
                best_result = (U, V, W)
                break
    
    U, V, W = best_result
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
