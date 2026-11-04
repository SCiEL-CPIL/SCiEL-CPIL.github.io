"""Optimized tensor decomposition search for 2x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]
# W index convention: W[k*n + i, r]
# U index: i*m + j  (row of A: i in 0..n-1, col of A: j in 0..m-1)
# V index: j*p + k  (row of B: j in 0..m-1, col of B: k in 0..p-1)
# W index: k*n + i  (row of C: k in 0..p-1, col of C: i in 0..n-1)

def _build_target_tensor(n, m, p):
    """Return the 3-tensor T[i, j, k] = 1 for all valid indices, else 0.
    The tensor is defined such that the bilinear map is:
      C[i, k] = sum_{j} A[i, j] * B[j, k]
    In tensor decomposition form:
      C[i, k] = sum_r U[i*m + j? ...] 
    Actually, the standard matrix multiplication tensor T has T[i, j, k] = 1
    where the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    
    For the decomposition to work:
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
      The condition is: for all i, j, k:
        sum_r U[i*m + j, r] * V[j*p + k, r] = W[k*n + i, r]  ... wait, no.
        
    Let me re-derive. The tensor T[i, j, k] for matrix multiplication satisfies:
      C[i, k] = sum_j A[i, j] B[j, k]
      
    A rank-r decomposition of the matrix multiplication tensor means:
      T[i, j, k] = sum_{r=1}^{rank} U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
      
    Wait, that's not right either. Let me think about this more carefully.
    
    The matrix multiplication tensor T is a 3-tensor with T[i, j, k] = 1 for all
    valid (i, j, k). A rank-r decomposition means:
      T[i, j, k] = sum_{r=1}^{rank} U[i, j, r] * V[j, k, r] * W[k, i, r]
      
    But in the flattened form given:
      U has shape (n*m, rank): U[i*m + j, r]
      V has shape (m*p, rank): V[j*p + k, r]  
      W has shape (p*n, rank): W[k*n + i, r]
      
    And the condition is:
      For all i, j, k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
      where T[i,j,k] = 1.
    """
    T = np.zeros((n, m, p))
    T[:] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid rank decomposition of the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # For each (i, j, k), check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Efficient check: build the tensor and compare
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is a contraction over r
    # We can do: for each r, outer product U3[:,:,r] * V3[:,:,r] * W3[:,:,r] but that's expensive
    # Better: use einsum
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Note W3 is indexed by (k, i, r), so W3[k,i,r]
    
    T_computed = np.einsum('ijr,jkr,ki r->ijk', U3, V3, W3)
    
    T_target = np.ones((n, m, p))
    return np.allclose(T_computed, T_target, atol=tol, rtol=0)

def _random_init(n, m, p, rank, rng):
    """Random initialization for U, V, W."""
    U = rng.normal(0, 1, (n * m, rank))
    V = rng.normal(0, 1, (m * p, rank))
    W = rng.normal(0, 1, (p * n, rank))
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor and its norm."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p))
    residual = T_computed - T_target
    return residual, np.linalg.norm(residual)

def _gradient_descent(U, V, W, n, m, p, lr=0.01, max_iter=5000, tol=1e-10, deadline=None):
    """Perform gradient descent to minimize ||T_computed - T_target||^2."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T_target = np.ones((n, m, p))
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute residual
        T_computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = T_computed - T_target
        loss = np.sum(residual**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Gradients
        # dL/dU3[i,j,r] = 2 * sum_{k} residual[i,j,k] * V3[j,k,r] * W3[k,i,r]
        # dL/dV3[j,k,r] = 2 * sum_{i} residual[i,j,k] * U3[i,j,r] * W3[k,i,r]
        # dL/dW3[k,i,r] = 2 * sum_{j} residual[i,j,k] * U3[i,j,r] * V3[j,k,r]
        
        dU3 = 2 * np.einsum('ijk,jkr,kir->ijr', residual, V3, W3)
        dV3 = 2 * np.einsum('ijk,ijr,kir->jkr', residual, U3, W3)
        dW3 = 2 * np.einsum('ijk,ijr,jkr->kir', residual, U3, V3)
        
        # Update
        U3 -= lr * dU3
        V3 -= lr * dV3
        W3 -= lr * dW3
        
        # Reshape back
        U = U3.reshape(n * m, rank)
        V = V3.reshape(m * p, rank)
        W = W3.reshape(p * n, rank)
    
    return best_U, best_V, best_W, best_loss

def _try_rank(n, m, p, rank, rng, deadline, max_restarts=3, gd_iters=2000):
    """Try to find a valid decomposition of given rank."""
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        U, V, W = _random_init(n, m, p, rank, rng)
        
        # Gradient descent
        U, V, W, loss = _gradient_descent(U, V, W, n, m, p, lr=0.01, max_iter=gd_iters, tol=1e-12, deadline=deadline)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U, V, W
        
        if best_loss < 1e-12:
            # Verify
            if _check_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-10):
                return best_U, best_V, best_W, best_loss
    
    if best_U is not None and _check_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-10):
        return best_U, best_V, best_W, best_loss
    
    return None, None, None, best_loss

def _to_integer_or_half(U, V, W, tol=1e-6):
    """Try to snap values to integers or half-integers."""
    def snap(arr):
        arr = np.array(arr, dtype=float)
        # Try snapping to nearest 0.5
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        # Try snapping to nearest integer
        snapped_int = np.round(arr)
        if np.allclose(arr, snapped_int, atol=tol):
            return snapped_int
        return arr
    
    U_snapped = snap(U)
    V_snapped = snap(V)
    W_snapped = snap(W)
    return U_snapped, V_snapped, W_snapped

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    
    # Known lower bound for matrix multiplication rank
    # For 2x4 * 4x5, the rank is at least max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But actually the rank of matmul tensor for (n,m,p) is at least max(n*m, m*p, p*n)
    # The schoolbook rank is n*m*p = 40
    # Best known for small cases may be less
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But we need to be careful - the actual minimum rank might be higher
    
    # First, try the schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Start from a reasonable lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        U, V, W, loss = _try_rank(n, m, p, rank, rng, deadline, max_restarts=5, gd_iters=3000)
        
        if U is not None and loss < 1e-10:
            # Try to snap to integer/half-integer
            U_s, V_s, W_s = _to_integer_or_half(U, V, W)
            if _check_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-8):
                best_rank = rank
                best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            else:
                # Use the non-snapped version if it's valid
                if _check_decomposition(U, V, W, n, m, p, tol=1e-8):
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # If we found a valid decomposition, try one more rank lower
        if rank == lower_bound and U is not None:
            continue
    
    return best_result
