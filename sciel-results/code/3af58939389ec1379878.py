"""Tensor decomposition search for 3x3x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _compute_residual(U, V, W, T_target):
    """Compute the residual tensor T - T_target."""
    n, m, p = T_target.shape[0] // m, m, T_target.shape[2] // n
    T = np.zeros_like(T_target)
    for c in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    return T - T_target

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_target = _build_target_tensor(n, m, p)
    residual = _compute_residual(U, V, W, T_target)
    return np.max(np.abs(residual)) < tol

def _try_rank_alternating_least_squares(r, n, m, p, deadline, max_iter=500, n_restarts=5):
    """Try to find a rank-r decomposition using alternating least squares."""
    T_target = _build_target_tensor(n, m, p)
    best_residual = np.inf
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute current tensor
            T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for c in range(r):
                T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            residual = T - T_target
            
            # Alternate: fix V, W and solve for U
            # For each column c, we want to minimize ||residual + outer(U[:,c], V[:,c], W[:,c])||^2
            # This is a linear least squares problem
            
            # Update U
            for c in range(r):
                # The contribution of column c to T is outer(U[:,c], V[:,c], W[:,c])
                # We want: sum_{c'} U[:,c'] outer(V[:,c'], W[:,c']) ≈ T_target
                # Fixing all except c: U[:,c] should satisfy
                # outer(U[:,c], V[:,c], W[:,c]) ≈ T_target - sum_{c'≠c} outer(U[:,c'], V[:,c'], W[:,c'])
                
                # Compute the target for column c
                T_other = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for c2 in range(r):
                    if c2 != c:
                        T_other += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                
                T_c_target = T_target - T_other
                
                # We need to find U[:,c] such that outer(U[:,c], V[:,c], W[:,c]) ≈ T_c_target
                # This means: for all u,v,w: U[u,c] * V[v,c] * W[w,c] ≈ T_c_target[u,v,w]
                # If V[:,c] and W[:,c] are known, we can solve for U[:,c]
                
                # Vectorize: T_c_target.reshape(n*m, m*p*p*n) - but this is tricky
                # Instead, note that T_c_target[u,v,w] = U[u,c] * (V[v,c] * W[w,c])
                # Let B[v,w] = V[v,c] * W[w,c]
                # Then T_c_target[u,v,w] = U[u,c] * B[v,w]
                # This is a rank-1 approximation of T_c_target (as a 3D tensor)
                
                # We can solve this by flattening: for fixed u, T_c_target[u,:,:] = U[u,c] * B
                # So U[u,c] = <T_c_target[u,:,:], B> / <B, B>
                
                B = np.outer(V[:, c], W[:, c])  # shape (m*p, p*n)
                B_norm_sq = np.sum(B * B)
                if B_norm_sq < 1e-12:
                    continue
                
                # For each u, compute U[u,c]
                for u in range(n*m):
                    # T_c_target[u, :, :] should be approximately U[u,c] * B
                    U[u, c] = np.sum(T_c_target[u, :, :] * B) / B_norm_sq
            
            # Update V similarly
            for c in range(r):
                T_other = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for c2 in range(r):
                    if c2 != c:
                        T_other += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                
                T_c_target = T_target - T_other
                
                A = np.outer(U[:, c], W[:, c])  # shape (n*m, p*n)
                A_norm_sq = np.sum(A * A)
                if A_norm_sq < 1e-12:
                    continue
                
                for v in range(m*p):
                    V[v, c] = np.sum(T_c_target[:, v, :] * A) / A_norm_sq
            
            # Update W similarly
            for c in range(r):
                T_other = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for c2 in range(r):
                    if c2 != c:
                        T_other += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
                
                T_c_target = T_target - T_other
                
                C_mat = np.outer(U[:, c], V[:, c])  # shape (n*m, m*p)
                C_norm_sq = np.sum(C_mat * C_mat)
                if C_norm_sq < 1e-12:
                    continue
                
                for w in range(p*n):
                    W[w, c] = np.sum(T_c_target[:, :, w] * C_mat) / C_norm_sq
            
            # Check convergence
            T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for c in range(r):
                T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            residual = T - T_target
            max_res = np.max(np.abs(residual))
            
            if max_res < best_residual:
                best_residual = max_res
            
            if max_res < 1e-6:
                return U, V, W, max_res
    
    return None

def _try_rank_gradient_descent(r, n, m, p, deadline, max_iter=2000, lr=0.01):
    """Try to find a rank-r decomposition using gradient descent."""
    T_target = _build_target_tensor(n, m, p)
    
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    current_lr = lr
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Compute T
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(r):
            T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
        residual = T - T_target
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(r):
            # dU[:,c] = sum_{v,w} residual[u,v,w] * V[v,c] * W[w,c]
            VW = np.outer(V[:, c], W[:, c])
            dU[:, c] = np.einsum('ijk,jk->i', residual, VW)
            
            # dV[:,c] = sum_{u,w} residual[u,v,w] * U[u,c] * W[w,c]
            UW = np.outer(U[:, c], W[:, c])
            dV[:, c] = np.einsum('ijk,ik->j', residual, UW)
            
            # dW[:,c] = sum_{u,v} residual[u,v,w] * U[u,c] * V[v,c]
            UV = np.outer(U[:, c], V[:, c])
            dW[:, c] = np.einsum('ijk,ij->k', residual, UV)
        
        # Update
        U -= current_lr * dU
        V -= current_lr * dV
        W -= current_lr * dW
        
        # Decay learning rate
        if it % 100 == 0:
            current_lr *= 0.9
        
        # Check convergence
        if np.max(np.abs(residual)) < 1e-6:
            return U, V, W, np.max(np.abs(residual))
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = (U_school, V_school, W_school, n*m*p)
    
    # Lower bound for matrix multiplication rank
    # For C = A * B where A is n×m and B is m×p, the rank is at least max(n*m, m*p, p*n)
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound up to schoolbook rank
    for r in range(lower_bound, n*m*p):
        if time.time() > deadline - 10:
            break
        
        # Try alternating least squares first (more stable)
        result = _try_rank_alternating_least_squares(r, n, m, p, deadline, max_iter=300, n_restarts=3)
        if result is not None:
            U, V, W, max_res = result
            if _verify(U, V, W, n, m, p):
                best_result = (U, V, W, r)
                break
        
        # If ALS didn't work, try gradient descent
        if time.time() > deadline - 10:
            break
        result = _try_rank_gradient_descent(r, n, m, p, deadline, max_iter=1000, lr=0.005)
        if result is not None:
            U, V, W, max_res = result
            if _verify(U, V, W, n, m, p):
                best_result = (U, V, W, r)
                break
    
    U, V, W, rank = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
