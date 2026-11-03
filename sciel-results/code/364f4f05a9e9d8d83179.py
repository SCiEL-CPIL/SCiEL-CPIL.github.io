"""Find rank-minimal exact tensor decomposition for 2x4 x 4x6 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor)
    # We want: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Actually, the multiplication tensor is: T[i,j,k] = 1 (it's the identity in the sense that
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k], so the tensor has T[i,j,k]=1)
    
    # Let's verify: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C = T * (A ⊗ B) where T[i,j,k] = 1
    # So T is the all-ones tensor of shape (n, m, p)
    
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # But also need at least max(n, m, p) = 6
    
    # Let's try to find a decomposition using iterative methods
    
    def try_rank(r):
        """Try to find a decomposition of rank r using random restarts and ALS."""
        if time.time() > deadline:
            return None
        
        # Number of equations: n*m*p = 48
        # Number of unknowns: 3*r (for U, V, W) minus scaling ambiguities
        # For r=24: 72 unknowns, 48 equations - overdetermined but might work
        
        best_result = None
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Scale to avoid numerical issues
            U /= np.sqrt(np.sum(U**2))
            V /= np.sqrt(np.sum(V**2))
            W /= np.sqrt(np.sum(W**2))
            
            # Alternating least squares
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each column u, solve min sum_{i,j,k} (U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                # This is nonlinear, so use gradient descent or linearize
                
                # Actually, for fixed V and W, the problem for U is:
                # min_U sum_{i,j,k} (U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]))^2 ... 
                # No, it's sum_r, so for each r independently if we separate...
                # But the sum is over r, so it's coupled.
                
                # Let's use a different approach: treat it as a system of equations
                # and use nonlinear least squares
                
                # Actually, let's use a simpler iterative method
                # For each r, the contribution is U_r ⊗ V_r ⊗ W_r
                # We want sum_r U_r ⊗ V_r ⊗ W_r = T (all-ones tensor)
                
                # Use gradient descent on the full problem
                # Loss = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                
                # Compute current tensor
                current = np.zeros((n, m, p))
                for r_idx in range(r):
                    # U[:, r_idx] is shape (n*m,), reshape to (n, m)
                    Ur = U[:, r_idx].reshape(n, m)
                    Vr = V[:, r_idx].reshape(m, p)
                    Wr = W[:, r_idx].reshape(p, n)
                    # Contribution: Ur[i,j] * Vr[j,k] * Wr[k,i]
                    # This is an outer product in a sense
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                current[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
                
                # Gradient
                loss = np.sum((current - 1.0)**2)
                
                if loss < 1e-10:
                    # Found a solution!
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Compute gradients (this is expensive, so do it carefully)
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r_idx in range(r):
                    Ur = U[:, r_idx].reshape(n, m)
                    Vr = V[:, r_idx].reshape(m, p)
                    Wr = W[:, r_idx].reshape(p, n)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual = current[i,j,k] - 1.0
                                # d/dU[i*m+j, r_idx] of sum = residual * Vr[j,k] * Wr[k,i]
                                dU[i*m+j, r_idx] += 2 * residual * Vr[j,k] * Wr[k,i]
                                dV[j*p+k, r_idx] += 2 * residual * Ur[i,j] * Wr[k,i]
                                dW[k*n+i, r_idx] += 2 * residual * Ur[i,j] * Vr[j,k]
                
                # Gradient descent with adaptive step size
                lr = 0.01
                # Scale learning rate based on gradient magnitude
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 0:
                    step = lr * 100 / grad_norm
                    U -= step * dU
                    V -= step * dV
                    W -= step * dW
            
            # Check final loss
            current = np.zeros((n, m, p))
            for r_idx in range(r):
                Ur = U[:, r_idx].reshape(n, m)
                Vr = V[:, r_idx].reshape(m, p)
                Wr = W[:, r_idx].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            current[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
            
            loss = np.sum((current - 1.0)**2)
            if loss < 1e-8 and best_result is None:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return best_result
    
    # Try ranks from lower bound to schoolbook
    # Lower bound for 2x4x6: at least max(2*4, 4*6, 6*2) = 24
    # But we can try smaller ranks too
    
    for r in range(24, 49):
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback: schoolbook
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
