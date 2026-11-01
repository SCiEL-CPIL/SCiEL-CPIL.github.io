import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=1e-9)
    
    def try_rank(r, max_restarts=50, max_iters=200):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with structured pattern based on 2x3x3 sub-problem
            # For the 2x3x3 case, rank 5 is known to be optimal
            # We'll use this as a template for initialization
            
            # Create initial U, V, W with some structure
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Fill with random values but with some structure
            for k in range(r):
                # Each column gets a random pattern
                scale = 0.5 if k % 2 == 0 else 1.0
                U[:, k] = np.random.choice([-1, 0, 1], size=n*m) * scale
                V[:, k] = np.random.choice([-1, 0, 1], size=m*p) * scale
                W[:, k] = np.random.choice([-1, 0, 1], size=p*n) * scale
            
            # Iterative refinement
            for iter in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Solve for W given U, V using least squares
                # Build the linear system
                A = np.zeros((n*m * m*p * p*n, r * p*n))
                b = np.zeros(n*m * m*p * p*n)
                
                for a in range(n*m):
                    for b_idx in range(m*p):
                        for c in range(p*n):
                            eq_idx = (a * m*p + b_idx) * p*n + c
                            b[eq_idx] = target[a, b_idx, c]
                            
                            for k in range(r):
                                col_idx = c * r + k
                                A[eq_idx, col_idx] = U[a, k] * V[b_idx, k]
                
                # Solve for W
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p*n, r)
                
                # Calculate error
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                err = np.sum((T_approx - target)**2)
                
                if err < best_err:
                    best_err = err
                    best_params = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-9:
                        # Found a solution! Try to round to half-integers
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if check_decomposition(U_rounded, V_rounded, W_rounded):
                            return U_rounded, V_rounded, W_rounded
                        elif check_decomposition(U, V, W):
                            return U, V, W
                
                # Update U and V using gradient descent
                # Gradient of err w.r.t. U[a,k] = sum_{b,c} V[b,k] * W[c,k] * (T_approx[a,b,c] - target[a,b,c])
                # This is expensive, so we do a simpler update
                
                # Perturb U and V slightly
                noise = 0.1 / (1 + iter)
                U += np.random.randn(n*m, r) * noise
                V += np.random.randn(m*p, r) * noise
                
                # Occasionally try a larger step
                if iter % 10 == 0:
                    U += np.random.randn(n*m, r) * 0.5
                    V += np.random.randn(m*p, r) * 0.5
        
        return None
    
    # Try rank 20 first (known lower bound is 15, upper bound is 24)
    res = try_rank(20, max_restarts=30, max_iters=100)
    if res is not None:
        U, V, W = res
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 21
    if time.time() < deadline - 20:
        res = try_rank(21, max_restarts=20, max_iters=80)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24 (trivial construction)
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
