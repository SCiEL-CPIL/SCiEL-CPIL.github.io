import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # The tensor we need to decompose: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # 
    # Strategy: Start from schoolbook (rank 36) and try to find a lower-rank decomposition
    # by using numerical optimization. We'll try ranks 32, 33, 34, 35 and see which is achievable.
    
    # First, let's establish the target tensor values
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (12, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # The constraint is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try a structured approach:
    # 1. Start with schoolbook rank 36
    # 2. Try to reduce by identifying linear dependencies
    
    # Actually, let me try a more direct numerical approach:
    # Fix the structure and solve for the entries.
    
    # For rank r, we have 9*r + 12*r + 12*r = 33*r variables
    # And we have n*m*p = 36 constraints
    # So for r >= 36/33 ≈ 1.09, the system is underdetermined in principle
    # But the equations are cubic, so it's non-trivial.
    
    # Let me try a different approach: use the fact that we can scale U and V
    # and compensate in W. This gives us gauge freedom.
    
    # Actually, let me try to construct a rank-32 decomposition using a known technique:
    # The 3x3x4 matrix multiplication can be seen as 3x3x3 plus an extra column.
    # The rank of 3x3x3 is known to be 23 (Winograd).
    # For 3x3x4, we can potentially share some products.
    
    # Let me try a numerical optimization approach:
    # Start from a random initialization and use gradient descent to minimize
    # the sum of squared violations.
    
    def make_violations(U_flat, V_flat, W_flat, rank):
        """Compute the violations for a given decomposition."""
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        violations = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    violations[idx] = prod.sum() - 1.0
                    idx += 1
        return violations
    
    def objective(params, rank):
        """Objective function: sum of squared violations."""
        total_size = 33 * rank
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        violations = make_violations(U_flat, V_flat, W_flat, rank)
        return np.sum(violations**2)
    
    def gradient(params, rank):
        """Gradient of the objective function."""
        total_size = 33 * rank
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank:n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank:]
        
        U = U_flat.reshape(n*m, rank)
        V = V_flat.reshape(m*p, rank)
        W = W_flat.reshape(p*n, rank)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    s = prod.sum() - 1.0
                    # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r] * 2*s
                    grad_U[i*m+j, :] += 2 * s * V[j*p+k, :] * W[k*n+i, :]
                    grad_V[j*p+k, :] += 2 * s * U[i*m+j, :] * W[k*n+i, :]
                    grad_W[k*n+i, :] += 2 * s * U[i*m+j, :] * V[j*p+k, :]
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    # Try ranks from 32 up to 36
    best_rank = 36
    best_result = None
    
    # Start with schoolbook as a valid solution
    rank = 36
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
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = 36
    
    # Check time
    if time.time() > deadline - 30:
        return best_result
    
    # Try to find lower-rank decompositions
    for target_rank in [32, 33, 34, 35]:
        if time.time() > deadline - 20:
            break
        
        found = False
        max_restarts = 5 if target_rank >= 34 else 3
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + restart + target_rank)
            
            # Initialize with small random values
            params = np.random.randn(33 * target_rank) * 0.1
            
            # Use L-BFGS-B for optimization
            try:
                result = minimize(
                    objective,
                    params,
                    args=(target_rank,),
                    method='L-BFGS-B',
                    jac=gradient,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < 1e-10:
                    # Success! Extract the solution
                    U_flat = result.x[:n*m*target_rank]
                    V_flat = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = result.x[n*m*target_rank + m*p*target_rank:]
                    
                    U = U_flat.reshape(n*m, target_rank)
                    V = V_flat.reshape(m*p, target_rank)
                    W = W_flat.reshape(p*n, target_rank)
                    
                    # Verify
                    violations = make_violations(U_flat, V_flat, W_flat, target_rank)
                    if np.max(np.abs(violations)) < 1e-8:
                        best_rank = target_rank
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    # Final check
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
