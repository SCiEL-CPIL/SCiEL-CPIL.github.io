"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n+i, r] convention
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    def verify_solution(U, V, W, rank):
        """Verify that the decomposition exactly produces the target tensor."""
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return np.allclose(T, np.ones((n,m,p)), atol=1e-10)
    
    def try_rank(rank, num_restarts=10, max_iter=1000):
        """Try to find a rank-`rank` decomposition using gradient-free optimization."""
        for restart in range(num_restarts):
            np.random.seed(1000 + rank * 100 + restart)
            
            # Initialize with small random values
            U = np.random.uniform(-1, 1, (n*m, rank))
            V = np.random.uniform(-1, 1, (m*p, rank))
            W = np.random.uniform(-1, 1, (p*n, rank))
            
            # Scale to help with convergence
            scale = 1.0 / np.sqrt(rank)
            U = U * scale
            V = V * scale
            W = W * scale
            
            # Define the objective: Frobenius norm of tensor decomposition error
            # Vectorized computation
            def objective(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Compute the tensor from decomposition
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T = np.zeros((n, m, p))
                for r in range(rank):
                    # For each r, compute the outer product contribution
                    # U_col = U[:, r], V_col = V[:, r], W_col = W[:, r]
                    U_col = U_flat[:, r]  # shape (n*m,)
                    V_col = V_flat[:, r]  # shape (m*p,)
                    W_col = W_flat[:, r]  # shape (p*n,)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i,j,k] += U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
                
                target = np.ones((n, m, p))
                error = np.sum((T - target) ** 2)
                return error
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(objective, params0, method='Nelder-Mead', 
                                  options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
                final_error = result.fun
                
                if final_error < 1e-8:
                    U_final = result.x[:n*m*rank].reshape(n*m, rank)
                    V_final = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_final = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_final * 2) / 2
                    V_rounded = np.round(V_final * 2) / 2
                    W_rounded = np.round(W_final * 2) / 2
                    
                    if verify_solution(U_rounded, V_rounded, W_rounded, rank):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    
                    # Try rounding to integers
                    U_int = np.round(U_final)
                    V_int = np.round(V_final)
                    W_int = np.round(W_final)
                    if verify_solution(U_int, V_int, W_int, rank):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                    
                    # If continuous solution is very good, return it (might not be exact integer but close)
                    # The task requires "integer/half-integer entries that exactly decompose"
                    # So we must return the rounded one if it verifies.
                    # If it doesn't verify, we continue searching.
            except Exception:
                pass
        
        return None
    
    # Try ranks from 24 to 35
    # 3x3x3 is 23. 3x3x5 is likely around 25-28.
    for rank in range(24, 36):
        result = try_rank(rank, num_restarts=5)
        if result:
            return result
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
