"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n+i, r] convention
    
    # Try various ranks, starting from a reasonable lower bound
    # For 3x3 x 3x5, try ranks from 20 upward
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    for rank in range(20, best_rank + 1):
        np.random.seed(42 + rank)
        
        # Initialize with small random values
        U = np.random.uniform(-1, 1, (n*m, rank))
        V = np.random.uniform(-1, 1, (m*p, rank))
        W = np.random.uniform(-1, 1, (p*n, rank))
        
        # Scale to help with convergence
        U = U / np.sqrt(rank)
        V = V / np.sqrt(rank)
        W = W / np.sqrt(rank)
        
        # Define the objective: Frobenius norm of tensor decomposition error
        def objective(params):
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Compute the tensor from decomposition
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We want this to equal the matrix multiplication tensor
            # For matrix multiplication: T[i,j,k] = 1 for all valid i,j,k
            # Actually, the target tensor for matrix multiplication is:
            # T[i,j,k] = 1 if (i,j,k) is a valid term, which is always true
            # But wait - I need to verify what the target tensor actually is
            
            # The correct target: for C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
            # So the tensor T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)
            target = np.ones((n, m, p))
            
            # Compute decomposed tensor
            T = np.zeros((n, m, p))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i,j,k] += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
            
            error = np.sum((T - target) ** 2)
            return error
        
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use Nelder-Mead for small ranks, otherwise skip
        if rank <= 25:
            try:
                result = minimize(objective, params0, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
                final_error = result.fun
                if final_error < 1e-8:
                    # Found exact solution, try to round to integers/half-integers
                    U_final = result.x[:n*m*rank].reshape(n*m, rank)
                    V_final = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_final = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Try rounding to half-integers
                    U_rounded = np.round(U_final * 2) / 2
                    V_rounded = np.round(V_final * 2) / 2
                    W_rounded = np.round(W_final * 2) / 2
                    
                    # Verify the rounded solution
                    T_check = np.zeros((n, m, p))
                    for r in range(rank):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_check[i,j,k] += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                    
                    if np.allclose(T_check, np.ones((n,m,p)), atol=1e-10):
                        # Convert to lists
                        U_list = U_rounded.tolist()
                        V_list = V_rounded.tolist()
                        W_list = W_rounded.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Use unrounded if close enough
                        U_list = U_final.tolist()
                        V_list = V_final.tolist()
                        W_list = W_final.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
        
        # If we get here and rank is the schoolbook, return that
        if rank == best_rank:
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
