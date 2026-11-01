import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T[i, j, k] for matrix multiplication: T[i,j,k] = 1 for all valid indices
    # where i in [0,n), j in [0,m), k in [0,p)
    # Rank-1 decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not standard. Let me use the given interface.
    
    # Given interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W[k*n + i, r] - so W is indexed by (k, i) where k is column of B/C, i is row of A/C
    # This suggests: C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But that would be: C[i,k] = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c]
    # Which is not the standard matrix product.
    
    # Let me reconsider. Perhaps the decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta for the tensor
    # The tensor for matrix mult: for each (i,j,k), the "value" is such that
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # I think the correct interpretation is that we're decomposing the identity-like tensor
    # where the product is: sum_{i,j,k} A[i,j] B[j,k] W[k*n+i, c] ... 
    
    # Let me just try to minimize: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This would make the "tensor" all ones, which corresponds to:
    # sum_{i,j,k} A[i,j] B[j,k] * 1 = (sum_i sum_j A[i,j]) * (sum_j sum_k B[j,k]) which is wrong.
    
    # I think I need to be more careful. Let me use a different approach:
    # The matrix multiplication tensor M has M[i,j,k] = 1 if we interpret it as
    # the coefficient such that <M, A ⊗ B ⊗ C> = sum_{i,k} C[i,k] (A*B)[i,k]
    # = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # So M[i,j,k] = 1 for all i,j,k.
    
    # A rank-1 decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, but U has n*m entries indexed by (i,j), V has m*p by (j,k), W has p*n by (k,i).
    # So the product U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over c should equal M[i,j,k] = 1.
    
    # But that means for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 54 equations. We want to find U, V, W with minimal rank r.
    
    def tensor_error(params, rank):
        """Compute the squared error of the tensor reconstruction."""
        U = params[:rank * n * m].reshape(rank, n * m).T  # (n*m, rank)
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m * p).T  # (m*p, rank)
        W = params[rank * n * m + rank * m * p :].reshape(rank, p * n).T  # (p*n, rank)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (val - 1.0) ** 2
        return error
    
    def tensor_error_grad(params, rank):
        """Compute gradient of tensor error."""
        U = params[:rank * n * m].reshape(rank, n * m).T  # (n*m, rank)
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m * p).T  # (m*p, rank)
        W = params[rank * n * m + rank * m * p :].reshape(rank, p * n).T  # (p*n, rank)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    diff = 2.0 * (val - 1.0)
                    for c in range(rank):
                        grad_U[i * m + j, c] += diff * V[j * p + k, c] * W[k * n + i, c]
                        grad_V[j * p + k, c] += diff * U[i * m + j, c] * W[k * n + i, c]
                        grad_W[k * n + i, c] += diff * U[i * m + j, c] * V[j * p + k, c]
        
        grad = np.concatenate([
            grad_U.flatten(),
            grad_V.flatten(),
            grad_W.flatten()
        ])
        return grad
    
    # Try different ranks, starting from a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    best_result = None
    best_rank = n * m * p  # fallback to schoolbook
    
    for rank in range(18, 55):
        # Try a few random initializations
        found = False
        for attempt in range(5):
            np.random.seed(42 + attempt * 100 + rank)
            # Sparse random initialization
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = minimize(
                tensor_error, params0, args=(rank,),
                jac=tensor_error_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success and result.fun < 1e-8:
                # Found a valid decomposition
                U_sol = result.x[:rank * n * m].reshape(n * m, rank)
                V_sol = result.x[rank * n * m : rank * n * m + rank * m * p].reshape(m * p, rank)
                W_sol = result.x[rank * n * m + rank * m * p :].reshape(p * n, rank)
                
                # Verify
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_sol[i*m+j, c] * V_sol[j*p+k, c] * W_sol[k*n+i, c] for c in range(rank))
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_rank = rank
                    best_result = (U_sol, V_sol, W_sol)
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        U_sol, V_sol, W_sol = best_result
        # Convert to lists with integer/half-integer entries if possible
        U_list = [[float(x) for x in row] for row in U_sol]
        V_list = [[float(x) for x in row] for row in V_sol]
        W_list = [[float(x) for x in row] for row in W_sol]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
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
