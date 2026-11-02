import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Target rank. 
    # Schoolbook is n*m*p = 54.
    # Known bound for 3x3x3 is 23. Direct sum for 3x3x6 is 46.
    # We will try to find a decomposition of rank 46.
    
    def get_tensor_error(params, rank):
        """
        Vectorized error calculation.
        params: flat array of U, V, W
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        
        We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
        """
        # Reshape
        U = params[:rank * n * m].reshape(n * m, rank)
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape(m * p, rank)
        W = params[rank * n * m + rank * m * p :].reshape(p * n, rank)
        
        # We need to compute T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is a bit tricky to vectorize directly with standard einsum because the indices are permuted.
        # Let's create index arrays.
        
        # i: 0..n-1, j: 0..m-1, k: 0..p-1
        # idx_u = i*m + j
        # idx_v = j*p + k
        # idx_w = k*n + i
        
        # We can loop over c (rank) and add the outer product of the vectors?
        # T = sum_c U[:, c] (indexed by i,j) * V[:, c] (indexed by j,k) * W[:, c] (indexed by k,i)
        # This is not a standard outer product because the indices link up.
        
        # Alternative: Iterate over c, and for each c, compute the contribution to the (n, m, p) tensor.
        # Contribution_c[i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        T = np.zeros((n, m, p))
        
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # We need T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            # This is a cyclic contraction.
            # Can be done with einsum: 'ij,jk,ki->ijk'
            # But w_c is (p, n) which is (k, i).
            # So yes, 'ij,jk,ki->ijk'
            T += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            
        # Error is sum of (T - 1)^2
        err = T - 1.0
        return np.sum(err * err)

    def get_tensor_grad(params, rank):
        """
        Gradient of the squared error.
        """
        U = params[:rank * n * m].reshape(n * m, rank)
        V = params[rank * n * m : rank * n * m + rank * m * p].reshape(m * p, rank)
        W = params[rank * n * m + rank * m * p :].reshape(p * n, rank)
        
        T = np.zeros((n, m, p))
        
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
            
        diff = 2.0 * (T - 1.0) # Gradient of sum (T-1)^2 w.r.t T is 2(T-1)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # d(T)/d(U[i,j,c]) = V[j*p+k, c] * W[k*n+i, c] summed over k?
            # No, T[i,j,k] = U[i,j,c] V[j,k,c] W[k,i,c] + ...
            # d(T[i,j,k])/d(U[i,j,c]) = V[j,k,c] W[k,i,c]
            # grad_U[i,j,c] = sum_k diff[i,j,k] * V[j,k,c] W[k,i,c]
            
            # Vectorized:
            # grad_U_c[i,j] = sum_k diff[i,j,k] * v_c[j,k] * w_c[k,i]
            # This is a contraction of diff (i,j,k), v (j,k), w (k,i) to (i,j).
            # 'ijk,jk,ki->ij'
            grad_U_c = np.einsum('ijk,jk,ki->ij', diff, v_c, w_c)
            grad_U[:, c] = grad_U_c.flatten()
            
            # grad_V[j,k,c] = sum_i diff[i,j,k] * U[i,j,c] * W[k,i,c]
            # 'ijk,ij,ki->jk'
            grad_V_c = np.einsum('ijk,ij,ki->jk', diff, u_c, w_c)
            grad_V[:, c] = grad_V_c.flatten()
            
            # grad_W[k,i,c] = sum_j diff[i,j,k] * U[i,j,c] * V[j,k,c]
            # 'ijk,ij,jk->ki'
            grad_W_c = np.einsum('ijk,ij,jk->ki', diff, u_c, v_c)
            grad_W[:, c] = grad_W_c.flatten()
            
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_result = None
    best_rank = n * m * p  # 54
    
    # Try rank 46 first (2 * 23)
    # Then maybe 44, 42...
    # But 46 is a known feasible bound.
    
    for rank in [46, 44, 42, 40]:
        found = False
        # Try a few random initializations
        for attempt in range(10):
            np.random.seed(42 + attempt * 100 + rank)
            # Initialize with small random values
            U_init = np.random.randn(n * m, rank) * 0.1
            V_init = np.random.randn(m * p, rank) * 0.1
            W_init = np.random.randn(p * n, rank) * 0.1
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = minimize(
                    get_tensor_error, params0, args=(rank,),
                    jac=get_tensor_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.fun < 1e-6:
                    # Verify
                    U_sol = result.x[:rank * n * m].reshape(n * m, rank)
                    V_sol = result.x[rank * n * m : rank * n * m + rank * m * p].reshape(m * p, rank)
                    W_sol = result.x[rank * n * m + rank * m * p :].reshape(p * n, rank)
                    
                    # Check max error
                    T_check = np.zeros((n, m, p))
                    for c in range(rank):
                        u_c = U_sol[:, c].reshape(n, m)
                        v_c = V_sol[:, c].reshape(m, p)
                        w_c = W_sol[:, c].reshape(p, n)
                        T_check += np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
                    
                    max_err = np.max(np.abs(T_check - 1.0))
                    if max_err < 1e-5:
                        best_rank = rank
                        best_result = (U_sol, V_sol, W_sol)
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
            
    if best_result is not None:
        U_sol, V_sol, W_sol = best_result
        U_list = [[float(x) for x in row] for row in U_sol]
        V_list = [[float(x) for x in row] for row in V_sol]
        W_list = [[float(x) for x in row] for row in W_sol]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook rank 54
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
