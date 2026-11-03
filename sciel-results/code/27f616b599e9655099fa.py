import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Target Tensor: T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Actually, the tensor T for matrix multiplication C = AB is defined such that:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1.
    # Wait, the task says "decompose the matrix multiplication tensor".
    # The tensor T is of shape (n, m, p) with T[i,j,k] = 1.
    # The decomposition is T = sum_c U[:,c] \otimes V[:,c] \otimes W[:,c].
    # U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Indexing convention:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # Construct the target tensor T as a 3D array (n, m, p)
    T = np.ones((n, m, p))
    
    # Reshape T to match the flattening in U, V, W?
    # U is flattened (i,j). V is flattened (j,k). W is flattened (k,i).
    # The product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k].
    
    # Let's define the target matrix for the optimization.
    # We want to find U, V, W such that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We can flatten the constraint into a vector of equations.
    # Number of equations: n * m * p = 18.
    # Number of variables: rank * (n*m + m*p + p*n) = rank * (6 + 9 + 6) = rank * 21.
    
    def compute_error(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    error += (val - 1.0)**2
        return error

    # Try to find rank 17
    rank_target = 17
    best_solution = None
    
    # Helper to check if a solution is valid and can be rounded
    def validate_and_round(U, V, W, rank):
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check exactness
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_r[i*m + j, c] * V_r[j*p + k, c] * W_r[k*n + i, c]
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-9:
            return U_r, V_r, W_r
        return None

    # Initialization strategy
    # Start with schoolbook rank 18, and try to project to rank 17?
    # Or just random restarts.
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        np.random.seed(42 + r)
        U = np.random.randn(n*m, rank_target)
        V = np.random.randn(m*p, rank_target)
        W = np.random.randn(p*n, rank_target)
        
        # Scale to help optimization
        U /= np.sqrt(rank_target)
        V /= np.sqrt(rank_target)
        W /= np.sqrt(rank_target)
        
        # Use scipy optimize
        try:
            import scipy.optimize as so
            # Flatten parameters
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(params):
                U_p = params[:n*m*rank_target].reshape(n*m, rank_target)
                V_p = params[n*m*rank_target:n*m*rank_target + m*p*rank_target].reshape(m*p, rank_target)
                W_p = params[n*m*rank_target + m*p*rank_target:].reshape(p*n, rank_target)
                return compute_error(U_p, V_p, W_p, rank_target)
                
            result = so.minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            
            if result.fun < 1e-6:
                U_opt = result.x[:n*m*rank_target].reshape(n*m, rank_target)
                V_opt = result.x[n*m*rank_target:n*m*rank_target + m*p*rank_target].reshape(m*p, rank_target)
                W_opt = result.x[n*m*rank_target + m*p*rank_target:].reshape(p*n, rank_target)
                
                sol = validate_and_round(U_opt, V_opt, W_opt, rank_target)
                if sol:
                    U_r, V_r, W_r = sol
                    # Convert to lists
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        except Exception:
            pass

    # Fallback to Rank 18 Schoolbook
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
