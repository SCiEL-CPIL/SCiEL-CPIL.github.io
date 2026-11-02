"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    schoolbook_rank = n * m * p  # 80
    
    # Build the target tensor T as a flat vector
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Indexing: U index = i*m + j, V index = j*p + k, W index = k*n + i
    T = np.ones(n * m * p)
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140
    
    # Try to find lower rank decompositions using structured search
    # The matrix multiplication tensor for A(n×m) * B(m×p) = C(n×p)
    # has entries T[i,j,k] where (i,j,k) corresponds to C[i,k] += A[i,j]*B[j,k]
    # Rank-1 term r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We'll use a smooth optimization approach:
    # Minimize ||sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] - T||^2
    # with the constraint that entries are "nice" (integers or half-integers)
    
    # For each target rank, use multiple random restarts with gradient descent
    def flatten_factors(U, V, W, rank):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_factors(x, n, m, p, rank):
        u_size = n * m * rank
        v_size = m * p * rank
        w_size = p * n * rank
        U = x[:u_size].reshape(n * m, rank)
        V = x[u_size:u_size + v_size].reshape(m * p, rank)
        W = x[u_size + v_size:].reshape(p * n, rank)
        return U, V, W
    
    def objective_fast(x, n, m, p, rank):
        u_size = n * m * rank
        v_size = m * p * rank
        U = x[:u_size].reshape(n * m, rank)
        V = x[u_size:u_size + v_size].reshape(m * p, rank)
        W = x[u_size + v_size:].reshape(p * n, rank)
        
        # Compute all entries efficiently
        error = 0.0
        for i in range(n):
            for j in range(m):
                ui = U[i * m + j]  # shape (rank,)
                for k in range(p):
                    vk = V[j * p + k]  # shape (rank,)
                    wi = W[k * n + i]  # shape (rank,)
                    val = np.sum(ui * vk * wi)
                    error += (val - 1.0) ** 2
        return error
    
    def als_step(U, V, W, n, m, p, rank, lr=0.1, momentum=0.9):
        """
        Perform one ALS iteration.
        Fix V and W, solve for U; then fix U and W, solve for V; then fix U and V, solve for W.
        Each subproblem is a linear least squares problem.
        """
        # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]) = 1 for all k
        # So: U[i*m+j, :] @ M = 1 where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
        
        # We can batch this: for fixed i,j, M is p x rank
        # We want U[i*m+j, :] such that U[i*m+j, :] @ M = ones(p)
        # This is a linear system: M^T @ U[i*m+j, :]^T = ones(p)
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build M: p x rank
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M^T @ u = ones(p)
                # Use least squares
                try:
                    u, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Solve for V: for each (j,k), V[j*p+k, :] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
        # So: V[j*p+k, :] @ (U[i*m+j, :] * W[k*n+i, :]) = 1 for all i
        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    v, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Solve for W: for each (k,i), W[k*n+i, :] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
        # So: W[k*n+i, :] @ (U[i*m+j, :] * V[j*p+k, :]) = 1 for all j
        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
        
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    w, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        return U_new, V_new, W_new
    
    # Try ranks from schoolbook-1 down
    for target_rank in range(schoolbook_rank - 1, max(15, schoolbook_rank // 2 - 5), -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        n_attempts = 5 if target_rank >= 70 else 3
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            
            # Initialize near schoolbook but with some structure
            # Try random initialization with small perturbations
            U_init = np.random.randn(n * m, target_rank) * 0.05
            V_init = np.random.randn(m * p, target_rank) * 0.05
            W_init = np.random.randn(p * n, target_rank) * 0.05
            
            # Also try initialization close to schoolbook (first target_rank columns)
            if target_rank <= schoolbook_rank:
                U_init2 = U_school[:, :target_rank].copy()
                V_init2 = V_school[:, :target_rank].copy()
                W_init2 = W_school[:, :target_rank].copy()
                # Add small noise to escape local minima
                U_init2 += np.random.randn(n * m, target_rank) * 0.01
                V_init2 += np.random.randn(m * p, target_rank) * 0.01
                W_init2 += np.random.randn(p * n, target_rank) * 0.01
                x_init2 = flatten_factors(U_init2, V_init2, W_init2, target_rank)
            else:
                x_init2 = None
            
            x_init = flatten_factors(U_init, V_init, W_init, target_rank)
            
            # Use ALS for refinement
            U_curr, V_curr, W_curr = U_init.copy(), V_init.copy(), W_init.copy()
            
            # Run ALS for a fixed number of iterations
            for iter in range(100):
                if time.time() > deadline - 25:
                    break
                U_curr, V_curr, W_curr = als_step(U_curr, V_curr, W_curr, n, m, p, target_rank)
                
                # Check convergence
                err = objective_fast(flatten_factors(U_curr, V_curr, W_curr, target_rank), n, m, p, target_rank)
                if err < 1e-8:
                    break
            
            # Check if we found a solution
            err = objective_fast(flatten_factors(U_curr, V_curr, W_curr, target_rank), n, m, p, target_rank)
            
            if err < 1e-6:
                # Round to nearest integer or half-integer
                def round_to_half(arr):
                    return np.round(arr * 2) / 2
                U_r = round_to_half(U_curr)
                V_r = round_to_half(V_curr)
                W_r = round_to_half(W_curr)
                
                # Verify
                err_rounded = objective_fast(flatten_factors(U_r, V_r, W_r, target_rank), n, m, p, target_rank)
                
                if err_rounded < 1e-6:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists with integer/half-integer entries
    def to_list_with_ints(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Check if it's close to an integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_int) < 1e-6:
                    row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-6:
                    row.append(nearest_half)
                else:
                    # Round to nearest half-integer anyway
                    row.append(round(val * 2) / 2)
            result.append(row)
        return result
    
    U_list = to_list_with_ints(best_U)
    V_list = to_list_with_ints(best_V)
    W_list = to_list_with_ints(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
