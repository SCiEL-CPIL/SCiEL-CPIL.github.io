# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for evaluator overhead

    n, m, p = 3, 5, 6

    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T = np.ones((n, m, p), dtype=np.float64)

    # We look for a rank-r decomposition:
    # T[i,j,k] = sum_a U[i,j,a] * V[j,k,a] * W[k,i,a]
    # U: (n, m, r), V: (m, p, r), W: (p, n, r)

    # Strategy: Try ranks from 89 down to some lower bound.
    # For each rank, try to find a decomposition using gradient descent / alternating minimization.

    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But we know rank >= max(n*m, m*p, p*n) is not tight; the true lower bound is more complex.
    # For 3x5x6, the known best is likely around 60-70.

    # We'll try ranks from 89 down to 50, with limited time per rank.
    
    best_rank = n * m * p  # 90, schoolbook
    best_decomp = None

    # Helper to convert matrices to lists
    def to_list(mat):
        return mat.tolist()

    # Try to find a rank-r decomposition
    def try_rank(r, time_budget=8.0):
        nonlocal deadline
        
        if time.time() > deadline - time_budget:
            return None
        
        # Initialize U, V, W randomly
        np.random.seed(42 + r)
        U = np.random.randn(n, m, r) * 0.5
        V = np.random.randn(m, p, r) * 0.5
        W = np.random.randn(p, n, r) * 0.5
        
        # Alternating minimization
        # For fixed V and W, U[i,j,a] should minimize:
        # sum_{i,j,k} (T[i,j,k] - sum_a U[i,j,a]*V[j,k,a]*W[k,i,a])^2
        # This is a linear system in U[i,j,a] for each (i,j).
        
        # Actually, let's do a simple gradient descent approach.
        # But that's slow. Let's try alternating least squares (ALS).
        
        # For each (i,j), U[i,j,:] is determined by:
        # min_{u} ||T[i,j,:] - u^T * (V[:,:,j,:] * W[:,i,:])||^2
        # Wait, the tensor is T[i,j,k] = sum_a U[i,j,a] V[j,k,a] W[k,i,a]
        # For fixed V,W and given (i,j), we want to find u = U[i,j,:] such that:
        # sum_a u[a] * (V[j,:,a] * W[:,i,a]) ≈ T[i,j,k] for all k
        # This is a least squares problem: A_{k,a} = V[j,k,a] * W[k,i,a]
        # We want u such that A u ≈ T[i,j,:]
        
        # Similarly for V and W.
        
        # Let's implement ALS:
        
        # Precompute nothing, just iterate.
        
        max_iter = 200
        tol = 1e-10
        
        for it in range(max_iter):
            if time.time() > deadline - time_budget + 2:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # A is (p, r): A[k, a] = V[j, k, a] * W[k, i, a]
                    A = V[j, :, :] * W[:, i, :]  # (p, r)
                    b = T[i, j, :]  # (p,)
                    # Solve A^T A u = A^T b
                    try:
                        U[i, j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # A is (n, r): A[i, a] = U[i, j, a] * W[k, i, a]
                    A = U[:, j, :] * W[k, :, :]  # (n, r)
                    b = T[:, j, k]  # (n,)
                    try:
                        V[j, k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # A is (m, r): A[j, a] = U[i, j, a] * V[j, k, a]
                    A = U[i, :, :] * V[:, k, :]  # (m, r)
                    b = T[i, :, k]  # (m,)
                    try:
                        W[k, i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                        err += (val - T[i, j, k]) ** 2
            if err < tol:
                # Found a solution
                # Round to nearest half-integer and check
                U_round = np.round(U * 2) / 2.0
                V_round = np.round(V * 2) / 2.0
                W_round = np.round(W * 2) / 2.0
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_round[i, j, :] * V_round[j, k, :] * W_round[k, i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to required format
                    U_mat = np.zeros((n*m, r))
                    V_mat = np.zeros((m*p, r))
                    W_mat = np.zeros((p*n, r))
                    for i in range(n):
                        for j in range(m):
                            U_mat[i*m+j, :] = U_round[i, j, :]
                    for j in range(m):
                        for k in range(p):
                            V_mat[j*p+k, :] = V_round[j, k, :]
                    for k in range(p):
                        for i in range(n):
                            W_mat[k*n+i, :] = W_round[k, i, :]
                    return {"U": to_list(U_mat), "V": to_list(V_mat), "W": to_list(W_mat)}
        
        return None

    # Try ranks from high to low
    for r in range(89, 49, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            best_decomp = result
            best_rank = r
            break
    
    # If no better found, return schoolbook
    if best_decomp is None:
        r = n * m * p  # 90
        U_mat = np.zeros((n*m, r))
        V_mat = np.zeros((m*p, r))
        W_mat = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_mat[i*m+j, col] = 1
                    V_mat[j*p+k, col] = 1
                    W_mat[k*n+i, col] = 1
                    col += 1
        best_decomp = {"U": to_list(U_mat), "V": to_list(V_mat), "W": to_list(W_mat)}
    
    return best_decomp
# EVOLVE-BLOCK-END
