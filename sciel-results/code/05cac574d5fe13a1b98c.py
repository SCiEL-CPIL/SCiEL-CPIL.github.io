"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 150s
    
    # Index mappings
    # U index: i*m + j, where i in [0,n), j in [0,m)
    # V index: j*p + k, where j in [0,m), k in [0,p)
    # W index: k*n + i, where k in [0,p), i in [0,n)
    
    # The tensor equation: for all (i,j,k) in [0,n)x[0,m)x[0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    best_rank = n * m * p  # schoolbook = 84
    best_solution = None
    
    # Generate schoolbook solution as baseline
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    for col in range(best_rank):
        i = col // (m*p)
        j = (col // p) % m
        k = col % p
        U_sb[i*m+j, col] = 1.0
        V_sb[j*p+k, col] = 1.0
        W_sb[k*n+i, col] = 1.0
    best_solution = (U_sb, V_sb, W_sb)
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if the solution is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def is_valid_half_integer(arr, tol=1e-9):
        """Check if all entries are integers or half-integers."""
        for val in arr.flatten():
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True

    def als_decompose(rank, max_iter=200, tol=1e-12, init_type='random'):
        """
        Find U, V, W of given rank such that
        sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        """
        if init_type == 'random':
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
        elif init_type == 'integer':
            U = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, rank)).astype(float)
        elif init_type == 'structured':
            # Initialize with structured patterns that might help convergence
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for c in range(rank):
                # Spread out the indices
                i_idx = (c * 7) % n
                j_idx = (c * 11) % m
                k_idx = (c * 13) % p
                U[i_idx*m + j_idx, c] = 1.0
                V[j_idx*p + k_idx, c] = 1.0
                W[k_idx*n + i_idx, c] = 1.0
        else:
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
        
        # Ensure no column is all zeros
        for c in range(rank):
            if np.allclose(U[:, c], 0):
                U[np.random.randint(n*m), c] = 1.0
            if np.allclose(V[:, c], 0):
                V[np.random.randint(m*p), c] = 1.0
            if np.allclose(W[:, c], 0):
                W[np.random.randint(p*n), c] = 1.0
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[i*m+j, :], residuals, rank_A, sv = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V_new[j*p+k, :], residuals, rank_A, sv = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((rank, m))
                    for j in range(m):
                        A[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W_new[k*n+i, :], residuals, rank_A, sv = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Check convergence
            if check_solution(U, V, W, tol=1e-6):
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                    return U_snap, V_snap, W_snap
        
        # Final check
        if check_solution(U, V, W, tol=1e-8):
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                return U_snap, V_snap, W_snap
        
        return None

    def gradient_refine(U, V, W, lr=0.01, max_iter=100):
        """Gradient descent refinement after ALS."""
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dc of (U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])
                        # = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1
                        for c in range(U.shape[1]):
                            prod = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = prod - 1.0
                            U_grad[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                            V_grad[j*p+k, c] += err * U[i*m+j, c] * W[k*n+i, c]
                            W_grad[k*n+i, c] += err * U[i*m+j, c] * V[j*p+k, c]
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            if check_solution(U, V, W, tol=1e-6):
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_solution(U_snap, V_snap, W_snap, tol=1e-8):
                    return U_snap, V_snap, W_snap
        
        return None

    # Try ranks from lower bound up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = 28
    min_rank_to_try = max(n*m, m*p, p*n)
    
    # Use most of the budget
    init_types = ['random', 'integer', 'structured']
    
    for rank in range(min_rank_to_try, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for init_type in init_types:
            if time.time() > deadline - 5:
                break
            
            # Multiple restarts per init type
            for restart in range(15):
                if time.time() > deadline - 3:
                    break
                
                # Use a different seed for each restart
                np.random.seed(hash((n,m,p,rank,restart,init_type)) % 2**32)
                
                result = als_decompose(rank, max_iter=150, init_type=init_type)
                if result is not None:
                    U, V, W = result
                    best_rank = rank
                    best_solution = (U, V, W)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            # Try gradient refinement to see if we can push lower
            if rank > min_rank_to_try and time.time() < deadline - 20:
                for lower_rank in range(min_rank_to_try, rank):
                    if time.time() > deadline - 10:
                        break
                    # Try to refine at lower rank using current as init
                    U_init, V_init, W_init = best_solution
                    # Can't directly use higher rank init for lower rank, skip for now
                    break
            
            break
    
    # If no better solution found, use schoolbook
    U, V, W = best_solution
    
    # Final safety check: ensure returned values are valid
    if not check_solution(U, V, W, tol=1e-6):
        # Fall back to schoolbook
        U = U_sb
        V = V_sb
        W = W_sb
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
