"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 35s safety margin before 150s kill
    
    # Index mappings
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    best_rank = n * m * p  # schoolbook 84
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
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def is_valid_half_integer(arr, tol=1e-9):
        for val in arr.flatten():
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True

    def als_decompose(rank, max_iter=100, tol=1e-10, seed=0):
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((rank, p))
                    for k in range(p):
                        A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U_new[i*m+j, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
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
                        V_new[j*p+k, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
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
                        W_new[k*n+i, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Check convergence
            if check_solution(U, V, W, tol=1e-6):
                # Snap to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if check_solution(U_r, V_r, W_r, tol=1e-8):
                    return U_r, V_r, W_r
        
        # Final check
        if check_solution(U, V, W, tol=1e-8):
             U_r = np.round(U * 2) / 2
             V_r = np.round(V * 2) / 2
             W_r = np.round(W * 2) / 2
             if check_solution(U_r, V_r, W_r, tol=1e-8):
                 return U_r, V_r, W_r
        return None

    def sparsity_search(rank, max_patterns=1000):
        """Search for sparse solutions by trying different support patterns."""
        # For each rank-1 term, we need to choose which (i,j), (j,k), (k,i) positions are non-zero
        # This is a complex search space, so we use a heuristic approach
        
        for pattern_seed in range(max_patterns):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(pattern_seed)
            
            # Randomly assign non-zero entries for each rank-1 term
            U_pattern = np.zeros((n*m, rank))
            V_pattern = np.zeros((m*p, rank))
            W_pattern = np.zeros((p*n, rank))
            
            for r in range(rank):
                # Choose a few random non-zero positions
                u_nonzero = np.random.choice(n*m, size=max(1, n*m//4), replace=False)
                v_nonzero = np.random.choice(m*p, size=max(1, m*p//4), replace=False)
                w_nonzero = np.random.choice(p*n, size=max(1, p*n//4), replace=False)
                
                for idx in u_nonzero:
                    U_pattern[idx, r] = 1.0
                for idx in v_nonzero:
                    V_pattern[idx, r] = 1.0
                for idx in w_nonzero:
                    W_pattern[idx, r] = 1.0
            
            # Try to solve for coefficients with this sparsity pattern
            # This is still complex, so we use ALS with this initialization
            U = U_pattern * 0.5
            V = V_pattern * 0.5
            W = W_pattern * 0.5
            
            # Run a few ALS iterations
            for iteration in range(20):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((rank, p))
                        for k in range(p):
                            A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U_new[i*m+j, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
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
                            V_new[j*p+k, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(n), rcond=None)
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
                            W_new[k*n+i, :], _, _, _ = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                        except:
                            W_new[k*n+i, :] = W[k*n+i, :]
                W = W_new
                
                if check_solution(U, V, W, tol=1e-6):
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if check_solution(U_r, V_r, W_r, tol=1e-8):
                        return U_r, V_r, W_r
        
        return None

    # Search ranks from low to high
    # Start from 60 to 84
    for rank in range(60, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # First try ALS
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            result = als_decompose(rank, max_iter=50, seed=rank * 100 + restart)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_solution = (U, V, W)
                found = True
                break
        
        # If ALS failed, try sparsity-based search
        if not found and time.time() < deadline - 15:
            result = sparsity_search(rank, max_patterns=200)
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_solution = (U, V, W)
                found = True
        
        if found:
            break

    U, V, W = best_solution
    
    if not check_solution(U, V, W, tol=1e-6):
        U = U_sb
        V = V_sb
        W = W_sb

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
