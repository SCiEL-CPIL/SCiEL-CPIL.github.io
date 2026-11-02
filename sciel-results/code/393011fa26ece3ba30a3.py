import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]
DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the target tensor: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T is 1 for consistent (i,j,k) triples, 0 otherwise
    # We only need to check the n*m*p = 42 entries where T=1
    # For all other entries, T=0, but we don't need to enumerate them explicitly
    # in the constraint system if we only enforce the 1s and verify at the end
    
    # First, get the schoolbook rank-42 solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    best_U, best_V, best_W, best_rank = U_fb, V_fb, W_fb, rank_fb
    
    # Try to find lower rank decompositions
    # Strategy: For each candidate rank R < 42, try to find U, V, W
    # such that for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of 42 polynomial equations in R*(n*m + m*p + p*n) variables
    
    # Use randomized local search:
    # 1. Initialize U, V, W randomly
    # 2. Compute residuals for the 42 constraints
    # 3. Use gradient-free optimization or random perturbations
    
    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the tensor"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(R, max_iter=5000):
        """Try to find a rank-R decomposition using randomized search"""
        if R >= best_rank:
            return None
        
        # Number of constraints: n*m*p = 42
        # Variables: R*(n*m + m*p + p*n)
        
        # Initialize with small random values
        np.random.seed(R)  # Deterministic for reproducibility
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # Precompute the index triples
        triples = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            # Compute residuals
            residuals = np.zeros(len(triples))
            for idx, (a, b, c) in enumerate(triples):
                residuals[idx] = np.sum(U[a, :] * V[b, :] * W[c, :]) - 1.0
            
            res_norm = np.linalg.norm(residuals)
            if res_norm < best_residual:
                best_residual = res_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res_norm < 1e-8:
                return best_U, best_V, best_W
            
            # Random perturbation: pick a random column r and random matrices to perturb
            r = np.random.randint(R)
            lr = np.random.uniform(0.01, 0.1)
            
            # Perturb one of U, V, W
            choice = np.random.randint(3)
            if choice == 0:
                row = np.random.randint(n * m)
                U[row, r] += lr * np.random.randn()
            elif choice == 1:
                row = np.random.randint(m * p)
                V[row, r] += lr * np.random.randn()
            else:
                row = np.random.randint(p * n)
                W[row, r] += lr * np.random.randn()
        
        # Check if we found a valid solution
        if best_residual < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from 41 down to 21 (lower bound)
    for R in range(rank_fb - 1, 20, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_rank(R)
        if result is not None:
            U_try, V_try, W_try = result
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, R
                # Found a valid solution, continue trying lower ranks
                continue
        else:
            # No valid solution found for this rank, try next
            continue
    
    # Return the best solution found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
